"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import gym

import torch
import numpy as np
import pandas as pd

from datasets import load_from_disk
import datasets

import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
#from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
#from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from evaluation_citylearn import create_eval_episodes_fn,evaluate_episode_rtg,create_test_episodes_fn
from trainer import SequenceTrainer#, SequenceTrainerCustom
from logger import Logger
from wrappers_custom import *
from utils_.helpers import *

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import *
from utils_.variant_dict import variant

MAX_EPISODE_LEN = 8760

def update_loss_csv(iter_value, loss, filename='loss_per_epoch.csv',type_name="Epoch"):
    # Try to read the existing CSV file
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        # If file does not exist, create a new DataFrame with headers
        df = pd.DataFrame(columns=[type_name, 'Loss'])
    
    # Append the new data to the DataFrame
    new_row = {type_name: iter_value, 'Loss': loss}
    df = df.append(new_row, ignore_index=True)
    
    # Write the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)


class Experiment:
    def __init__(self, variant,dataset_path):

        env = CityLearnEnv(schema="citylearn_challenge_2022_phase_1")
        env.central_agent = True
        env = NormalizedObservationWrapper(env)
        env = StableBaselines3WrapperCustom(env)

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(env)
        
        self.initial_trajectories = self._get_initial_trajectories(dataset_path=dataset_path)



        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            self.initial_trajectories
        )
        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        self.aug_trajs = []

        self.device = variant["device"] #variant.get("device", "cuda:0")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=8760,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_ctx = 72,  # because K = 24
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 
        self.logger = Logger(variant)
    
    def _get_initial_trajectories(self,dataset_path):
        dataset = load_from_disk(dataset_path)
        dataset,_ = segment_v2(dataset["observations"],dataset["actions"],dataset["rewards"],dataset["dones"])
        trajectories = datasets.Dataset.from_dict({k: [s[k] for s in dataset] for k in dataset[0].keys()})

        return trajectories
   
    def _get_env_spec(self,env):
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
                -1,#float(env.action_space.low.min()) ,
                1#float(env.action_space.high.max()) ,
            ]
        return state_dim,act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _augment_trajectories(
        self,
        target_explore,
        n,
        online_schema = "citylearn_challenge_2022_phase_1",
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = target_explore * self.reward_scale 

            returns, lengths, trajs ,_= evaluate_episode_rtg(
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
                schema=online_schema
            )

        for traj in trajs:
            if np.isnan(traj["rewards"]).any():
                print("NaNs found in rewards")
            if np.isnan(traj["observations"]).any():
                print("NaNs found in states")
            if np.isnan(traj["actions"]).any():
                print("NaNs found in actions")
            if np.isnan(traj["terminals"]).any():
                print("NaNs found in terminals")
        
        predicted_actions = trajs[0]["actions"].reshape(-1,5,1)
        modified_actions = predicted_actions + 0.000001

        # Clip the values to be within the range [-0.99, 0.99]
        clipped_actions = np.clip(modified_actions, -0.99, 0.99)


        small_trajs = [{
            'observations': trajs[0]["observations"].reshape(-1,44),  # Dummy observations
            'actions': clipped_actions,  # Dummy actions
            'rewards': trajs[0]["rewards"],  # Dummy rewards
            'dones': trajs[0]["terminals"]  # Dummy dones
        }]
        self.replay_buffer.add_new_trajs(small_trajs)

        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def _save_model_online_tuning(self, path_prefix, iter):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model_iter_{iter}.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model_iter_{iter}.pt")

        

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self,trajectories):
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(np.array(path["rewards"]).sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

            # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: city_learn")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        print(sorted_inds)
        #print(trajectories[1])
        for ii in sorted_inds:
            print(ii)
        #print(trajectories[0].keys())
        trajectories = [trajectories[int(ii)] for ii in sorted_inds]

        for trajectory in trajectories:
            for key in trajectory.keys():
                trajectory[key] = np.array(trajectory[key])


        return trajectories, state_mean, state_std

   

    def pretrain(self, loss_fn, schema_eval = "citylearn_challenge_2022_phase_2"):
        print("\n\n\n*** Pretrain ***")

        """

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        """
        train_fn = create_eval_episodes_fn(self.variant["eval_rtg"],self.state_dim,self.act_dim,self.state_mean,self.state_std,self.device,
                                          use_mean = True,schema="citylearn_challenge_2022_phase_1")
        eval_fn = create_eval_episodes_fn(self.variant["eval_rtg"],self.state_dim,self.act_dim,self.state_mean,self.state_std,self.device,
                                          use_mean = True,schema= "citylearn_challenge_2022_phase_2")
        
        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

       
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )

            update_loss_csv(iter_value = self.pretrain_iter , loss = train_outputs["training/train_loss_mean"],filename=self.logger.log_path+"/pretrain_loss.csv",type_name ="Epoch")
            update_loss_csv(iter_value = self.pretrain_iter , loss = train_outputs["training/nll"],filename=self.logger.log_path+"/pretrain_nll_loss.csv",type_name ="Epoch")
            update_loss_csv(iter_value = self.pretrain_iter , loss = train_outputs["training/entropy"],filename=self.logger.log_path+"/pretrain_entropy.csv",type_name ="Epoch")


            train_outputs, train_reward, df_train = self.evaluate(train_fn)
            eval_outputs, eval_reward, df_evaluate = self.evaluate(eval_fn)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )
            df_train.to_csv(self.logger.log_path+f"/train_pretrain_iter_{self.pretrain_iter}.csv")
            df_evaluate.to_csv(self.logger.log_path+f"/eval_pretrain_iter_{self.pretrain_iter}.csv")

            self.pretrain_iter += 1

    
    
    def evaluate(self,eval_fn):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        
        o,df_evaluate = eval_fn(self.model)
        outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return"]
        return outputs, eval_reward,df_evaluate

    def online_tuning(self,  online_schema, schema_eval, loss_fn):

        print("\n\n\n*** Online Finetuning ***")

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        train_fn = create_eval_episodes_fn(self.variant["eval_rtg"],self.state_dim,self.act_dim,self.state_mean,self.state_std,self.device,
                                          use_mean = True,schema="citylearn_challenge_2022_phase_1")
        eval_fn = create_eval_episodes_fn(self.variant["eval_rtg"],self.state_dim,self.act_dim,self.state_mean,self.state_std,self.device,
                                          use_mean = True,schema= "citylearn_challenge_2022_phase_2")
        test_fn = create_test_episodes_fn(self.variant["eval_rtg"],self.state_dim,self.act_dim,self.state_mean,self.state_std,self.device,
                                          use_mean = True,schema= "citylearn_challenge_2022_phase_3")
        
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )

        
        while self.online_iter < self.variant["max_online_iters"]:

            outputs = {}
            
            augment_outputs = self._augment_trajectories(
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
                online_schema=online_schema,
                
            )
            

            #print(self.replay_buffer.trajectories[0]["dones"].dtype)
            #print(self.replay_buffer.trajectories[2]["dones"].dtype)  
            #print(self.replay_buffer.trajectories[1]["rewards"].shape)      
                

            #outputs.update(augment_outputs)
            #print(len(self.replay_buffer.trajectories))
            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )

            update_loss_csv(iter_value = self.online_iter , loss = train_outputs["training/train_loss_mean"],filename=self.logger.log_path+"/online_loss.csv",type_name ="Epoch")
            update_loss_csv(iter_value = self.online_iter , loss = train_outputs["training/nll"],filename=self.logger.log_path+"/online_nll_loss.csv",type_name ="Epoch")
            update_loss_csv(iter_value = self.online_iter , loss = train_outputs["training/entropy"],filename=self.logger.log_path+"/online_entropy.csv",type_name ="Epoch")

            outputs.update(train_outputs)

            if evaluation:
                train_outputs, train_reward, df_train = self.evaluate(train_fn)
                eval_outputs, eval_reward, df_evaluate = self.evaluate(eval_fn)
                

                df_evaluate.to_csv(self.logger.log_path+f"/eval_online_iter_{self.online_iter}.csv")
                df_train.to_csv(self.logger.log_path+f"/train_online_iter_{self.online_iter}.csv")

                self._save_model_online_tuning(
                path_prefix=self.logger.log_path,
                iter = self.online_iter
                )


            if is_last_iter:
                test_outputs, test_reward, df_test = self.evaluate(test_fn)
                df_test.to_csv(self.logger.log_path+f"/test_FINAL.csv")
                

            outputs["time/total"] = time.time() - self.start_time



            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            

            

            self.online_iter += 1

    def __call__(self):

        #utils.set_seed_everywhere(args.seed)

       

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            #print(a)
            #size = (20, 24)

# Create an attention mask with all values set to 1
            
            #print("==================")
            #attention_mask = torch.ones(size)
            #print(a_hat_dist.log_likelihood(a))
            a_hat_dist_log = torch.nan_to_num(a_hat_dist.log_likelihood(a))
            log_likelihood = a_hat_dist_log[attention_mask > 0].mean()
            #print("Log Likelihood " + str(log_likelihood))
            #print(log_likelihood)

            

            entropy = a_hat_dist.entropy().mean()
            #print("Entropy "+str(entropy))
            #print("Entropy Reg "+str(entropy_reg))
            
            loss = -(log_likelihood + entropy_reg * entropy)

            #print("Final loss "+str(loss))

            return (
                loss,
                -log_likelihood,
                entropy,
            )

       

        print("\n\nMaking Eval Env.....")
        
        eval_env_schema = "citylearn_challenge_2022_phase_2"

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(loss_fn, schema_eval=eval_env_schema)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            

            online_schema = "citylearn_challenge_2022_phase_1"
            schema_eval = "citylearn_challenge_2022_phase_2"
            #online_schema = "citylearn_challenge_2022_phase_1"

            self.online_tuning(online_schema,schema_eval, loss_fn)
            #online_env.close()

        #eval_envs.close()

def experiment(seed):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=seed)
    #parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument("--K", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=20)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=-9000)
    parser.add_argument("--num_eval_episodes", type=int, default=1)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=200) #30

    
    # finetuning options

    parser.add_argument("--max_online_iters", type=int, default=200) #10
    parser.add_argument("--online_rtg", type=int, default=-9000)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default= 60) #20
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cuda") ##cuda 
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args),dataset_path="data_interactions/model_RBCAgent1_timesteps_8760_rf_CombinedReward_phase_1_new.pkl")
    #experiment = Experiment(vars(args),dataset_path="data_interactions/sac_dataset.pkl")
    print("=" * 50)
    experiment()

if __name__ == "__main__":
    seeds = [10,100,200,1024,2048]

    for seed in seeds:
        experiment(seed)
