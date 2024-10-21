"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch

MAX_EPISODE_LEN = 8760

from wrappers_custom import *
from utils_.helpers import *

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import *

def create_eval_episodes_fn(eval_rtg,
                            state_dim,
                            act_dim,
                            state_mean, 
                            state_std, 
                            device, 
                            use_mean = False, 
                            reward_scale = 1 ,
                            schema = "citylearn_challenge_2022_phase_2"):

    def eval_episodes_fn(model):
        target_return = eval_rtg * reward_scale
        returns,lengths,_,df_evaluate = evaluate_episode_rtg(state_dim,act_dim,model, target_return, MAX_EPISODE_LEN, device = device, use_mean = True,schema=schema)

        return {
                f"evaluation/return": returns,
                f"evaluation/length": lengths,
            },df_evaluate
    return eval_episodes_fn


def create_test_episodes_fn(eval_rtg,
                            state_dim,
                            act_dim,
                            state_mean, 
                            state_std, 
                            device, 
                            use_mean = False, 
                            reward_scale = 1 ,
                            schema = "citylearn_challenge_2022_phase_3"):

    def eval_episodes_fn(model):
        target_return = eval_rtg * reward_scale
        returns,lengths,_,df_evaluate = test_episode_rtg(state_dim,act_dim,model, target_return, MAX_EPISODE_LEN, device = device, use_mean = True,schema=schema)

        return {
                f"evaluation/return": returns,
                f"evaluation/length": lengths,
            },df_evaluate
    return eval_episodes_fn

def create_eval_episodes_fn_ori(eval_rtg,
                            state_dim,
                            act_dim,
                            state_mean, 
                            state_std, 
                            device, 
                            use_mean = False, 
                            reward_scale = 1 ,
                            schema = "citylearn_challenge_2022_phase_2"):

    def eval_episodes_fn(model):
        target_return = eval_rtg * reward_scale
        returns,lengths,_,df_evaluate,data_interaction,df_ts = evaluate_episode_rtg_ori(state_dim,act_dim,model, target_return, MAX_EPISODE_LEN, device = device, use_mean = True,
                                                                                       schema=schema)

        return {
                f"evaluation/return": returns,
                f"evaluation/length": lengths,
            },df_evaluate,data_interaction,df_ts
    return eval_episodes_fn

def create_test_episodes_fn_ori(eval_rtg,
                            state_dim,
                            act_dim,
                            state_mean, 
                            state_std, 
                            device, 
                            use_mean = False, 
                            reward_scale = 1 ,
                            schema = "citylearn_challenge_2022_phase_3"):

    def eval_episodes_fn(model):
        target_return = eval_rtg * reward_scale
        returns,lengths,_,df_evaluate,data_interaction,df_ts = test_episode_rtg_ori(state_dim,act_dim,model, target_return, MAX_EPISODE_LEN, device = device, use_mean = True)

        return {
                f"evaluation/return": returns,
                f"evaluation/length": lengths,
            },df_evaluate,data_interaction,df_ts
    return eval_episodes_fn

@torch.no_grad()
def evaluate_episode_rtg(
    state_dim,
    act_dim,
    model,
    target_return,
    max_ep_len = 8760,
    reward_scale = 1 ,
    state_mean = 0.0,
    state_std = 1.0,
    device = "cuda",
    mode="normal",
    use_mean = False,
    schema="citylearn_challenge_2022_phase_2"
):
    model.eval()
    model.to(device=device)

    state_mean = torch.tensor(state_mean, device=device, dtype=torch.float32)
    state_std = torch.tensor(state_std, device=device, dtype=torch.float32)

    num_envs = 1 

    env = CityLearnEnv(schema=schema)
    env.central_agent = True
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3WrapperCustom(env)

    state,_ = env.reset()

    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)

    actions = torch.zeros(0, device=device, dtype=torch.float64)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    
    ep_return = target_return

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
            num_envs, -1, 1
        )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
            num_envs, -1
        )
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)

    for t in range(max_ep_len):
            # add padding
        actions = torch.cat(
                [
                    actions,
                    torch.zeros((num_envs, act_dim), device=device).reshape(
                        num_envs, -1, act_dim
                    ),
                ],
                dim=1,
            )
        #print(actions)
        rewards = torch.cat(
                [
                    rewards,
                    torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
                ],
                dim=1,
            )
    
        state_pred, action_dist, reward_pred = model.get_predictions(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float64),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                num_envs=num_envs,
            )
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)
    
            # the return action is a SquashNormal distribution
        if use_mean:
            action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
        else:
            action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
       
        action = action.clamp(*model.action_range)
        #print(action.detach().cpu().numpy())
    
        state, reward, done, _,_ = env.step(action.detach().cpu().numpy()[0])
    
            # eval_env.step() will execute the action for all the sub-envs, for those where
            # the episodes have terminated, the envs will be reset. Hence we use
            # "unfinished" to track whether the first episode we roll out for each sub-env is
            # finished. In contrast, "done" only relates to the current episode
        #print(reward)
        episode_return += reward
    
        actions[:, -1] = action
        #print(actions)
        state = (
                torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
            )
        states = torch.cat([states, state], dim=1)
        #reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward
    
        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
                [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
            )
    
        timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                        num_envs, 1
                    )
                    * (t + 1),
                ],
                dim=1,
            )
    
        if t == max_ep_len - 1:
            done = done
            ind = 0
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)
            
        if np.any(done):
            ind = np.where(done)[0]
            #print("ind " + str(ind))
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)
    
        if not np.any(unfinished):
            break


    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
                "observations": states[ii].detach().cpu().numpy()[:ep_len],
                "actions": actions[ii].detach().cpu().numpy()[:ep_len].reshape(-1,5,1),
                "rewards": rewards[ii].detach().cpu().numpy()[:ep_len].reshape(-1,),
                "terminals": terminals,
            }
        trajectories.append(traj)

    df_evaluate = env.env.evaluate()
    

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
        df_evaluate
        
    )





@torch.no_grad()
def test_episode_rtg(
    state_dim,
    act_dim,
    model,
    target_return,
    max_ep_len = 8760,
    reward_scale = 1 ,
    state_mean = 0.0,
    state_std = 1.0,
    device = "cuda",
    mode="normal",
    use_mean = False,
    schema="citylearn_challenge_2022_phase_3"
):
    model.eval()
    model.to(device=device)

    state_mean = torch.tensor(state_mean, device=device, dtype=torch.float32)
    state_std = torch.tensor(state_std, device=device, dtype=torch.float32)

    num_envs = 1 

    env = CityLearnEnv(schema=schema,buildings=[0,2,4,5,6])
    env.central_agent = True
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3WrapperCustom(env)

    state,_ = env.reset()

    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)

    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    
    ep_return = target_return

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
            num_envs, -1, 1
        )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
            num_envs, -1
        )
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)

    for t in range(max_ep_len):
            # add padding
        actions = torch.cat(
                [
                    actions,
                    torch.zeros((num_envs, act_dim), device=device).reshape(
                        num_envs, -1, act_dim
                    ),
                ],
                dim=1,
            )
        #print(actions)
        rewards = torch.cat(
                [
                    rewards,
                    torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
                ],
                dim=1,
            )
    
        state_pred, action_dist, reward_pred = model.get_predictions(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                num_envs=num_envs,
            )
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)
    
            # the return action is a SquashNormal distribution
        if use_mean:
            action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
        else:
            action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
       
        action = action.clamp(*model.action_range)
        #print(action.detach().cpu().numpy())
    
        state, reward, done, _,_ = env.step(action.detach().cpu().numpy()[0])
    
            # eval_env.step() will execute the action for all the sub-envs, for those where
            # the episodes have terminated, the envs will be reset. Hence we use
            # "unfinished" to track whether the first episode we roll out for each sub-env is
            # finished. In contrast, "done" only relates to the current episode
        #print(reward)
        episode_return += reward
    
        actions[:, -1] = action
        #print(actions)
        state = (
                torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
            )
        states = torch.cat([states, state], dim=1)
        #reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward
    
        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
                [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
            )
    
        timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                        num_envs, 1
                    )
                    * (t + 1),
                ],
                dim=1,
            )
    
        if t == max_ep_len - 1:
            done = done
            ind = 0
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)
            
        if np.any(done):
            ind = np.where(done)[0]
            #print("ind " + str(ind))
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)
    
        if not np.any(unfinished):
            break


    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
                "observations": states[ii].detach().cpu().numpy()[:ep_len],
                "actions": actions[ii].detach().cpu().numpy()[:ep_len],
                "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
                "terminals": terminals,
            }
        trajectories.append(traj)

    df_evaluate = env.env.evaluate()
    

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
        df_evaluate
        
    )



@torch.no_grad()
def evaluate_episode_rtg_ori(
    state_dim,
    act_dim,
    model,
    target_return,
    max_ep_len = 8760,
    reward_scale = 1 ,
    state_mean = 0.0,
    state_std = 1.0,
    device = "cuda",
    mode="normal",
    use_mean = False,
    schema="citylearn_challenge_2022_phase_2"
):
    model.eval()
    model.to(device=device)

    state_mean = torch.tensor(state_mean, device=device, dtype=torch.float32)
    state_std = torch.tensor(state_std, device=device, dtype=torch.float32)

    num_envs = 1 

    env = CityLearnEnv(schema=schema)
    env.central_agent = True
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3WrapperCustom(env)

    state,_ = env.reset()

    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)

    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    
    ep_return = target_return

    
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
            num_envs, -1, 1
        )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
            num_envs, -1
        )
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)

    for t in range(max_ep_len):
            # add padding
        actions = torch.cat(
                [
                    actions,
                    torch.zeros((num_envs, act_dim), device=device).reshape(
                        num_envs, -1, act_dim
                    ),
                ],
                dim=1,
            )
        #print(actions)
        rewards = torch.cat(
                [
                    rewards,
                    torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
                ],
                dim=1,
            )
    
        state_pred, action_pred, reward_pred = model.get_predictions(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                num_envs=num_envs,
            )
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)
    
            # the return action is a SquashNormal distribution
       
        action = action_pred

        
       
        action = action.clamp(*model.action_range)
        #print(action.detach().cpu().numpy())
    
        state, reward, done, _,_ = env.step(action.detach().cpu().numpy()[0])
    
            # eval_env.step() will execute the action for all the sub-envs, for those where
            # the episodes have terminated, the envs will be reset. Hence we use
            # "unfinished" to track whether the first episode we roll out for each sub-env is
            # finished. In contrast, "done" only relates to the current episode
        #print(reward)
        episode_return += reward
    
        actions[:, -1] = action
        #print(actions)
        state = (
                torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
            )
        states = torch.cat([states, state], dim=1)
        #reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward
    
        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
                [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
            )
    
        timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                        num_envs, 1
                    )
                    * (t + 1),
                ],
                dim=1,
            )
    
        if t == max_ep_len - 1:
            done = done
            ind = 0
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)
            
        if np.any(done):
            ind = np.where(done)[0]
            #print("ind " + str(ind))
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)
    
        if not np.any(unfinished):
            break


    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
                "observations": states[ii].detach().cpu().numpy()[:ep_len],
                "actions": actions[ii].detach().cpu().numpy()[:ep_len],
                "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
                "terminals": terminals,
            }
        trajectories.append(traj)

    df_evaluate = env.env.evaluate()
    data_interaction = env.dataset
    df_ts = env.env.df_ts_eval

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
        df_evaluate,
        data_interaction,
        df_ts
    )

@torch.no_grad()
def test_episode_rtg_ori(
    state_dim,
    act_dim,
    model,
    target_return,
    max_ep_len = 8760,
    reward_scale = 1 ,
    state_mean = 0.0,
    state_std = 1.0,
    device = "cuda",
    mode="normal",
    use_mean = False,
    schema="citylearn_challenge_2022_phase_3"
):
    model.eval()
    model.to(device=device)

    state_mean = torch.tensor(state_mean, device=device, dtype=torch.float32)
    state_std = torch.tensor(state_std, device=device, dtype=torch.float32)

    num_envs = 1 

    env = CityLearnEnv(schema=schema,buildings=[0,2,4,5,6])
    env.central_agent = True
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3WrapperCustom(env)

    state,_ = env.reset()

    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)

    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    
    ep_return = target_return

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
            num_envs, -1, 1
        )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
            num_envs, -1
        )
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)

    for t in range(max_ep_len):
            # add padding
        actions = torch.cat(
                [
                    actions,
                    torch.zeros((num_envs, act_dim), device=device).reshape(
                        num_envs, -1, act_dim
                    ),
                ],
                dim=1,
            )
        #print(actions)
        rewards = torch.cat(
                [
                    rewards,
                    torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
                ],
                dim=1,
            )
    
        state_pred, action_pred, reward_pred = model.get_predictions(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                num_envs=num_envs,
            )
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)
    
            # the return action is a SquashNormal distribution
       
        action = action_pred

        
       
        action = action.clamp(*model.action_range)
        #print(action.detach().cpu().numpy())
    
        state, reward, done, _,_ = env.step(action.detach().cpu().numpy()[0])
    
            # eval_env.step() will execute the action for all the sub-envs, for those where
            # the episodes have terminated, the envs will be reset. Hence we use
            # "unfinished" to track whether the first episode we roll out for each sub-env is
            # finished. In contrast, "done" only relates to the current episode
        #print(reward)
        episode_return += reward
    
        actions[:, -1] = action
        #print(actions)
        state = (
                torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
            )
        states = torch.cat([states, state], dim=1)
        #reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward
    
        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
                [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
            )
    
        timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                        num_envs, 1
                    )
                    * (t + 1),
                ],
                dim=1,
            )
    
        if t == max_ep_len - 1:
            done = done
            ind = 0
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)
            
        if np.any(done):
            ind = np.where(done)[0]
            #print("ind " + str(ind))
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)
    
        if not np.any(unfinished):
            break


    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
                "observations": states[ii].detach().cpu().numpy()[:ep_len],
                "actions": actions[ii].detach().cpu().numpy()[:ep_len],
                "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
                "terminals": terminals,
            }
        trajectories.append(traj)

    df_evaluate = env.env.evaluate()
    data_interaction = env.dataset
    df_ts = env.env.df_ts_eval

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
        df_evaluate,
        data_interaction,
        df_ts
    )
    