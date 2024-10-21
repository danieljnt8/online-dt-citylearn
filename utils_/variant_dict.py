variant = {
    "seed": 10,
    "env": "hopper-medium-v2",

    # model options
    "K": 20,
    "embed_dim": 128, #512
    "n_layer": 4,
    "n_head": 1, #4
    "activation_function": "relu",
    "dropout": 0.1,
    "eval_context_length": 5,
    "ordering": 0,

    # shared evaluation options
    "eval_rtg": 3600,
    "num_eval_episodes": 10,

    # shared training options
    "init_temperature": 0.1,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "weight_decay": 5e-4,
    "warmup_steps": 10000,

    # pretraining options
    "max_pretrain_iters": 1,
    "num_updates_per_pretrain_iter": 5000,

    # finetuning options
    "max_online_iters": 1500,
    "online_rtg": 7200,
    "num_online_rollouts": 1,
    "replay_size": 1000,
    "num_updates_per_online_iter": 300,
    "eval_interval": 10,

    # environment options
    "device": "cuda",
    "log_to_tb": True,
    "save_dir": "./exp",
    "exp_name": "default"
}
