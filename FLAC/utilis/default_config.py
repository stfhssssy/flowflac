from utilis.config import Config
# test tau and batchsizie

default_config = Config({
    "seed": 0,
    "tag": "default",
    "start_steps": 5000,
    "cuda": True,
    "num_steps": 1000001,
    "save": True,
    
    "eval": True,

    "eval_numsteps": 10000,
    "eval_times": 5,
    "replay_size": 1000000,

    "algo": "FlowAC",
    "policy": "Flow", 
    "steps": 1,
    "gamma": 0.99, 
    "tau": 0.1,
    "lr": 0.0003, #0.0003
    "batch_size": 256, 
    "updates_per_step": 1,
    "target_update_interval": 2, # for delayed policy update and target network update
    "hidden_size": 512,

    "normalize_obs": True,
    "obs_norm_clip": 10.0,
    "obs_norm_eps": 1e-8,

    "distributional_critic": False,
    "critic_num_atoms": 101,
    "critic_v_min": -10.0,
    "critic_v_max": 150.0,

})
