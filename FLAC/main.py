import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"
import wandb
import torch
import gymnasium as gym
import numpy as np
from utilis.config import ARGConfig
from utilis.default_config import default_config
from model.algo import flowAC
from utilis.Replaybuffer import ReplayMemory
from utilis.video import recorder
import datetime
import itertools
from torch.utils.tensorboard import SummaryWriter
import shutil
from humanoid_bench.env import ROBOTS, TASKS

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True


def evaluation(agent, env, total_numsteps, writer, best_reward, video_path=None):
    avg_reward = 0.
    avg_success = 0.
    if video_path is not None:
        eval_recoder = recorder(video_path)
        eval_recoder.init(f'{total_numsteps}.mp4', enabled=True)
    else:
        eval_recoder = None
    for _  in range(config.eval_times):
        state, _ = env.reset()
        if eval_recoder is not None:
            eval_recoder.record(env.render())
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)

            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            if eval_recoder is not None:
                eval_recoder.record(env.render())
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
        if 'solved' in info.keys():
            avg_success += float(info['solved'])
        elif 'success' in info.keys():
            avg_success += float(info['success'])
    avg_reward /= config.eval_times
    avg_success /= config.eval_times

    if eval_recoder is not None and avg_reward >= best_reward:
        eval_recoder.release('%d_%d.mp4'%(total_numsteps, int(avg_reward)))

    wandb.log({'test/reward': avg_reward,
               "test/avg_success": avg_success}, step=total_numsteps)
    print("----------------------------------------")
    print("Env: {}, Test Episodes: {}, Avg. Reward: {}, Avg. Success: {}".format(config.task, config.eval_times, round(avg_reward, 2), round(avg_success, 2)))
    print("----------------------------------------")
    
    return avg_reward

def train_loop(config, msg = "default"):
    # set seed
    env = gym.make(config.task)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    env.action_space.seed(config.seed)
    np.random.seed(config.seed)

    # Agent
    agent = flowAC(env.observation_space.shape[0], env.action_space, config)

    result_path = './results/{}/{}/{}/{}_{}_{}'.format(config.task, config.algo, msg,
                                                      datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                      config.policy, config.seed)

    checkpoint_path = result_path + '/' + 'checkpoint'
    video_path = result_path + '/eval_video'
    log_filename = result_path + '/' + 'training_log.txt'

    # training logs
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    with open(os.path.join(result_path, "config.log"), 'w') as f:
        f.write(str(config))

    writer = SummaryWriter(result_path)

    # memory
    memory = ReplayMemory(config.replay_size, config.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_reward = -1e6
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, _ = env.reset(seed=config.seed)
        agent.observe(state)

        while not done:
            if config.start_steps > total_numsteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            if config.start_steps <= total_numsteps:
                # Number of updates per step in environment
                for i in range(config.updates_per_step):
                    # Update parameters of all the networks
                    agent.update_parameters(memory, config.batch_size, updates)
                    updates += 1

            next_state, reward, done, truncated, info = env.step(action)
            agent.observe(next_state)
            done = done or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            if "_max_episode_steps" in dir(env):
                mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            else:
                mask = 1 if episode_steps == 1000 else float(not done)

            memory.push(state, action, reward, next_state, mask)
            state = next_state

            # test agent
            if total_numsteps % config.eval_numsteps == 0 and config.eval is True:
                video_path = None
                avg_reward = evaluation(agent, env, total_numsteps, writer, best_reward, video_path)
                if avg_reward >= best_reward and config.save is True:
                    best_reward = avg_reward
                    agent.save_checkpoint(checkpoint_path, 'best')

        if total_numsteps > config.num_steps:
            break

        log_alpha_value = float(agent.log_alpha.detach().cpu().item())
        wandb.log(
            {
                'train/reward': episode_reward,
                'train/log_alpha': log_alpha_value,
                'train/alpha': float(np.exp(log_alpha_value)),
            },
            step=total_numsteps,
        )
        if i_episode % 10 == 0:
            log_message = "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}\n".format(
                    i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
                )
            with open(log_filename, 'a') as log_file:
                log_file.write(log_message)
            print(log_message, end='', flush=True)

    env.close()
    wandb.finish()



if __name__ == "__main__":
    arg = ARGConfig()
    arg.add_arg("task", "h1hand-walk", "Humanoid Bench task name")
    arg.add_arg("device", 0, "Computing device")
    arg.add_arg("algo", "check_opens", "choose algo")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", 0, "experiment seed")
    arg.add_arg("steps", 1, "Flow policy integration steps")
    arg.add_arg("epsilon", 0.0, "random noise for exploration")
    arg.add_arg("normalize_obs", True, "Running mean/std normalization for observations")
    arg.add_arg("obs_norm_clip", 10.0, "Observation normalization clip value")
    arg.add_arg("obs_norm_eps", 1e-8, "Observation normalization epsilon")
    arg.add_arg("distributional_critic", True, "Use C51-style distributional critic")
    arg.add_arg("critic_num_atoms", 101, "C51 number of atoms")
    arg.add_arg("critic_v_min", -10, "C51 value support min")
    arg.add_arg("critic_v_max", 150.0, "C51 value support max")
    # LAC hyperparameters
    arg.add_arg("target_kinetic_coef", 2.5, "LAC: Target kinetic energy coefficient (target = coef * action_dim)")
    arg.add_arg("init_log_alpha", -2, "LAC: Initial log(alpha) value for temperature parameter")
    arg.add_arg("auto_alpha", True, "LAC: Enable automatic alpha tuning")
    arg.parser()

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Torch CUDA Available:", torch.cuda.is_available())

    config = default_config
    config.update(arg)

    wandb.init(
        project = config.algo,
        name = config.task + "_coef" + str(config.target_kinetic_coef) + "_seed" + str(config.seed) + "_step" +str(config.steps)+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        config = config
    )
    print(f">>>> Training {config.algo} on {config.task} environment, on {config.device}")
    train_loop(config, msg=config.tag)
