"""
Collect data using an RL policy and store
"""


"""
Script to collect offline dataset from a logging policy
"""


# ===============================================================================
# Get command line arguments
# ===============================================================================
import os
import numpy as np
import pickle
import argparse
import torch
import mjrl.envs
import gym
import mjrl.samplers.core as sampler
from mjrl.algos.mbrl.sampling import evaluate_policy
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.utils import import_from_path

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


class RandomPolicy():
    def __init__(self, action_lows, action_highs):
        self.action_highs = action_highs
        self.action_lows = action_lows

    def get_action(self, observation):
        return np.random.uniform(self.action_lows, self.action_highs)        


def collect_dataset(*,
                    env_name,
                    policy='random',
                    output='datasets',
                    num_samples=1e5,
                    act_repeat=1,
                    include=None,
                    header=None,
                    seed=123,
                    visualize=False):

    if header:
        exec(header)
    SEED = seed
    e = GymEnv(env_name)
    act_repeat = act_repeat
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    e.set_seed(SEED)
    num_episodes = int(num_samples // e.env.spec.max_episode_steps)
    if include:
        mod = import_from_path(include, base_path=FILE_PATH)
        if 'obs_mask' in vars(mod):
            e.obs_mask = mod.obs_mask

    if policy == 'random':
        policy = RandomPolicy(e.action_space.low, e.action_space.high)
    else:   policy = pickle.load(open(policy, 'rb'))
    # raw_paths = sampler.sample_data_batch(num_samples=num_samples, env=e, policy=policy, eval_mode=False,
                                        #   base_seed=SEED, num_cpu='max', paths_per_call='max')
    raw_paths = evaluate_policy(e, policy, None, noise_level=0.0, real_step=True,
                                 num_episodes=num_episodes, visualize=visualize)

    # print some statistics
    returns = np.array([np.sum(p['rewards']) for p in raw_paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in raw_paths])
    try:
        success_metric = e.env.env.evaluate_success(raw_paths)
    except:
        success_metric = 0.0
    print("Number of samples collected = %i" % num_samples)
    print("Collected trajectory return mean, std, min, max = %.2f , %.2f , %.2f, %.2f" %
          (np.mean(returns), np.std(returns), np.min(returns), np.max(returns)))
    print("Success metric = {}".format(success_metric))
    # prepare trajectory dataset (scaling, transforms etc.)
    paths = []
    for p in raw_paths:
        path = dict()
        raw_obs = p['observations']
        raw_act = p['actions']
        raw_rew = p['rewards']
        traj_length = raw_obs.shape[0]
        obs = e.obs_mask * raw_obs[::act_repeat]
        act = np.array([np.mean(raw_act[i * act_repeat: (i+1) * act_repeat], axis=0)
                       for i in range(traj_length // act_repeat)])
        rew = np.array([np.sum(raw_rew[i * act_repeat: (i+1) * act_repeat])
                       for i in range(traj_length // act_repeat)])
        path['observations'] = obs
        path['actions'] = act
        path['rewards'] = rew
        paths.append(path)

    output_dir = output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    pickle.dump(paths, open(os.path.join(
        output_dir, env_name+'.pickle'), 'wb'))
    
    return paths


if __name__ == '__main__':

    # ===============================================================================
    # Get command line arguments
    # ===============================================================================

    parser = argparse.ArgumentParser(
        description='Convert dataset from d4rl format to paths.')
    parser.add_argument('--env_name', type=str,
                        required=True, help='environment ID')
    parser.add_argument('--output', type=str,  help='location to store data')
    parser.add_argument('--policy', type=str,
                        help='location of policy file for data collection')
    parser.add_argument('--num_samples', type=int, default=1e5, 
                        help='number of samples to collect')
    parser.add_argument('--act_repeat', type=int,
                        help='action repeat, will average actions over the repeat')
    parser.add_argument('--include', type=str,
                        help='a file to include (can contain imports and function definitions)')
    parser.add_argument('--header', type=str,
                        help='header commands to execute (can include imports)')
    parser.add_argument('--seed', type=int,  help='random seed for sampling')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize env while data collection')

    args = {k: v for k, v in vars(
        parser.parse_args()).items() if v is not None}
    
    collect_dataset(**args)
