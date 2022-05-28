"""
Script to convert D4RL dataset into MJRL format
"""

import os
import numpy as np
import pickle
import argparse
import torch
import mjrl.envs
import gym
import d4rl
import mjrl.samplers.core as sampler
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.tensor_utils import d4rl2paths

# ===============================================================================
# Get command line arguments
# ===============================================================================


def prep_d4rl_dataset(*,
                      env_name,
                      output='datasets',
                      act_repeat=1,
                      include=None,
                      header=None,
                      seed=123):

    if header: exec(header)
    SEED = seed
    e = GymEnv(env_name)
    act_repeat = act_repeat
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    e.set_seed(SEED)

    if include:
        import sys
        splits = include.split("/")
        dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
        for x in splits[:-1]: dirpath = dirpath + "/" + x
        filename = splits[-1].split(".")[0]
        sys.path.append(dirpath)
        exec("from "+filename+" import *")
    if 'obs_mask' in globals(): e.obs_mask = obs_mask

    dataset = e.env.env.get_dataset()
    raw_paths = d4rl2paths(dataset)

    # print some statistics
    returns = np.array([np.sum(p['rewards']) for p in raw_paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in raw_paths])
    print("Number of samples collected = %i" % num_samples)
    print("Collected trajectory return mean, std, min, max = %.2f , %.2f , %.2f, %.2f" % \
        (np.mean(returns), np.std(returns), np.min(returns), np.max(returns)) )

    # prepare trajectory dataset (scaling, transforms etc.)
    paths = []
    for p in raw_paths:
        path = dict()
        raw_obs = p['observations']
        raw_act = p['actions']
        raw_rew = p['rewards']
        traj_length = raw_obs.shape[0]
        obs = e.obs_mask * raw_obs[::act_repeat]
        act = np.array([np.mean(raw_act[i * act_repeat : (i+1) * act_repeat], axis=0) for i in range(traj_length // act_repeat)])
        rew = np.array([np.sum(raw_rew[i * act_repeat : (i+1) * act_repeat]) for i in range(traj_length // act_repeat)])
        path['observations'] = obs
        path['actions'] = act
        path['rewards'] = rew
        paths.append(path)


    output_dir = output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    pickle.dump(paths, open(os.path.join(output_dir,env_name+'.pickle'), 'wb'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Convert dataset from d4rl format to paths.')
    parser.add_argument('--env_name', type=str, required=True, help='environment ID')
    parser.add_argument('--output', type=str,  help='location to store data')
    parser.add_argument('--act_repeat', type=int,  help='action repeat, will average actions over the repeat')
    parser.add_argument('--include', type=str,  help='a file to include (can contain imports and function definitions)')
    parser.add_argument('--header', type=str, help='header commands to execute (can include imports)')
    parser.add_argument('--seed', type=int,  help='random seed for sampling')
    args = { k:v for k,v in vars(parser.parse_args()).items() if v is not None }
    prep_d4rl_dataset(**args)