#!/usr/bin/env python
import logging
import argparse
import numpy as np
import os
import gym
import d4rl
from mjrl.utils.gym_env import GymEnv
from mjrl.utils import tensor_utils
logging.disable(logging.CRITICAL)
import torch
import pickle


parser = argparse.ArgumentParser(description='Convert dataset from d4rl format to paths.')
parser.add_argument('--env_name', type=str, required=True, help='environment ID')
parser.add_argument('--include', type=str, required=True, help='a file to include (can contain imports and function definitions)')
parser.add_argument('--seed', type=int, default=123, help='random seed for sampling')

args = parser.parse_args()
SEED = args.seed
e = GymEnv(args.env_name)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
e.set_seed(SEED)
e.action_space.seed(SEED)

import sys
splits = args.include.split("/")
dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
for x in splits[:-1]: dirpath = dirpath + "/" + x
filename = splits[-1].split(".")[0]
sys.path.append(dirpath)
exec("from "+filename+" import *")
if 'obs_mask' in globals(): e.obs_mask = obs_mask

#load dataset
data_file = os.path.join('../../datasets', args.env_name+'.pickle')
dataset_paths = pickle.load(open(data_file, 'rb'))
for path in dataset_paths:
    path_new = {}
    path_new['observations'] = torch.tensor(path['observations']).unsqueeze(0).unsqueeze(0)
    print(path_new['observations'].shape)
    path_new['actions'] = torch.tensor(path['actions']).unsqueeze(0).unsqueeze(0)
    path_new = reward_function(path_new)
    print(path['rewards'])
    print(path_new['rewards'])
    input('....')


#run a random policy in environment for a few episodes
num_episodes = 10

paths = []
for ep in range(num_episodes):
    e.reset()
    observations = []
    actions = []
    rewards = []
    env_infos = []
    t = 0
    done = False
    while t < e.horizon and done is False:
        o = e.get_obs()
        ifo = e.get_env_infos()
        a = e.action_space.sample()
        # pos_before = e.env.sim.data.qpos[0]
        next_o, r, done, ifo2 = e.step(a)
        # pos_after = e.env.sim.data.qpos[0]
        # print('vel_from_pos', (pos_after - pos_before) / e.env.dt)
        # print('qvel', e.env.sim.data.qvel[0])
        # input('....')

        ifo = ifo2 if ifo == {} else ifo
        t = t + 1
        observations.append(next_o)
        actions.append(a)
        rewards.append(r)
        env_infos.append(ifo)

    path = dict(observations=torch.tensor(np.array(observations)), actions=torch.tensor(np.array(actions)),
                rewards=torch.tensor(np.array(rewards)),
                env_infos=tensor_utils.stack_tensor_dict_list(env_infos))
    paths.append(path)


# stacked_rewards = torch.cat([path['rewards'] for path in paths], axis=0).unsqueeze(0).unsqueeze(0)
# stacked_obs = torch.cat([path['observations'] for path in paths], axis=0).unsqueeze(0).unsqueeze(0)
# stacked_actions = torch.cat([path['actions'] for path in paths], axis=0).unsqueeze(0).unsqueeze(0)
# stacked_paths=dict(observations=stacked_obs, actions=stacked_actions)
# stacked_paths = reward_function(stacked_paths)

for path in paths:
    path_new = {}
    path_new['observations'] = path['observations'].unsqueeze(0).unsqueeze(0)
    path_new['actions'] = path['actions'].unsqueeze(0).unsqueeze(0)
    path_new = reward_function(path_new)
    # print(path['rewards'])
    # print(path_new['rewards'])
    # input('....')







print(stacked_paths['rewards'], stacked_rewards)
print(torch.equal(stacked_paths['rewards'], stacked_rewards))
print(torch.argmax(torch.abs(stacked_paths['rewards'] - stacked_rewards)))