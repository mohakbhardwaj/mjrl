#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import argparse
from copy import deepcopy
from datetime import datetime
import gym
from itertools import product
import json
import numpy as np
import pickle
import tqdm
import yaml
from mjrl.utils.logger import DataLog
from tabulate import tabulate

import mjrl.envs
from mjmpc.envs import GymEnvWrapper
from mjmpc.envs.vec_env import SubprocVecEnv
from mjrl.utils import tensor_utils
from mjmpc.utils import timeit, helpers
from mjmpc.policies import MPCPolicy
import numpy as np
import torch


gym.logger.set_level(40)
parser = argparse.ArgumentParser(
    description='MPPI on mujoco environments')
parser.add_argument('--output', '-o', type=str,
                    #required=True,
                    default='datasets',
                    help='location to store results')
parser.add_argument('--config', type=str, required=True,
                    help='path to config file with exp params')
parser.add_argument('--act_repeat', type=int, default=1, 
                    help='action repeat, will average actions over the repeat')
parser.add_argument('--include', '-i', type=str,
                    required=False, help='package to import')
parser.add_argument('--seed', '-s', type=int,
                     default=None, help='seed')
parser.add_argument('--num_episodes', '-e', type=int,
                     default=10, help='seed')
args = parser.parse_args()

with open(args.config, 'r') as f:  # load config
    job_data = eval(f.read())

ENV_NAME = job_data['env_name']
#Create the main environment
env = gym.make(ENV_NAME)
env = GymEnvWrapper(env)
env.real_env_step(True)
act_repeat = args.act_repeat

if args.include:
    import sys
    splits = args.include.split("/")
    dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
    for x in splits[:-1]: dirpath = dirpath + "/" + x
    filename = splits[-1].split(".")[0]
    sys.path.append(dirpath)
    exec("from "+filename+" import *")
if 'obs_mask' in globals(): env.obs_mask = obs_mask

if args.seed is not None: job_data['seed'] = args.seed
SEED = job_data['seed']
del(job_data['seed'])
job_data['base_seed'] = SEED

OUT_DIR = os.path.join(args.output, ENV_NAME+'_'+str(SEED))
if not os.path.exists(args.output):
    os.mkdir(args.output)
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


if not os.path.exists(OUT_DIR+'/logs'):
    os.mkdir(OUT_DIR+'/logs')

assert 'sample_mode' in job_data.keys()
job_data['npg_hp'] = dict(
) if 'npg_hp' not in job_data.keys() else job_data['npg_hp']

logger = DataLog()
EXP_FILE = OUT_DIR + '/job_data.json'

with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)

policy_params = job_data['mpc_params']
policy_params['d_obs'] = env.d_obs
policy_params['d_state'] = env.d_state
policy_params['d_action'] = env.d_action
policy_params['d_action'] = env.action_space.low.shape[0]
policy_params['action_lows'] = env.action_space.low
policy_params['action_highs'] = env.action_space.high
num_cpu = 10 #policy_params['num_cpu']

#change required policy params to match mjmpc
policy_params['init_cov'] = pow(policy_params['init_std'], 2)
del(policy_params['init_std'])
policy_params['step_size'] = policy_params['step_size_mean']
del(policy_params['step_size_mean'])
policy_params['lam'] = policy_params['beta']
del(policy_params['beta'])
policy_params['alpha'] = 1
policy_params['sample_mode'] = 'mean'
policy_params['time_based_weights'] = False
del(policy_params['mixing_factor'])
del(policy_params['td_lam'])
del(policy_params['hotstart'])
del(policy_params['shift_steps'])
del(policy_params['squash_fn'])
del(policy_params['optimize_open_loop'])
del(policy_params['pessimism_mode'])
del(policy_params['epsilon'])

# ===============================================================================
# Setup Envs
# ===============================================================================
np.random.seed(SEED)
torch.random.manual_seed(SEED)
env.seed(SEED)

# Function to create vectorized environments for controller simulations
def make_env():
    gym_env = gym.make(ENV_NAME)
    rollout_env = GymEnvWrapper(gym_env)
    rollout_env.real_env_step(False)
    return rollout_env
#Vectorized Env for rollouts
sim_env = SubprocVecEnv([make_env for i in range(num_cpu)])  
#Function for rollouts with vectorized envs
def rollout_fn(num_particles, horizon, mean, noise, mode):
    """
    Given a batch of sequences of actions, rollout 
    in sim envs and return sequence of costs. The controller is 
    agnostic of how the rollouts are generated.
    """
    obs_vec, rew_vec, act_vec, done_vec, info_vec, next_obs_vec = sim_env.rollout(num_particles,
                                                                                horizon, 
                                                                                mean.copy(), 
                                                                                noise, 
                                                                                mode)
    #we assume environment returns rewards, but controller needs costs
    sim_trajs = dict(
        observations=obs_vec.copy(),
        actions=act_vec.copy(),
        costs=-1.0*rew_vec.copy(),
        dones=done_vec.copy(),
        next_observations=next_obs_vec.copy(),
        infos=helpers.stack_tensor_dict_list(info_vec.copy())
    )

    return sim_trajs

# ===============================================================================
# Data Collection Loop
# ===============================================================================
    
raw_paths = []
timeit.start('start_mppi')
for i in tqdm.tqdm(range(args.num_episodes)):

    policy = MPCPolicy(controller_type='mppi',
                        param_dict=policy_params, batch_size=1)
    policy.controller.set_sim_state_fn = sim_env.set_env_state
    policy.controller.rollout_fn = rollout_fn
    
    obs = env.reset()
    sim_env.reset()

    observations = []; actions = []; rewards = []; dones  = []
    infos = []; states = []; next_states = []
    done = False
    t = 0
    while t < env._max_episode_steps and done is False:    
        curr_state = deepcopy(env.get_env_state())
        action, value = policy.get_action(curr_state, calc_val=False)
        next_obs, reward, done, info = env.step(action)

        observations.append(obs); actions.append(action)
        rewards.append(reward); dones.append(done)
        infos.append(info); states.append(curr_state)
        obs = next_obs.copy()
    
    path = dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        # dones=np.array(dones),
        env_infos=tensor_utils.stack_tensor_dict_list(infos),
        states=states,
        terminated=done,
    )
    raw_paths.append(path)
        
timeit.stop('start_mppi') #stop timer after trajectory collection
success_metric = env.env.unwrapped.evaluate_success(raw_paths)
returns = np.array([np.sum(p['rewards']) for p in raw_paths])
num_samples = np.sum([p['rewards'].shape[0] for p in raw_paths])
print("Number of samples collected = %i" % num_samples)
print("Collected trajectory return mean, std, min, max = %.2f , %.2f , %.2f, %.2f" % \
       (np.mean(returns), np.std(returns), np.min(returns), np.max(returns)) )
print("Collected trajectory success percentage = %.2f" % \
       (success_metric))
logger.log_kv('eval_score', np.mean(returns))
logger.log_kv('eval_success', success_metric)
logger.log_kv('eval_score_std', np.std(returns))
print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                            logger.get_current_log_print().items()))
print(tabulate(print_data))
logger.save_log(OUT_DIR+'/logs')
sim_env.close()
env.close() 

#prep paths and dump
paths = []
for p in raw_paths:
    path = dict()
    raw_obs = p['observations']
    raw_act = p['actions']
    raw_rew = p['rewards']
    traj_length = raw_obs.shape[0]
    obs = env.obs_mask * raw_obs[::act_repeat]
    act = np.array([np.mean(raw_act[i * act_repeat : (i+1) * act_repeat], axis=0) for i in range(traj_length // act_repeat)])
    rew = np.array([np.sum(raw_rew[i * act_repeat : (i+1) * act_repeat]) for i in range(traj_length // act_repeat)])
    path['observations'] = obs
    path['actions'] = act
    path['rewards'] = rew
    paths.append(path)


pickle.dump(paths, open('datasets/'+ENV_NAME+'.pickle', 'wb'))
