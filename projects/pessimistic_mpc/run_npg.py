"""
This is a job script for running policy gradient algorithms on gym tasks.
Separate job scripts are provided to run few other algorithms
- For DAPG see here: https://github.com/aravindr93/hand_dapg/tree/master/dapg/examples
- For model-based NPG see here: https://github.com/aravindr93/mjrl/tree/master/mjrl/algos/model_accel
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.logger import DataLog
# from mjrl.algos.batch_reinforce import BatchREINFORCE
# from mjrl.algos.ppo_clip import PPO
from mjrl.utils.train_agent import train_agent
import os
import json
import gym
import mjrl.envs
import time as timer
import pickle
import argparse
import numpy as np
import torch

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(
    description='Natural policy gradient on mujoco environments')
parser.add_argument('--output', '-o', type=str,
                    #required=True,
                    default='npg_results',
                    help='location to store results')
parser.add_argument('--config', type=str, required=True,
                    help='path to config file with exp params')
args = parser.parse_args()
parser.add_argument('--include', '-i', type=str,
                    required=False, help='package to import')
parser.add_argument('--seed', '-s', type=int,
                     default=None, help='seed')
args = parser.parse_args()

with open(args.config, 'r') as f:  # load config
    job_data = eval(f.read())
if args.include:
    exec("import "+args.include)  # import extra stuff
ENV_NAME = job_data['env_name']
if args.seed is not None: job_data['seed'] = args.seed
SEED = job_data['seed']
del(job_data['seed'])
job_data['base_seed'] = SEED

OUT_DIR = os.path.join(args.output, ENV_NAME+'_'+str(SEED))
if not os.path.exists(args.output):
    os.mkdir(args.output)
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
if not os.path.exists(OUT_DIR+'/iterations'):
    os.mkdir(OUT_DIR+'/iterations')
if not os.path.exists(OUT_DIR+'/logs'):
    os.mkdir(OUT_DIR+'/logs')

assert 'sample_mode' in job_data.keys()
job_data['npg_hp'] = dict(
) if 'npg_hp' not in job_data.keys() else job_data['npg_hp']

logger = DataLog()
EXP_FILE = OUT_DIR + '/job_data.json'

with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)

if job_data['sample_mode'] == 'trajectories':
    assert 'rl_num_traj' in job_data.keys()
    job_data['rl_num_samples'] = 0  # will be ignored
elif job_data['sample_mode'] == 'samples':
    assert 'rl_num_samples' in job_data.keys()
    job_data['rl_num_traj'] = 0    # will be ignored
else:
    print("Unknown sampling mode. Choose either trajectories or samples")
    exit()

# ===============================================================================
# Train Loop
# ===============================================================================
np.random.seed(SEED)
torch.random.manual_seed(SEED)

e = GymEnv(job_data['env_name'])
e.set_seed(SEED)
e.action_space.seed(SEED)
policy = MLP(e.spec, hidden_sizes=job_data['policy_size'],
             seed=SEED, init_log_std=job_data['init_log_std'],
             min_log_std=job_data['min_log_std'])
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'], hidden_sizes=job_data['vf_hidden_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

# Construct the algorithm
agent = NPG(e, policy, baseline, normalized_step_size=job_data['step_size'],
            seed=SEED, save_logs=True, **job_data['npg_hp'])


print("========================================")
print("Starting policy learning")
print("========================================")

ts = timer.time()
train_agent(job_name=OUT_DIR,
            agent=agent,
            seed=SEED,
            niter=job_data['rl_num_iter'],
            gamma=job_data['gamma'],
            gae_lambda=job_data['gae_lambda'],
            num_cpu=job_data['num_cpu'],
            sample_mode=job_data['sample_mode'],
            num_traj=job_data['rl_num_traj'],
            num_samples=job_data['rl_num_samples'],
            save_freq=job_data['save_freq'],
            evaluation_rollouts=job_data['eval_rollouts'])
print("time taken = %f" % (timer.time()-ts))
