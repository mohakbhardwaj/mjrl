"""
Job script to learn policy using MOReL
"""

from mjrl.algos.mbrl.sampling import sample_paths, evaluate_policy
from mjrl.algos.mbrl.model_based_npg import ModelBasedNPG
from mjrl.algos.mpc.mpc_agent import MPCAgent
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.utils.logger import DataLog
from mjrl.utils.gym_env import GymEnv
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.policies.gaussian_mlp import MLP
from tabulate import tabulate
from tqdm import tqdm
import d4rl
import mjrl.utils.tensor_utils as tensor_utils
import mjrl.samplers.core as sampler
import json
import os
import argparse
import time as timer
import mjrl.envs
import pickle
import torch.nn as nn
import torch
import copy
import numpy as np
from os import environ
environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
environ['MKL_THREADING_LAYER'] = 'GNU'
from mjrl.algos.mpc.ensemble_nn_dynamics import batch_call
from mjrl.algos.mpc.pretrain import train_dynamics_models

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(
    description='Model accelerated policy optimization.')
parser.add_argument('--output', '-o', type=str,
                    #required=True,
                    default='exp_results',
                    help='location to store results')
parser.add_argument('--config', '-c', type=str, required=True,
                    help='path to config file with exp params')
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


# Unpack args and make files for easy access
logger = DataLog()
EXP_FILE = OUT_DIR + '/job_data.json'

# base cases
if 'eval_rollouts' not in job_data.keys():
    job_data['eval_rollouts'] = 0
if 'save_freq' not in job_data.keys():
    job_data['save_freq'] = 10
if 'device' not in job_data.keys():
    job_data['device'] = 'cpu'
if 'hvp_frac' not in job_data.keys():
    job_data['hvp_frac'] = 1.0
if 'start_state' not in job_data.keys():
    job_data['start_state'] = 'init'
if 'learn_reward' not in job_data.keys():
    job_data['learn_reward'] = True
if 'num_cpu' not in job_data.keys():
    job_data['num_cpu'] = 1
if 'npg_hp' not in job_data.keys():
    job_data['npg_hp'] = dict()
if 'act_repeat' not in job_data.keys():
    job_data['act_repeat'] = 1
if 'model_file' not in job_data.keys():
    job_data['model_file'] = None

assert job_data['start_state'] in ['init', 'buffer']
# assert 'data_file' in job_data.keys()
job_data['data_file'] = os.path.join('datasets', ENV_NAME+'.pickle')
job_data['model_file'] = os.path.join(OUT_DIR, 'ensemble_model.pickle')
job_data['init_policy'] = os.path.join(OUT_DIR, 'bc_policy.pickle')


# ===============================================================================
# Helper functions
# ===============================================================================
def buffer_size(paths_list):
    return np.sum([p['observations'].shape[0]-1 for p in paths_list])


# ===============================================================================
# Setup functions and environment
# ===============================================================================

np.random.seed(SEED)
torch.random.manual_seed(SEED)

if ENV_NAME.split('_')[0] == 'dmc':
    # import only if necessary (not part of package requirements)
    import dmc2gym
    backend, domain, task = ENV_NAME.split('_')
    env = dmc2gym.make(domain_name=domain, task_name=task, seed=SEED)
    env = GymEnv(env, act_repeat=job_data['act_repeat'])
else:
    env = GymEnv(ENV_NAME, act_repeat=job_data['act_repeat'])
    env.set_seed(SEED)

# check for reward and termination functions
if 'reward_file' in job_data.keys():
    import sys
    splits = job_data['reward_file'].split("/")
    dirpath = "" if splits[0] == "" else os.path.dirname(
        os.path.abspath(__file__))
    for x in splits[:-1]:
        dirpath = dirpath + "/" + x
    filename = splits[-1].split(".")[0]
    sys.path.append(dirpath)
    exec("from "+filename+" import *")
if 'reward_function' not in globals():
    reward_function = getattr(env.env.env, "compute_path_rewards", None)
    job_data['learn_reward'] = False if reward_function is not None else True
if 'termination_function' not in globals():
    termination_function = getattr(env.env.env, "truncate_paths", None)
if 'obs_mask' in globals():
    env.obs_mask = obs_mask

mpc_params = job_data['mpc_params']

# ===============================================================================
# Load Data
# ===============================================================================

paths = pickle.load(open(job_data['data_file'], 'rb'))

rollout_score = np.mean([np.sum(p['rewards']) for p in paths])
num_samples = np.sum([p['rewards'].shape[0] for p in paths])
logger.log_kv('rollout_score', rollout_score)
logger.log_kv('num_samples', num_samples)
if hasattr(env.env.env, 'evaluate_success'):
    rollout_metric = env.env.env.evaluate_success(paths)
    logger.log_kv('rollout_metric', rollout_metric)


# ===============================================================================
# Behavior Cloning Initialization
# ===============================================================================

try:
    policy = pickle.load(open(job_data['init_policy'], 'rb'))
    policy.set_param_values(policy.get_param_values())
    init_log_std = job_data['init_log_std']
    min_log_std = job_data['min_log_std']
    if init_log_std:
        params = policy.get_param_values()
        params[:policy.action_dim] = tensor_utils.tensorize(init_log_std)
        policy.set_param_values(params)
    if min_log_std:
        policy.min_log_std[:] = tensor_utils.tensorize(min_log_std)
        policy.set_param_values(policy.get_param_values())
    policy_trained = True
    print('Policy Loaded')
except FileNotFoundError:
    policy = MLP(env.spec, seed=SEED, hidden_sizes=job_data['policy_size'],
            init_log_std=job_data['init_log_std'], min_log_std=job_data['min_log_std'])
    policy_trained = False


if not policy_trained:
    from mjrl.algos.behavior_cloning import BC
    print('Training behavior cloning')
    policy.to(job_data['device'])
    bc_agent = BC(paths, policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                  lr=job_data['bc_lr'], loss_type='MSE') #epochs=5
    bc_agent.train()
    print('Saving behavior cloned policy')
    pickle.dump(policy, open(OUT_DIR + '/bc_policy.pickle', 'wb'))

print("Performing validation rollouts for BC policy ... ")
eval_paths = evaluate_policy(env, policy, None, noise_level=0.0, real_step=True,
                                num_episodes=job_data['eval_rollouts'], visualize=False)
eval_score_bc = np.mean([np.sum(p['rewards']) for p in eval_paths])
print('BC', eval_score_bc)
logger.log_kv('eval_score_bc', eval_score_bc)

# ===============================================================================
# Model Training
# ===============================================================================

try:
    ensemble_model = pickle.load(open(job_data['model_file'], 'rb'))
    models_trained = True
    print('Dynamics model Loaded')
except FileNotFoundError:
    models = [WorldModel(state_dim=env.observation_dim, act_dim=env.action_dim, seed=SEED+i,
                        **job_data) for i in range(job_data['num_models'])]
    models_trained = False


if not models_trained:
    ts = timer.time()
    ensemble_model, model_train_info = train_dynamics_models(models, paths, **job_data)
    print('Saving trained dynamics models')
    pickle.dump(ensemble_model, open(os.path.join(OUT_DIR, 'ensemble_model.pickle'), 'wb'))
    pickle.dump(model_train_info, open(os.path.join(OUT_DIR, 'model_train_info.pickle'), 'wb'))
    tf = timer.time()
    logger.log_kv('model_learning_time', tf-ts)

print("Model learning statistics")
logger.log_kv('fit_epochs', job_data['fit_epochs'])
for i in range(len(ensemble_model)):
    logger.log_kv('dyn_loss_' + str(i), ensemble_model.train_info['dyn_loss'][i])
    logger.log_kv('dyn_loss_gen_' + str(i), ensemble_model.train_info['dyn_loss_gen'][i])
print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1, logger.get_current_log().items()))
print(tabulate(print_data))
logger.log_kv('act_repeat', job_data['act_repeat']) # log action repeat for completeness


# ===============================================================================
# Pessimistic MDP parameters
# ===============================================================================


if 'pessimism_coef' in job_data.keys():
    if job_data['pessimism_coef'] is None or job_data['pessimism_coef'] == 0.0:
        truncate_lim = None
        print("No pessimism used. Running naive MBRL.")
    else:
        # truncate_lim = (1.0 / job_data['pessimism_coef']) * np.max(delta)
        # print("Maximum error before truncation (i.env. unknown region threshold) = %f" % truncate_lim)
        truncate_lim = 1.0 / job_data['pessimism_coef'] / ensemble_model.train_info['ratio'] /  job_data['mpc_params']['horizon']
    job_data['truncate_lim'] = truncate_lim.tolist()
    job_data['truncate_reward'] = job_data['truncate_reward'] if 'truncate_reward' in job_data.keys() else 0.0
else:
    job_data['truncate_lim'] = None
    job_data['truncate_reward'] = 0.0


# ===============================================================================
# Policy Evaluation Loop
# ===============================================================================

# Save the config
json.dump(job_data, open(EXP_FILE, 'w'), indent=4)


agent = MPCAgent(env=env, learned_model=ensemble_model, sampling_policy=policy, mpc_params=mpc_params,
                 seed=SEED, save_logs=True, reward_function=reward_function,
                 #  reward_function2 = reward_function2,
                 termination_function=termination_function, termination_function2=termination_function2,
                 truncate_lim=job_data['truncate_lim'], truncate_reward=job_data['truncate_reward'],
                 device=job_data['device'])

best_perf = -1e8
for outer_iter in range(job_data['num_iter']):
    ts = timer.time()
    agent.to(job_data['device'])

    # evaluate true policy performance
    if job_data['eval_rollouts'] > 0:
        print("Performing validation rollouts ... ")
        # set the policy device back to CPU for env sampling
        eval_paths = evaluate_policy(agent.env, agent, None, noise_level=0.0,
                                     real_step=True, num_episodes=job_data['eval_rollouts'], visualize=False)
        eval_score = np.mean([np.sum(p['rewards']) for p in eval_paths])
        print(eval_score)
        # print('scores', np.array(agent._avg_scores))
        print('avg_scores', np.mean(agent._avg_scores))

        logger.log_kv('eval_score', eval_score)
        try:
            eval_metric = env.env.env.evaluate_success(eval_paths)
            logger.log_kv('eval_metric', eval_metric)
        except:
            pass
    else:
        eval_score = -1e8

    # track best performing policy
    policy_score = eval_score if job_data['eval_rollouts'] > 0 else rollout_score
    if policy_score > best_perf:
        # safe as policy network is clamped to CPU
        best_policy = copy.deepcopy(policy)
        best_perf = policy_score

    tf = timer.time()
    logger.log_kv('iter_time', tf-ts)
    for key in agent.logger.log.keys():
        logger.log_kv(key, agent.logger.log[key][-1])
    print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                               logger.get_current_log_print().items()))
    print(tabulate(print_data))
    logger.save_log(OUT_DIR+'/logs')

    if outer_iter > 0 and outer_iter % job_data['save_freq'] == 0:
        # convert to CPU before pickling
        agent.to('cpu')
        # make observation mask part of policy for easy deployment in environment
        old_in_scale = policy.in_scale
        for pi in [policy, best_policy]:
            pi.set_transformations(in_scale=1.0 / env.obs_mask)
        pickle.dump(agent, open(OUT_DIR + '/iterations/agent_' +
                    str(outer_iter) + '.pickle', 'wb'))
        pickle.dump(policy, open(OUT_DIR + '/iterations/policy_' +
                    str(outer_iter) + '.pickle', 'wb'))
        pickle.dump(best_policy, open(
            OUT_DIR + '/iterations/best_policy.pickle', 'wb'))
        agent.to(job_data['device'])
        for pi in [policy, best_policy]:
            pi.set_transformations(in_scale=old_in_scale)
        make_train_plots(log=logger.log, keys=['rollout_score', 'eval_score', 'rollout_metric', 'eval_metric'],
                         x_scale=float(job_data['act_repeat']), y_scale=1.0, save_loc=OUT_DIR+'/logs/')

# final save
pickle.dump(agent, open(OUT_DIR + '/iterations/agent_final.pickle', 'wb'))
policy.set_transformations(in_scale=1.0 / env.obs_mask)
pickle.dump(policy, open(OUT_DIR + '/iterations/policy_final.pickle', 'wb'))
