"""
Job script to learn policy using MOReL
"""

from mjrl.algos.mbrl.sampling import sample_paths, evaluate_policy
from mjrl.algos.mbrl.model_based_npg import ModelBasedNPG
# from mjrl.algos.mpc.mpc_agent import MPCAgent
from mjrl.algos.mpc.pmppi_agent import PMPPIAgent as MPCAgent
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.utils.make_train_plots import make_train_plots
import mjrl.utils.process_samples as process_samples
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
from mjrl.algos.mpc.nn_value_fn import ValueFunctionNet
from mjrl.utils.utils import import_from_path
from mjrl.envs.env_utils import eval_success_robel
torch.set_printoptions(precision=10)



import os
FILE_PATH = os.path.dirname(os.path.abspath(__file__))  # all paths specified are relative to this file

            
def train(*,
          job_data,
          output='exp_results',
          datapath='datasets',
          modelpath='cached_models',  # path to cached data (e.g. pretrained models)
          readonly=True,
          seed=123):

    ENV_NAME = job_data['env_name']
    if seed is not None: job_data['seed'] = seed
    SEED = job_data['seed']
    del(job_data['seed'])
    job_data['base_seed'] = SEED

    OUT_DIR = os.path.join(output, ENV_NAME+'_'+str(SEED))
    MODEL_DIR = os.path.join(modelpath, ENV_NAME)
    DATA_DIR = datapath #   os.path.join(datapath, ENV_NAME)

    if not os.path.exists(output):
        os.makedirs(output)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(OUT_DIR+'/iterations'):
        os.makedirs(OUT_DIR+'/iterations')
    if not os.path.exists(OUT_DIR+'/logs'):
        os.makedirs(OUT_DIR+'/logs')

    # Unpack args and make files for easy access
    logger = DataLog()
    EXP_FILE = OUT_DIR + '/job_data.json'

    # base cases
    if 'eval_rollouts' not in job_data.keys():
        job_data['eval_rollouts'] = 0
    if 'save_freq' not in job_data.keys():
        job_data['save_freq'] = 10
    if 'device' not in job_data.keys() or not torch.cuda.is_available():
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

    gamma = job_data['mpc_params']['gamma']

    assert job_data['start_state'] in ['init', 'buffer']
    # assert 'data_file' in job_data.keys()
    job_data['data_file'] = os.path.join(DATA_DIR, ENV_NAME+'.pickle')
    #job_data['model_file'] = os.path.join(MODEL_DIR, 'ensemble_model_'+str(job_data['num_models'])+'.pickle')
    job_data['model_file'] = os.path.join(MODEL_DIR, 'ensemble_model.pickle')
    job_data['init_policy'] = os.path.join(MODEL_DIR, 'bc_policy.pickle')
    job_data['init_val_fn'] = os.path.join(MODEL_DIR, 'val_fn.pickle')

    # ===============================================================================
    # Helper functions
    # ===============================================================================
    def buffer_size(paths_list):
        return np.sum([p['observations'].shape[0]-1 for p in paths_list])

    # ===============================================================================
    # Setup functions and environment
    # ===============================================================================
    is_robel_env = False
    
    if ENV_NAME.split('_')[0] == 'dmc':
        # import only if necessary (not part of package requirements)
        import dmc2gym
        backend, domain, task = ENV_NAME.split('_')
        env = dmc2gym.make(domain_name=domain, task_name=task, seed=SEED)
        env = GymEnv(env, act_repeat=job_data['act_repeat'])
    elif ENV_NAME.split('-')[0] in ['sawyer', 'robel']:
        splits = ENV_NAME.split('-')
        if splits[0] == 'robel':
            is_robel_env = True
        splits.pop(-2)
        ENV_NAME = "-".join(splits)
        env = GymEnv(ENV_NAME, act_repeat=job_data['act_repeat'])
    else:
        env = GymEnv(ENV_NAME, act_repeat=job_data['act_repeat'])
    
    
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)    
    env.set_seed(SEED)


    # check for reward and termination functions
    reward_function = getattr(env.env.env, "compute_path_rewards", None)
    termination_function = getattr(env.env.env, "truncate_paths", None)
    if 'reward_file' in job_data.keys():
        mod = import_from_path(job_data['reward_file'], base_path=FILE_PATH)
        if 'reward_function' in vars(mod):
            reward_function = mod.reward_function
        if 'termination_function' in vars(mod):
            termination_function = mod.termination_function
        if 'obs_mask' in vars(mod):
            env.obs_mask = mod.obs_mask
    job_data['learn_reward'] = False if reward_function is not None else True

    mpc_params = job_data['mpc_params']

    # ===============================================================================
    # Load Data
    # ===============================================================================
    try:
        paths = pickle.load(open(job_data['data_file'], 'rb'))
    except FileNotFoundError:
        if readonly: raise Exception('No cached model/data is found but the mode is read-only.')
        from prep_d4rl_dataset import prep_d4rl_dataset
        paths =  prep_d4rl_dataset(env_name=ENV_NAME,
                                output=DATA_DIR,
                                include=job_data.get('reward_file', None),
                                seed=SEED)


    rollout_score = np.mean([np.sum(p['rewards']) for p in paths])
    try:
        rollout_norm_score = np.mean([env.env.get_normalized_score(
            np.sum(p['rewards'])) for p in paths])
    except:
        rollout_norm_score = 0.0

    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    logger.log_kv('rollout_score', rollout_score)
    logger.log_kv('rollout_norm_score', rollout_norm_score)
    logger.log_kv('num_samples', num_samples)
    if hasattr(env.env.env, 'evaluate_success') and 'env_infos' in paths[0].keys(): #TODO: Mohak: D4RL dataset has env_infos in different format
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

        try:
            bc_train_info = pickle.load(
                open(os.path.join(MODEL_DIR, 'bc_train_info.pickle'), 'rb'))
            logger.log_kv('BC_error_before', bc_train_info['BC_error_before'])
            logger.log_kv('BC_error_after', bc_train_info['BC_error_after'])
        except FileNotFoundError:
            pass

        print('Policy Loaded')

    except FileNotFoundError:
        if readonly: raise Exception('No cached model/data is found but the mode is read-only.')

        policy = MLP(env.spec, seed=SEED, hidden_sizes=job_data['policy_size'],
                init_log_std=job_data['init_log_std'], min_log_std=job_data['min_log_std'])

        from mjrl.algos.behavior_cloning import BC
        print('Training behavior cloning')
        policy.to(job_data['device'])
        bc_agent = BC(paths, policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                    lr=job_data['bc_lr'], loss_type='MSE') #epochs=5
        error_before, error_after = bc_agent.train()
        print('Saving behavior cloned policy')
        pickle.dump(policy, open(job_data['init_policy'], 'wb'))

        logger.log_kv('BC_error_before', error_before)
        logger.log_kv('BC_error_after', error_after)
        bc_stats = dict(BC_error_before=error_before,
                        BC_error_after=error_after)
        pickle.dump(bc_stats, open(
            MODEL_DIR + '/bc_train_info.pickle', 'wb'))



    print("Performing validation rollouts for BC policy ... ")
    eval_paths = evaluate_policy(env, policy, None, noise_level=0.0, real_step=True,
                                 num_episodes=job_data['eval_rollouts'], visualize=False)
    eval_score_bc = np.mean([np.sum(p['rewards']) for p in eval_paths])
    eval_score_bc_std = np.std([np.sum(p['rewards']) for p in eval_paths])
    print('BC', eval_score_bc)
    try:
        eval_metric_bc = env.env.env.evaluate_success(eval_paths)
    except:
        eval_metric_bc = 0.0

    #Get normalized score for bc
    try:
        norm_score_bc = np.mean([ env.env.get_normalized_score(np.sum(p['rewards'])) for p in eval_paths])
        norm_score_bc_std = np.std([ env.env.get_normalized_score(np.sum(p['rewards'])) for p in eval_paths])
    except:
        norm_score_bc = 0.0
    logger.log_kv('eval_score_bc', eval_score_bc)
    logger.log_kv('eval_score_bc_std', eval_score_bc_std)
    logger.log_kv('eval_metric_bc', eval_metric_bc)
    logger.log_kv('eval_norm_score_bc', norm_score_bc)
    logger.log_kv('eval_norm_score_bc_std', norm_score_bc_std)
    print_data = sorted(filter(lambda v: np.asarray(
        v[1]).size == 1, logger.get_current_log().items()))
    print(tabulate(print_data))
    # ===============================================================================
    # Value Function Initialization
    # ===============================================================================
    process_samples.compute_returns(paths, gamma)

    try:
        value_fn = pickle.load(open(job_data['init_val_fn'], 'rb'))
        # value_fn.set_params(value_fn.get_params())
        try:
            val_fn_train_info = pickle.load(open(os.path.join(MODEL_DIR,'val_fn_train_info.pickle'), 'rb'))
            logger.log_kv('time_VF', val_fn_train_info['time_vf'])
            logger.log_kv('VF_error_before', val_fn_train_info['VF_error_before'])
            logger.log_kv('VF_error_after', val_fn_train_info['VF_error_after'])
        except FileNotFoundError:
            pass

    except FileNotFoundError:
        if readonly: raise Exception('No cached model/data is found but the mode is read-only.')

        value_fn = ValueFunctionNet(state_dim=env.observation_dim, hidden_size=job_data['val_fn_size'],
                            epochs=job_data['val_fn_epochs'], reg_coef=job_data['val_fn_wd'],
                            batch_size=job_data['val_fn_batch_size'], learn_rate=job_data['val_fn_lr'],
                            device=job_data['device'])

        print('Training value function')
        # compute returns
        ts = timer.time()
        error_before, error_after, epoch_losses = value_fn.fit(paths, return_errors=True)
        time_vf = timer.time() - ts
        logger.log_kv('time_VF', time_vf)
        logger.log_kv('VF_error_before', error_before)
        logger.log_kv('VF_error_after', error_after)
        print('Saving value function')
        pickle.dump(value_fn, open(job_data['init_val_fn'], 'wb'))
        vf_stats = dict(VF_error_before=error_before, VF_error_after=error_after, time_vf=time_vf)
        pickle.dump(vf_stats, open(MODEL_DIR + '/val_fn_train_info.pickle', 'wb'))

    # ===============================================================================
    # Model Training
    # ===============================================================================
    try:
        ensemble_model = pickle.load(open(job_data['model_file'], 'rb'))
        print('Dynamics model Loaded')
    except FileNotFoundError:
        if readonly: raise Exception('No cached model/data is found but the mode is read-only.')

        models = [WorldModel(state_dim=env.observation_dim - job_data['context_dim'], act_dim=env.action_dim, seed=SEED+i,
                            **job_data) for i in range(job_data['num_models'])]
        ts = timer.time()
        ensemble_model, model_train_info = train_dynamics_models(models, paths, **job_data)
        print('Saving trained dynamics models')
        pickle.dump(ensemble_model, open(job_data['model_file'], 'wb'))
        pickle.dump(model_train_info, open(os.path.join(MODEL_DIR, 'model_train_info.pickle'), 'wb'))
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


    agent = MPCAgent(env=env, learned_model=ensemble_model, sampling_policy=policy, value_fn=value_fn,
                    mpc_params=mpc_params, seed=SEED, save_logs=True, reward_function=reward_function,
                    #  reward_function2 = reward_function2,
                    termination_function=termination_function, # termination_function2=termination_function2,
                    truncate_lim=job_data['truncate_lim'], truncate_reward=job_data['truncate_reward'],
                    device=job_data['device'])


    for outer_iter in range(job_data['num_iter']):
        ts = timer.time()
        agent.to(job_data['device'])

        # evaluate true policy performance
        assert job_data['eval_rollouts'] > 0
        print("Performing validation rollouts ... ")
        eval_paths = evaluate_policy(agent.env, agent, None, noise_level=0.0,
                                     real_step=True, num_episodes=job_data['eval_rollouts'], visualize=False)
        eval_score = np.mean([np.sum(p['rewards']) for p in eval_paths])
        eval_score_std = np.std([np.sum(p['rewards']) for p in eval_paths])

        try:
            norm_score = np.mean(
                [env.env.get_normalized_score(np.sum(p['rewards'])) for p in eval_paths])
            norm_score_std = np.std(
                [env.env.get_normalized_score(np.sum(p['rewards'])) for p in eval_paths])
        except:
            norm_score = 0.0
            norm_score_std = 0.0
        print(eval_score)
        # print('scores', np.array(agent._avg_scores))
        print('avg_scores', np.mean(agent._scores))

        # Update and print Logger
        logger.log_kv('eval_score', eval_score)
        logger.log_kv('eval_score_std', eval_score_std)
        logger.log_kv('eval_norm_score', norm_score)
        logger.log_kv('eval_norm_score_std', norm_score_std)
        try:
            eval_metric = env.env.env.evaluate_success(eval_paths)
            logger.log_kv('eval_metric', eval_metric)
        except:
            pass

        tf = timer.time()
        logger.log_kv('iter_time', tf-ts)
        print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1, logger.get_current_log_print().items()))
        print(tabulate(print_data))
        logger.save_log(OUT_DIR+'/logs')

    return norm_score

if __name__=='__main__':

    # ===============================================================================
    # Get command line arguments
    # ===============================================================================

    parser = argparse.ArgumentParser(description='Model accelerated policy optimization.')
    parser.add_argument('--config', '-c', type=str, required=True, help='path to config file with exp params')
    parser.add_argument('--output', '-o', type=str,  default='exp_results', help='location to store results')
    # parser.add_argument('--include', '-i', type=str, required=False, help='package to import')
    parser.add_argument('--seed', '-s', type=int, default=None, help='seed')
    parser.add_argument('--env_name', '-e', type=str, default=None)
    kwargs = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}

    # Load config
    with open(kwargs['config'], 'r') as f:
        job_data = eval(f.read())
        kwargs['job_data'] = job_data
        del kwargs['config']

        if 'env_name' in kwargs:
            kwargs['job_data']['env_name'] = kwargs['env_name']
            del kwargs['env_name']

    train(readonly=False, **kwargs)
