import torch
import torch.autograd.profiler as profiler
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import time as timer
import random

from mjrl.utils.gym_env import GymEnv
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.algos.mpc.mpc_utils import scale_ctrl, cost_to_go, cost_to_go_debug
from mjrl.utils.logger import DataLog
from mjrl.algos.mpc.ensemble_nn_dynamics import EnsembleWorldModel
torch.set_printoptions(precision=8)


from mjrl.algos.mpc.abstract_mpc_agent import AbstractMPCAgent


class PMPPIAgent(AbstractMPCAgent):
    """ A Pessimistic MPC algorithm based on MPPI. """

    def __init__(self, *,
                 learned_model,
                 sampling_policy,
                 value_fn=None,
                 mpc_params,
                 reward_function,
                 termination_function,
                 termination_function2=None, # TODO remove
                 truncate_lim=None,
                 truncate_reward=0.0,
                 **kwargs,
                 ):

        ## Extra attributes
        self.sampling_policy = sampling_policy
        self.truncate_lim = torch.tensor(truncate_lim) if truncate_lim is not None else None
        self.truncate_reward = truncate_reward

        # MPC parameters
        self.mpc_params = mpc_params
        self.n_iters = self.mpc_params['n_iters']
        self.horizon = self.mpc_params['horizon']
        self.num_particles = self.mpc_params['num_particles']
        self.init_std = self.mpc_params['init_std']
        self.gamma = self.mpc_params['gamma']
        self.beta = self.mpc_params['beta']
        self.td_lam = self.mpc_params['td_lam']
        self.step_size_mean = self.mpc_params['step_size_mean']
        self.sample_mode = self.mpc_params['sample_mode']
        self.base_action = self.mpc_params['base_action']
        self.hotstart = self.mpc_params['hotstart']
        self.shift_steps = self.mpc_params['shift_steps']
        self.squash_fn = self.mpc_params['squash_fn']
        self.filter_coeffs = self.mpc_params['filter_coeffs']
        self.mixing_factor = self.mpc_params['mixing_factor']
        self.optimize_open_loop = self.mpc_params['optimize_open_loop']
        self.pessimism_mode = self.mpc_params['pessimism_mode']
        self.epsilon = self.mpc_params['epsilon']
        self.sync_model_randomness = self.mpc_params['sync_model_randomness']

        # atac
        self.actions_to_take = None

        self._value_fn = value_fn
        super().__init__(# NOTE currently for backward compatability
                         dynamics_model=learned_model,
                         reward_model=reward_function,
                         termination_model=termination_function,
                         heuristic=self._heuristic,
                         gamma=self.gamma,
                         horizon=self.horizon,
                         lambd=self.td_lam,
                         num_particles=self.num_particles, # number of rollouts (in parallel)
                         filter_actions=True,  #XXX
                         squash_fn=self.squash_fn,
                         filter_coeffs=self.filter_coeffs,
                         n_iters=self.n_iters,
                         **kwargs)

    def _heuristic(self, *args, **kwargs):
        return self._value_fn.forward(*args, **kwargs)

    def reset(self):
        super().reset()
        self.reset_distribution()

    def preprocess_and_reset(self, observation, infos=None, sample_zero_action=True):
        # Sample open-loop actions for subsequent rollouts
        eps = torch.randn(self.num_models, self.num_particles, self.horizon, self.action_dim, device=self.device)
        if self.sync_model_randomness:
            eps[:] = eps[0] # use the same perturbation for all model
        eps[:,0] = 0. # set first perturbation to be zeros
        self._open_loop_actions = self.mean_action.unsqueeze(1).repeat(1,self.num_particles, 1, 1) + self.init_std * eps
        if self.optimize_open_loop and sample_zero_action:
            self._open_loop_actions[:,-1] = 0.  # always sample a zero action sequence

        # Alaways sample closed-loop actions in rollouts in optimization
        self._sample_cl_actions = True

        return observation

    def warmstart(self):
        if self.hotstart:
            self.shift(self.shift_steps)
        else:
            self.reset_distribution()

    def reset_distribution(self):
        self.num_means = 1 if self.pessimism_mode == 'atac' else self.num_models
        self.mean_action = torch.zeros((self.num_means, self.horizon, self.action_dim), device=self.device)
        self.std_action = torch.tensor(self.init_std, device=self.device)

    def shift(self, shift_steps):
        self.mean_action = self.mean_action.roll(-shift_steps, 1)
        if self.base_action == 'random':
            self.mean_action[:,-shift_steps:] = self.action_range * torch.rand(self.num_models, shift_steps, self.action_dim, device=self.device) + self.action_lows
        elif self.base_action == 'zero':
            self.mean_action[:,-shift_steps:].zero_()
        elif self.base_action == 'repeat':
            self.mean_action[:,-shift_steps:] = self.mean_action[:, -shift_steps - 1].unsqueeze(1).repeat(1,shift_steps,1)#clone()
        else:
            raise NotImplementedError(
                "invalid option {} for base action during shift".format(self.base_action))

    def optimize(self, observation):
        with torch.no_grad():  # since MPPI is derivative-free
            with profiler.record_function("mppi_update"):
                act = super().optimize(observation)
        return act[0]


    def postprocess(self, paths):
        # Compute prediction error, which is used in modifying the reward and termination.
        paths['open_loop_actions'] = self._open_loop_actions.clone()
        if self.truncate_lim is not None and len(self.dynamics_model) > 1 and (self.pessimism_mode != 'atac') :
            num_models, num_particles, horizon = paths['observations'].shape[:-1]
            pred_err = self.dynamics_model.compute_delta(
                            paths['observations'].view(-1, self.observation_dim), # model*particles*horizon x dim
                            paths['actions'].view(-1, self.action_dim)) # model*particles*horizon
            paths['pred_err'] = pred_err.view(num_models, num_particles, horizon) # model x particles x horizon
            # if self.pessimism_mode == "discount":
            #     termination_prob = 2.0 * (torch.sigmoid(pred_err / self.truncate_lim.view(-1,1,1)) - 0.5)
            #     new_discount = self.gamma * (1.0 - termination_prob)
            #     new_discount[:,:,0] = 1.0
            #     gamma_seq = torch.cumprod(new_discount, dim=-1)

        super().postprocess(paths)

    def reward_function(self, paths):
        rewards = super().reward_function(paths)
        if self.pessimism_mode == "bonus":
            bonus =  (paths['pred_err'] / self.truncate_lim.view(-1,1,1)) * (1.0 - paths["dones"])
            rewards -= bonus
        return rewards

    def termination_function(self, paths):
        if self.pessimism_mode == "truncation":
            # Set extra done flags based on uncertinaty
            violations = paths['pred_err'] > self.truncate_lim.view(-1,1,1)
            violations[:,:,1:] = violations[:,:,0:-1]  # last dim is horizon
            violations[:,:,0] = False
            paths['dones'] = violations
        return  super().termination_function(paths)


    ## Below are methods that require algorithm-specific instantiations
    def rollout_policy(self, observations, t, include_noise=False):
        a_ol = self._open_loop_actions[:, :, t] # model x particles x horizon x dim
        if self._sample_cl_actions:
            # Sample from the base policy
            a_policy = self.sampling_policy.forward(observations.view(-1,self.observation_dim))
            if include_noise:
                eps = torch.randn_like(a_policy)
                noise = torch.exp(self.sampling_policy.log_std) * eps
                a_policy = a_policy + noise
            a_policy = a_policy.view(a_ol.shape)
            # Combine with open loop actions
            if self.optimize_open_loop:
                at = a_policy + a_ol
            else:
                at = (1.0-self.mixing_factor) * a_policy + self.mixing_factor * a_ol # mixed action
        else:
            at = a_ol

        return at

    def evaluate_action_sequence(self, observation, action_sequence, model_idx=-1):
        # TODO double check
        # Make the rollout_policy to use the action_sequence
        self._open_loop_actions = action_sequence  # model x particles x horizon x dim
        num_particles = action_sequence.shape[1]
        self._sample_cl_actions = self.optimize_open_loop
        paths = self.generate_rollouts(observation, num_particles=num_particles, model_idx=model_idx)
        self.postprocess(paths)
        return paths


    def update_policy(self, observation, paths, infos=None):

        if self.pessimism_mode=='atac2':
            assert self.sync_model_randomness
            assert self.optimize_open_loop
            assert self.optimization_freq==1
            base_disc_return = paths["discounted_return"][:,0:1]  # assume the first action is mean
            new_mean_disc_return = paths["discounted_return"] # models x policies
            val_diff = new_mean_disc_return - base_disc_return  # model x particles
            obj, _ = val_diff.min(axis=0)
            # obj = val_diff.mean(axis=0) - 3*val_diff.std(axis=0)
            ind = obj>= self.epsilon  # throw away impossible ones
            if sum(ind)>0:
                print(sum(ind))
                obj = obj.unsqueeze(0)
                w = torch.softmax((1.0/self.beta) * obj[:,ind], dim=-1)
                actions = paths['open_loop_actions'][:,ind]
                # assert torch.norm((actions[:,0]-self.mean_action))<1e-10  # assume the first action is mean
                weighted_seq = w[:,:,None,None] * actions
                new_mean = torch.sum(weighted_seq, dim=1)  # over particles
                new_mean = (1.0 - self.step_size_mean) * self.mean_action + self.step_size_mean * new_mean  # generate candidates
                self.mean_action = new_mean
            scores, _ = torch.max(paths["discounted_return"], dim=1)  # over particles
            return scores

        actions = paths['open_loop_actions'] if self.optimize_open_loop else paths['actions']
        w = torch.softmax((1.0/self.beta) * paths["discounted_return"], dim=-1)
        # Update mean
        weighted_seq = w[:,:,None,None] * actions
        new_mean = torch.sum(weighted_seq, dim=1)  # over particles
        new_mean = (1.0 - self.step_size_mean) * self.mean_action + self.step_size_mean * new_mean  # generate candidates

        if self.pessimism_mode == "atac":
            assert self.optimize_open_loop
            assert self.optimization_freq==1

            all_means = torch.cat([new_mean, self.mean_action])
            # rollout to get its discounted return
            paths_new = self.evaluate_action_sequence(observation, all_means.unsqueeze(0).repeat(self.num_models,1,1,1))  # for each mean, roll for all the models.
            new_mean_disc_return = paths_new["discounted_return"][:,:-1] # models x policies
            base_disc_return = paths_new["discounted_return"][:,-1:]
            # compute pessimistic value difference
            pessimistic_val_diff, _ = (new_mean_disc_return - base_disc_return).min(axis=0)
            # select mean with max value difference
            max_perf_gap, max_policy_idx = torch.max(pessimistic_val_diff, dim=0)
            # check whether to accept the candidate
            if max_perf_gap.item() >= self.epsilon:
                new_mean = new_mean[max_policy_idx].unsqueeze(0)
            else:
                new_mean = self.mean_action
                max_policy_idx = -1
            self.actions_to_take = paths_new['actions'][0, max_policy_idx]  # since we take only the first action, the model index 0 doesn't matter.

        self.mean_action = new_mean
        scores, _ = torch.max(paths["discounted_return"], dim=1)  # over particles
        return scores


    def make_decision(self, observation, infos):
        mode = self.sample_mode
        score = infos[-1].cpu().numpy() # TODO

        if self.pessimism_mode == 'atac':
            return self.actions_to_take, score
        if mode == 'best_mean':
            paths = self.evaluate_action_sequence(observation, self.mean_action.unsqueeze(1))
            best_idx = torch.argmax(paths['discounted_return'])
            act_seq = paths["actions"][best_idx].squeeze(0) #self.mean_action[best_idx].clone()
        elif mode == 'worst_mean':
            paths = self.evaluate_action_sequence(observation, self.mean_action.unsqueeze(1))
            worst_idx = torch.argmin(paths['discounted_return'])
            # print(paths['discounted_return'], worst_idx)
            act_seq = paths["actions"][worst_idx].squeeze(0) #self.mean_action[worst_idx].clone()
        elif mode == 'softmax_mean':
            paths = self.evaluate_action_sequence(observation, self.mean_action.unsqueeze(1))
            weights = torch.softmax((1.0 / self.beta) *  paths['discounted_return'], dim=0)
            weighted_mean = (weights.T * self.mean_action.T).T
            act_seq = torch.sum(weighted_mean, dim=0).clone()
        elif mode == "softmin_mean":
            paths = self.evaluate_action_sequence(observation, self.mean_action.unsqueeze(1))
            weights = torch.softmax((-1.0 / self.beta) *  paths['discounted_return'], dim=0)
            weighted_mean = (weights.T * self.mean_action.T).T
            act_seq = torch.sum(weighted_mean, dim=0).clone()
        elif mode == 'random_mean':
            rand_idx = np.random.randint(self.num_models)
            act_seq = self.mean_action[rand_idx]
        elif mode == 'average_mean':
            act_seq = torch.mean(self.mean_action, dim=0)
        elif type(mode) is int:
            paths = self.evaluate_action_sequence(observation, self.mean_action.unsqueeze(1))
            act_seq = paths["actions"][mode].squeeze(0)

        elif mode == 'best_mean_sample':
            raise ValueError('To be implemented')
        elif mode == 'random_mean_sample':
            raise ValueError('To be implemented')
        else:
            raise ValueError('Sampling mode not recognized')

        return act_seq, score # TODO
