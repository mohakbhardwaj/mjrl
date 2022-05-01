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

torch.set_printoptions(precision=8)

class MPCAgent():
    def __init__(self, 
    env,
    learned_model=None,
    sampling_policy=None,
    mpc_params=None,
    reward_function=None,
    # reward_function2=None,
    termination_function=None,
    termination_function2=None,
    truncate_lim=None,
    truncate_reward=0.0,
    seed=1234,
    device='cpu',
    save_logs=True):

        self.env = env
        if learned_model is None:
                print("Algorithm requires a (list of) learned dynamics model")
                quit()
        elif isinstance(learned_model, WorldModel):
            self.learned_model = [learned_model]
        else:
            self.learned_model = learned_model

        self.num_models = len(self.learned_model)
        self.sampling_policy = sampling_policy
        # self.refine, self.kappa, self.plan_horizon, self.plan_paths = refine, kappa, plan_horizon, plan_paths
        if mpc_params is None:
            print("Algorithm requires mpc params")
            quit()
        else:
            self.mpc_params = mpc_params
        self.reward_function, self.termination_function = reward_function, termination_function
        # self.reward_function2 = reward_function2
        self.termination_function2 = termination_function2
        self.truncate_lim, self.truncate_reward = truncate_lim, truncate_reward
        self.seed = seed
        self.save_logs = save_logs

        # number of states
        self.observation_dim = self.env.spec.observation_dim #if env_spec is not None else observation_dim
        # number of actions
        self.action_dim = self.env.spec.action_dim #if env_spec is not None else action_dim
        self.action_lows = torch.tensor(self.env.action_space.low)
        self.action_highs = torch.tensor(self.env.action_space.high)
        self.action_range = self.action_highs - self.action_lows


        self.n_iters = self.mpc_params['n_iters'] 
        self.horizon = self.mpc_params['horizon'] 
        self.num_particles = self.mpc_params['num_particles'] 
        self.init_cov = self.mpc_params['init_cov'] 
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


        #set the initial mean and covariance of mpc
        self.init_mean = torch.zeros((self.num_models, self.horizon, self.action_dim))
        self.init_cov = torch.tensor(self.init_cov)
        self.init_std = torch.sqrt(self.init_cov)
        #calling reset distribution sets current mean and cov 
        #to initial values
        self.reset_distribution()
        self.gamma_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma] * (self.horizon - 1)),dim=0).reshape(1, self.horizon)
        # self.mvn = MultivariateNormal(
        #     loc=torch.zeros(self.horizon * self.action_dim), 
        #     covariance_matrix=self.init_cov * torch.eye(self.horizon * self.action_dim)) 
        self.sample_shape = torch.Size([self.num_particles])
        self.num_steps = 0
        self.to(device)
        if save_logs: self.logger = DataLog()


    def to(self, device):
        # Convert all the networks (except policy network which is clamped to CPU)
        # to the specified device
        for model in self.learned_model:
            model.to(device)
        # try:
        #     self.baseline.model.to(device)
        # except:
        #     pass
        try:
            self.sampling_policy.to(device)
        except:
            pass
        self.init_mean, self.init_cov = self.init_mean.to(device), self.init_cov.to(device)
        self.init_std = self.init_std.to(device)
        self.mean_action, self.cov_action = self.mean_action.to(device), self.cov_action.to(device)
        self.gamma_seq = self.gamma_seq.to(device)
        self.sample_shape = torch.Size([self.num_particles], device=device)
        self.action_lows, self.action_highs, self.action_range = self.action_lows.to(device), self.action_highs.to(device), self.action_range.to(device)
        self.mvn = MultivariateNormal(
            loc=torch.zeros(self.horizon * self.action_dim, device=device), 
            covariance_matrix=self.init_cov * torch.eye(self.horizon * self.action_dim, device=device)) 
        self.sampling_policy.to(device)
        self.device = device

    def is_cuda(self):
        # Check if any of the networks are on GPU
        model_cuda = [model.is_cuda() for model in self.learned_model]
        model_cuda = any(model_cuda)
        # baseline_cuda = next(self.baseline.model.parameters()).is_cuda
        return any([model_cuda])

    def train_step(self):
        #train the heuristic (and sampling policy?) here
        pass

    def get_action(self, observation):
        act_seq = self.optimize(torch.as_tensor(observation, device=self.device).float())
        action = act_seq[0]
        # action = self.env.action_space.sample()
        return action.cpu().numpy()

    def optimize(self, observation):
        # inp_dtype = observation.dtype
        # inp_device = observation.device

        if self.hotstart:
            self.shift(self.shift_steps)
        else:
            self.reset_distribution()
        
        # with torch.cuda.amp.autocast(enabled=False):
        with torch.no_grad():
            for _ in range(self.n_iters):
                # generate random simulated trajectories
                paths = self.generate_rollouts(observation)

                # update distribution parameters
                with profiler.record_function("mppi_update"):
                    self.update_distribution(paths)
                # info['rollout_time'] += paths['rollout_time']

                # check if converged
                if self.check_convergence():
                    break

        self.trajectories = paths
        curr_action_seq = self.get_action_seq(observation, mode=self.sample_mode)
        #calculate optimal value estimate if required
        # info['entropy'].append(self.entropy)

        self.num_steps += 1

        return curr_action_seq


    def get_action_seq(self, observation, mode='mean'):
        if mode == 'best_mean':
            returns = self.evaluate_act_sequences(observation, self.mean_action.unsqueeze(1))
            best_idx = torch.argmax(returns)
            act_seq = self.mean_action[best_idx].clone()
        elif mode == 'worst_mean':
            returns = self.evaluate_act_sequences(observation, self.mean_action.unsqueeze())
            worst_idx = torch.argmin(returns)
            act_seq = self.mean_action[worst_idx].clone()
        elif mode == 'random_mean':
            rand_idx = np.random.randint(self.num_models)
            act_seq = self.mean_action[rand_idx]
        elif mode == 'average_mean':
            act_seq = torch.mean(self.mean_action, dim=0)
        elif mode == 'best_mean_sample':
            raise ValueError('To be implemented')
        elif mode == 'random_mean_sample':
            raise ValueError('To be implemented')
        else:
            raise ValueError('Sampling mode not recognized')
        
        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

        return act_seq

    # def generate_rollouts(self, observation):
    #     # action_batch = self.sample_actions()
    #     paths = self.rollout_actions(observation) #, action_batch)
    #     return paths
    
    def generate_rollouts(self, start_obs): # , action_batch):
        #Mohak - right now we only use the first model.
        # paths = []
        ts = timer.time()
        
        # rand_indices = [random.randint(0, len(self.learned_model)-1) for _ in range(self.num_particles)]
        # curr_model = self.learned_model[0]
        # for b in range(self.num_particles):
            #choose random model
            # curr_model = self.learned_model[0]
            # rand_idx = np.random.randint(0, len(self.learned_model))
            # # print('idx', idx)
            # curr_model = self.learned_model[rand_idx]

        #sample open-loop actions using current means
        open_loop_actions = self.sample_actions_batch()

        rollouts = []
        for m, model in enumerate(self.learned_model):
            obs = torch.zeros(self.num_particles, self.horizon, self.observation_dim, device=self.device)
            act = torch.zeros(self.num_particles, self.horizon, self.action_dim, device=self.device)

            st = start_obs.clone().unsqueeze(0).repeat(self.num_particles, 1)
            for t in range(self.horizon):
                a_policy, _ = self.sample_actions_policy(st) #closed-loop actions from behavior policy
                a_ol = open_loop_actions[m,:,t] #get corresponding open-loop action from shifted mean
                at = (1.0-self.mixing_factor) * a_policy + self.mixing_factor * a_ol #mixed action
                stp1 = model.forward(st, at)
                obs[:,t,:] = st.clone()
                act[:,t,:] = at.clone()
                # obs.append(st.clone())
                st = stp1

            model_rollouts = dict(observations=obs, actions=act)

            #here we compute rewards and terminations for the paths
            # use learned reward function if available
            if model.learn_reward:
                model.compute_path_rewards(model_rollouts)
            else:
                model_rollouts = self.reward_function(model_rollouts)
                # print('rew1')
                # print(path['rewards'])
                # path2 = dict(observations=obs.cpu().numpy(),
                #             actions=action_batch.cpu().numpy())
                # path2 = self.reward_function2(path2)
                # print(path2['rewards'])
                # scale by action repeat if necessary
                model_rollouts["rewards"] = model_rollouts["rewards"] * self.env.act_repeat
            rollouts.append(model_rollouts)
        
        #concatenate all the rollouts
        paths = dict()
        for key in rollouts[0].keys():
            paths[key] = torch.cat([rollout[key].unsqueeze(0) for rollout in rollouts])
        
        # paths["actions"] = action_batch
        # print(paths["observations"][1].shape, paths["observations"][0].shape, paths["rewards"].shape)
        # # print(paths["observations"][0])
        # print("1")
        # print(torch.argmin(paths["observations"][1] - paths["observations"][0]))
        # input('...')
        # for m in range(len(self.learned_model)):

        #     print(rollouts[m]["rewards"])
        #     input('....')
            # #convert rollouts to list of dicts
            # num_traj, horizon, state_dim = rollouts['observations'].shape
            # for i in range(num_traj):
            #     path = dict()
            #     for key in rollouts.keys():
            #         path[key] = rollouts[key][i, ...].cpu().numpy()
            #     paths.append(path)

        # # NOTE: If tasks have termination condition, we will assume that the env has
        # # a function that can terminate paths appropriately.
        # # Otherwise, termination is not considered.

        # if callable(self.termination_function): 
        #     paths = self.termination_function2(paths)
        # else:
        #     # mark unterminated
        #     for path in paths: path['terminated'] = False


        if callable(self.termination_function):
            paths = self.termination_function(paths)
        else:
            paths["terminated"] = torch.zeros(paths["observations"].shape[0])
       
        # additional truncation based on error in the ensembles
        # if self.truncate_lim is not None and len(self.learned_model) > 1:
        #     pred_err_list = []
        #     for path in paths:
        #         pred_err = np.zeros(path['observations'].shape[0] - 1)
        #         s = path['observations'][:-1]
        #         a = path['actions'][:-1]
        #         s_next = path['observations'][1:]
        #         for idx_1, model_1 in enumerate(self.learned_model):
        #             pred_1 = model_1.predict(s, a)
        #             for idx_2, model_2 in enumerate(self.learned_model):
        #                 if idx_2 > idx_1:
        #                     pred_2 = model_2.predict(s, a)
        #                     # model_err = np.mean((pred_1 - pred_2)**2, axis=-1)
        #                     model_err = np.linalg.norm(
        #                         (pred_1-pred_2), axis=-1)
        #                     pred_err = np.maximum(pred_err, model_err)
        #         pred_err_list.append(np.expand_dims(pred_err, 0).copy())
        #         violations = np.where(pred_err > self.truncate_lim)[0]
        #         truncated = (not len(violations) == 0)
        #         T = violations[0] + \
        #             1 if truncated else path['observations'].shape[0]
        #         # we don't want corner cases of very short truncation
        #         T = max(4, T)
        #         path["observations"] = path["observations"][:T]
        #         path["actions"] = path["actions"][:T]
        #         path["rewards"] = path["rewards"][:T]
        #         if truncated:
        #             path["rewards"][-1] += self.truncate_reward
        #         path["terminated"] = False if T == path['observations'].shape[0] else True

        # pred_err_arr_np=  np.concatenate(pred_err_list, axis=0)
        # print(pred_err_arr_np.shape)
        # input('...')
        # print('from torch')
        if self.truncate_lim is not None and len(self.learned_model) > 1:
            pred_err = torch.zeros(self.num_models, self.num_particles, self.horizon-1, device=self.device)
            s = paths['observations'][:,:,:-1]
            a = paths['actions'][:,:,:-1] #.unsqueeze(0).repeat(self.num_models, 1, 1, 1)
            s_next = paths['observations'][:,:,:-1]
            for idx_1, model_1 in enumerate(self.learned_model):
                pred_1 = model_1.forward(s,a)
                for idx_2, model_2 in enumerate(self.learned_model):
                    if idx_2 > idx_1:
                        pred_2 = model_2.forward(s,a)
                        model_err = torch.norm(
                            (pred_1 - pred_2), dim=-1, p=2)
                        pred_err = torch.maximum(pred_err, model_err)

            violations = torch.where(pred_err > self.truncate_lim)
            dones = paths["dones"]
            dones[violations] = 1
            dones = torch.cumsum(dones, dim=-1)
            dones[dones > 0] = 1.0
            paths["dones"] = dones
            paths['terminated'] = torch.any(dones, dim=-1)
            paths['rewards'] += paths["dones"] * self.truncate_reward
            paths = self.compute_discounted_return(paths)

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)


        # # remove paths that are too short
        # paths = [path for path in paths if path['observations'].shape[0] >= 5]

        return paths

    def evaluate_act_sequences(self, start_obs, action_batch):
        """
            Takes an action_batch of size [num_models x horizon x d_act]
            and returns the discounted cost to go for each of them 
            as predicted by the corresponding model with proper termination
            using disagreement.
        """
        ts = timer.time()

        rollouts = []


        for m, model in enumerate(self.learned_model):
            obs = torch.zeros(1, self.horizon, self.observation_dim, device=self.device)
            act_b = action_batch[m, :, :, :]
            st = start_obs.clone().unsqueeze(0)
        
            for t in range(self.horizon):
                at = act_b[:, t, :]
                stp1 = model.forward(st, at)
                obs[:, t, :] = st.clone()
                st = stp1

            model_rollouts = dict(observations=obs, actions=act_b)

            #here we compute rewards and terminations for the paths
            # use learned reward function if available
            if model.learn_reward:
                model.compute_path_rewards(model_rollouts)
            else:
                model_rollouts = self.reward_function(model_rollouts)
                # scale by action repeat if necessary
                model_rollouts["rewards"] = model_rollouts["rewards"] * \
                    self.env.act_repeat
            rollouts.append(model_rollouts)

        #concatenate all the rollouts
        paths = dict()
        for key in rollouts[0].keys():
            paths[key] = torch.cat([rollout[key].unsqueeze(0)
                                   for rollout in rollouts])


        # # NOTE: If tasks have termination condition, we will assume that the env has
        # # a function that can terminate paths appropriately.
        # # Otherwise, termination is not considered.
        if callable(self.termination_function):
            paths = self.termination_function(paths)
        else:
            paths["terminated"] = torch.zeros(paths["observations"].shape[0])

        if self.truncate_lim is not None and len(self.learned_model) > 1:
            pred_err = torch.zeros(
                self.num_models, 1, self.horizon-1, device=self.device)
            
            s = paths['observations'][:, :, :-1]
            a = paths['actions'][:, :, :-1]
            s_next = paths['observations'][:, :, :-1]
            for idx_1, model_1 in enumerate(self.learned_model):
                pred_1 = model_1.forward(s, a)
                for idx_2, model_2 in enumerate(self.learned_model):
                    if idx_2 > idx_1:
                        pred_2 = model_2.forward(s, a)
                        model_err = torch.norm(
                            (pred_1 - pred_2), dim=-1, p=2)
                        pred_err = torch.maximum(pred_err, model_err)

            violations = torch.where(pred_err > self.truncate_lim)
            dones = paths["dones"]
            dones[violations] = 1
            dones = torch.cumsum(dones, dim=-1)
            dones[dones > 0] = 1.0
            paths["dones"] = dones
            # print(paths["dones"].shape, paths["rewards"].shape)
            paths['terminated'] = torch.any(dones, dim=-1)
            paths['rewards'] += paths["dones"] * self.truncate_reward
            paths = self.compute_discounted_return(paths)
        
        if self.save_logs:
            self.logger.log_kv('time_mean_evaluation', timer.time() - ts)


        return paths["discounted_return"]

    def update_distribution(self, paths):
        actions = paths['actions'] #torch.cat([path['actions'].unsqueeze(0) for path in paths], dim=0)
        # discounted_return = torch.max(paths["discounted_return"], dim=0)[0]
        w = torch.softmax((1.0/self.beta) * paths["discounted_return"], dim=-1)
        #Update mean
        weighted_seq = w.T * actions.T
        new_mean = torch.sum(weighted_seq.T, dim=1)
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean

        # delta = actions - self.mean_action.unsqueeze(0)

    def compute_discounted_return(self, paths):
        rewards = paths['rewards']
        dones = paths['dones']
        discounted_return = torch.zeros(rewards.shape[0], rewards.shape[1], device=self.device)
        for i in range(self.horizon):
            discounted_return += (self.gamma**i) * rewards[:, :, i] * (1.0 - dones[:, :,i]) 
        paths["discounted_return"] = discounted_return
        return paths

    # def compute_weights(self, paths):
    #     # rewards = paths['rewards']
    #     # rewards = torch.cat([path['rewards'].unsqueeze(0) for path in paths], dim=0)
    #     # traj_rewards = cost_to_go(rewards, self.gamma_seq)
    #     # traj_rewards = traj_rewards[:, 0]
    #     # num_paths = len(paths)
    #     # path_rewards = []
    #     # for path in paths:
    #     #     path_reward = 0.0
    #     #     for t in range(len(path['rewards'])):
    #     #         path_reward += (self.gamma ** t) * path['rewards'][t]
    #     #     path_rewards.append(path_reward)
        
    #     # path_rewards = torch.tensor(path_rewards, device=self.device)

    #     # paths = self.compute_discounted_return(paths)
    #     # print(traj_rewards)
    #     #control_costs = self._control_costs(actions)
    #     # total_costs = traj_costs  # + self.beta * control_costs

    #     # #calculate soft-max
    #     w = torch.softmax((1.0/self.beta) * paths["discounted_return"], dim=0)
    #     return w

    def control_costs(self, delta):
        pass

    # def sample_actions(self, observations):
    #     #mix actions from gaussian based on current mean
    #     #and sampling policy
    #     open_loop_actions = self.sample_actions_batch()
    #     policy_actions, _ = self.sample_actions_policy(observations)
    #     print(open_loop_actions.shape, policy_actions.shape)
    #     actions = (1.0 - self.mixing_factor) * policy_actions + self.mixing_factor * open_loop_actions
    #     return actions

    def sample_actions_batch(self):
        """
        sample batch of open-loop actions from current mean
        """
        # delta = self.mvn.sample(self.sample_shape)
        # delta = delta.view(delta.shape[0], self.horizon, self.action_dim)
        eps = torch.randn(self.num_models, self.num_particles, self.horizon, self.action_dim, device=self.device) 
        action_batch = self.mean_action.unsqueeze(1).repeat(1,self.num_particles, 1, 1) + self.init_std * eps
        action_batch = self.filter_actions(action_batch)
        return action_batch

    def sample_actions_policy(self, observations):
        # assert type(observation) == np.ndarray
        # if self.device != 'cpu':
        #     print("Warning: get_action function should be used only for simulation.")
        #     print("Requires policy on CPU. Changing policy device to CPU.")
        #     self.to('cpu')
        # o = np.float32(observation.reshape(1, -1))
        # self.obs_var.data = torch.from_numpy(o)
        with torch.no_grad():
            mean = self.sampling_policy.forward(observations)
            eps = torch.randn_like(mean)
            noise = torch.exp(self.sampling_policy.log_std) * eps
            action = mean + noise
        return action, {'mean': mean, 'log_std': self.sampling_policy.log_std_val, 'evaluation': mean}

    def filter_actions(self, action_batch):
        beta_0, beta_1, beta_2 = self.filter_coeffs
        action_batch[:,:,0] *=  (beta_0 + beta_1 + beta_2)
        action_batch[:,:,1] = beta_0 * action_batch[:,:,1] + (beta_1 + beta_2) * action_batch[:,:,0]
        for i in range(2, action_batch.shape[2]):
            action_batch[:,:,i] = beta_0*action_batch[:,:,i] + beta_1*action_batch[:,:,i-1] + beta_2*action_batch[:,:,i-2]
        return action_batch


    def shift(self, shift_steps):
        self.mean_action = self.mean_action.roll(-shift_steps, 1)

        if self.base_action == 'random':
            # self.mean_action[-1] = self.generate_noise(shape=torch.Size((1, 1)),
            #                                            base_seed=self.seed_val + 123*self.num_steps)
            self.mean_action[-1] = self.action_range * torch.rand(self.action_dim, device=self.device) + self.action_lows
        elif self.base_action == 'null':
            self.mean_action[-shift_steps:].zero_()
        elif self.base_action == 'repeat':
            self.mean_action[-shift_steps:] = self.mean_action[-shift_steps - 1].clone()
        else:
            raise NotImplementedError(
                "invalid option for base action during shift")

    def reset_distribution(self):
        self.mean_action = self.init_mean.clone()
        self.cov_action = self.init_cov.clone() 

    def reset(self):
        self.reset_distribution()
        self.num_steps = 0

    def check_convergence(self):
        return False
    
