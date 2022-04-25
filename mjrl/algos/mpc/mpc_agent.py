import torch
import torch.autograd.profiler as profiler
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

from mjrl.utils.gym_env import GymEnv
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.algos.mpc.mpc_utils import scale_ctrl, cost_to_go, cost_to_go_debug
from mjrl.utils.logger import DataLog


class MPCAgent():
    def __init__(self, 
    env,
    learned_model=None,
    sampling_policy=None,
    mpc_params=None,
    reward_function=None,
    termination_function=None,
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
        
        self.sampling_policy = sampling_policy
        # self.refine, self.kappa, self.plan_horizon, self.plan_paths = refine, kappa, plan_horizon, plan_paths
        if mpc_params is None:
            print("Algorithm requires mpc params")
            quit()
        else:
            self.mpc_params = mpc_params
        self.reward_function, self.termination_function = reward_function, termination_function
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

        assert self.sample_mode in ['mean', 'sample']

        #set the initial mean and covariance of mpc
        self.init_mean = torch.zeros((self.horizon, self.action_dim))
        self.init_cov = torch.tensor(self.init_cov)
        #calling reset distribution sets current mean and cov 
        #to initial values
        self.reset_distribution()
        self.gamma_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma] * (self.horizon - 1)),dim=0).reshape(1, self.horizon)
        self.mvn = MultivariateNormal(
            loc=torch.zeros(self.horizon * self.action_dim), 
            covariance_matrix=self.init_cov * torch.eye(self.horizon * self.action_dim)) 
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
        self.mean_action, self.cov_action = self.mean_action.to(device), self.cov_action.to(device)
        self.gamma_seq = self.gamma_seq.to(device)
        self.sample_shape = torch.Size([self.num_particles], device=device)
        self.action_lows, self.action_highs, self.action_range = self.action_lows.to(device), self.action_highs.to(device), self.action_range.to(device)
        self.mvn = MultivariateNormal(
            loc=torch.zeros(self.horizon * self.action_dim, device=device), 
            covariance_matrix=self.init_cov * torch.eye(self.horizon * self.action_dim, device=device)) 
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
        curr_action_seq = self.get_action_seq(mode=self.sample_mode)
        #calculate optimal value estimate if required
        # info['entropy'].append(self.entropy)

        self.num_steps += 1

        return curr_action_seq


    def get_action_seq(self, mode='mean'):
        if mode == 'mean':
            act_seq = self.mean_action.clone()
        elif mode == 'sample':
            raise ValueError('To be implemented')
        
        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

        return act_seq

    def generate_rollouts(self, observation):
        action_batch = self.sample_actions()
        paths = self.rollout_actions(observation, action_batch)
        return paths
    
    def rollout_actions(self, start_obs, action_batch):
        #Mohak - right now we only use the first model.
        curr_model = self.learned_model[0]
        obs = torch.zeros(self.num_particles, self.horizon, self.observation_dim, device=self.device)
        st = start_obs.unsqueeze(0).repeat(self.num_particles, 1)
        
        for t in range(self.horizon):
            at = action_batch[:,t,:]
            stp1 = curr_model.forward(st, at)
            obs[:,t,:] = st.clone()
            # obs.append(st.clone())
            st = stp1
        
        #TODO: Right now the reward_function is defined in numpy only. We need to translate it 
        # to torch and compute in batch
        path = dict(observations=obs.cpu().numpy(), actions=action_batch.cpu().numpy())

        #here we compute rewards and terminations for the paths

        # use learned reward function if available
        if curr_model.learn_reward:
            curr_model.compute_path_rewards(path)
        else:
            path = self.reward_function(path)
            # scale by action repeat if necessary
            path["rewards"] = path["rewards"] * self.env.act_repeat
        # num_traj, horizon, state_dim = rollouts['observations'].shape
        # for i in range(num_traj):
        #     path = dict()
        #     for key in rollouts.keys():
        #         path[key] = rollouts[key][i, ...]
        #     paths.append(path)

        path['observations'] = torch.from_numpy(path['observations']).float().to(self.device)
        path['actions'] = torch.from_numpy(path['actions']).float().to(self.device)
        path['rewards'] = torch.from_numpy(path['rewards']).float().to(self.device)

        return path

    def update_distribution(self, paths):
        actions = paths['actions']
        w = self.compute_traj_weights(paths)
        
        #Update mean
        weighted_seq = w.T * actions.T
        new_mean = torch.sum(weighted_seq.T, dim=0)
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean

        # delta = actions - self.mean_action.unsqueeze(0)


    def compute_traj_weights(self, paths):
        rewards = paths['rewards']
        # traj_rewards = cost_to_go(rewards, self.gamma_seq)
        # if not self.time_based_weights: traj_costs = traj_costs[:,0]
        # traj_rewards = traj_rewards[:, 0]
        traj_rewards = cost_to_go_debug(rewards, self.gamma)
        # print(traj_rewards)
        #control_costs = self._control_costs(actions)
        # total_costs = traj_costs  # + self.beta * control_costs

        # #calculate soft-max
        w = torch.softmax((1.0/self.beta) * traj_rewards, dim=0)
        return w

    def control_costs(self, delta):
        pass

    def sample_actions(self):
        """
        Mohak TODO: use generate_perturbed_actiions 
        with filter_coeffs for smooth sampling.
        """
        delta = self.mvn.sample(self.sample_shape)
        delta = delta.view(delta.shape[0], self.horizon, self.action_dim)
        action_batch = self.mean_action + delta
        # print('unfiltered actions')
        # print(action_batch)
        # input('...')
        action_batch = self.filter_actions(action_batch)
        # print('filtered actions')
        # print(action_batch)
        # input('...')
        return action_batch

    def filter_actions(self, action_batch):
        beta_0, beta_1, beta_2 = self.filter_coeffs
        action_batch[0] = action_batch[0] * (beta_0 + beta_1 + beta_2)
        action_batch[1] = beta_0 * action_batch[1] + (beta_1 + beta_2) * action_batch[0]
        for i in range(2, action_batch.shape[0]):
            action_batch[i] = beta_0*action_batch[i] + beta_1*action_batch[i-1] + beta_2*action_batch[i-2]
        return action_batch


    def shift(self, shift_steps):

        self.mean_action = self.mean_action.roll(-shift_steps, 0)

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
    
