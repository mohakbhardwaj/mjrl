from abc import abstractmethod, ABC
import torch
from mjrl.algos.mpc.mpc_utils import scale_ctrl
from mjrl.utils.logger import DataLog
torch.set_printoptions(precision=8)


def to_device(d, device):
    """ Move all torch objects in a dict to device. """
    for k, v in d.items():
        if hasattr(v, 'to') and callable(v.to):
            new_v = v.to(device)
            if new_v is not None:  # otherwise, it's in-place.
                d[k] = new_v

class ARFilter:
    """ An Autoregressive Filter. """
    def __init__(self, filter_coeffs):
        self.filter_coeffs = filter_coeffs
        self.memory = None

    def filter(self, x):
        self.update_memory(x)
        x_new = torch.sum(self.memory * self.filter_coeffs.view([-1]+[1]*len(x.shape)), axis=0)
        return x_new

    def update_memory(self, x):
        if self.memory is None:
            self.memory = x.repeat([len(self.filter_coeffs)]+[1]*len(x.shape))
        self.memory = torch.roll(self.memory, 1, dims=0)
        self.memory[0] = x

    def to(self, device):
        to_device(self.__dict__, device)


class AbstractMPCAgent(ABC):
    """ An abstract MPC agent. """

    def __init__(self, *,
                 env,  # TODO really needed?? maybe it should not know the true env
                 dynamics_model,
                 reward_model,
                 termination_model=None,
                 heuristic=None,
                 gamma=0.999, # discount factor
                 horizon=10, #  simulation horizon
                 lambd=0.99, # value blending factor
                 terminal_reward=0.0,  # reward for done flag
                 num_particles=100, # number of rollouts (in parallel)
                 filter_actions=False,
                 squash_fn='clamp',
                 filter_coeffs=(1.0,),
                 n_iters=1, # number of optimization steps
                 optimization_freq=1,
                 seed=0,
                 device='cpu',
                 save_logs=True):

            # Basic components for simulation
            self.dynamics_model = dynamics_model
            self.reward_model = reward_model
            self.termination_model = termination_model
            self.heuristic = heuristic
            self.terminal_reward = terminal_reward
            self.squash_fn = squash_fn
            self.env = env

            # Simulation spec
            self.gamma = gamma  # discount factor
            self.hoirzon = horizon  # rollout horizon
            self.lambd = 1.0 if heuristic is None else lambd  # blending factor with heuristic
            self.num_particles = num_particles  # number of rollouts (in parallel)

            # Extra processing for simulation
            self.filter_actions = filter_actions
            self.filter_coeffs = torch.tensor(filter_coeffs)
            self.ar_filter = ARFilter(self.filter_coeffs)  # to keep tracks of past actions

            # Optimization parameters
            self.optimization_freq = optimization_freq
            self.num_steps = 0  # number of optimization steps performed

            # Misc.
            self.device = device
            self.seed = seed

            ## Initialization
            self.observation_dim = self.env.spec.observation_dim #if env_spec is not None else observation_dim
            self.action_dim = self.env.spec.action_dim #if env_spec is not None else action_dim
            self.action_lows = torch.tensor(self.env.action_space.low)
            self.action_highs = torch.tensor(self.env.action_space.high)
            self.action_range = self.action_highs - self.action_lows

            # Precompute useful attributes
            self.num_models = len(self.dynamics_model)
            self.discount_seq = torch.cumprod(torch.tensor([1.0]+[self.gamma*self.lambd]*(self.horizon)),dim=0).reshape(1, self.horizon+1)
            # Its length is horizon+1 because we want to consider the terminal values too.

            # Cache buffers
            self._buffers = dict()

            self.to(device)
            self.reset()
            if save_logs: self.logger = DataLog()

    def reset(self):
        """ Reset the agent. Should be called at the start of an episode. """
        self.num_steps = 0
        self._scores = []  # TODO perhaps a generic logger

    def get_action(self, observation):
        """ Get action from the agent. """
        action = self.optimize(torch.as_tensor(observation, device=self.device).float())
        self.ar_filter.update_memory(action)  # keep tracks of past actions
        return action.cpu().numpy()

    def optimize(self, observation):
        """ Optimize the policy and return the action for the current observation. """
        if self.num_steps % self.optimization_freq != 0:
            raise NotImplementedError  # TODO
        else:
            # Perform warmstart for the optimization
            self.warmstart()
            infos = []
            for i in range(self.n_iters):
                # Generate the initial observation for the rollout
                start_obs = self.preprocess_and_reset(observation, infos=infos)
                # Generate simulated samples using ollout policy with the dynamics_model and reward_model.
                paths = self.generate_rollouts(start_obs)
                # Label rewards, etc.
                self.postprocess(paths)
                # Update distribution parameters
                info = self.update_policy(start_obs, paths, infos=infos)
                infos.append(info)
                # check if converged
                if self.check_convergence(infos):
                    break
            # Select the action to output
            act, score = self.make_decision(observation, infos)
            self._scores.append(score)  # logging

        self.num_steps += 1
        return act

    def generate_rollouts(self, start_obs, num_particles=None, model_idx=-1):
        """ Rollout a policy to generate observation-action sequences. """

        num_particles = num_particles or self.num_particles
        obs_buff, act_buff = self._get_buffers(num_particles)

        if self.filter_actions:
            ar_filter = ARFilter(self.filter_coeffs)
            # NOTE Not sure why setting the memory makes it worse
            # if self.ar_filter.memory is not None: # filter_coeff x dim -> filter_coeff x num_models x num_particles x dim
            #     ar_filter.memory = self.ar_filter.memory[:,None, None].repeat(1, self.num_models, num_particles, 1).clone()

        # Rollout the policy.
        ob = start_obs.view(1,1,-1).repeat(self.num_models, num_particles, 1)  # model x particles x dim
        for t in range(self.horizon):
            at = self.rollout_policy(ob, t)
            at = ar_filter.filter(at) if self.filter_actions else at
            ob_next = self.dynamics_function(ob, at)
            if model_idx >-1:
                ob_next[:] = ob_next[model_idx] # rollout using a specific model
            obs_buff[:,:, t,:] = ob  # model x particles x horizon x dim
            act_buff[:,:, t,:] = at  # model x particles x horizon x dim
            ob = ob_next
        obs_buff[:,:, t,:] = ob

        # TODO generic buffer
        paths = dict(observations=obs_buff[:,:,:-1].clone(),
                     next_observations=obs_buff[:,:,1:].clone(),
                     actions=act_buff.clone())
        return paths

    def postprocess(self, paths):
        """ Postprocess the observation-action sequence, e.g., by computing the rewards. """
        # Label rewards and done flags
        paths['dones'] = self.termination_function(paths)
        paths['rewards'] = self.reward_function(paths)
        self.compute_shaped_reward_and_return(paths)

    ## Below are methods that may require overwrites
    def warmstart(self):
        pass

    def preprocess_and_reset(self, observation, infos=None):
        """ Preprocess before each simulated rollout and return the initial observation. """
        return observation

    def dynamics_function(self, observation, action):
        action = scale_ctrl(action, self.action_lows, self.action_highs, squash_fn=self.squash_fn)
        return self.dynamics_model.forward(observation, action) # model x particles x dim

    def reward_function(self, paths):
        rewards = self.reward_model(paths)  # compute paths["rewards"]
        return rewards * self.env.act_repeat

    def compute_shaped_reward_and_return(self, paths):
        """ Reshape the reward based on heuristic and compute the return while considering timeout. """
        num_models, num_particles, horizon, _ = paths["observations"].shape
        dones, rewards = paths['dones'], paths['rewards']
        terminal_values = torch.zeros(num_models, num_particles,1)

        # Reshape the reward
        if self.heuristic is not None:
            value_preds = self.heuristic(paths["next_observations"].view(-1,self.observation_dim)).view(paths['rewards'].shape) #model*particles*horizon x observation
            rewards = rewards + (1.0-self.lambd) * self.gamma * value_preds
            terminal_values = value_preds[:,:,-1:]

        # Make sure they are consitent with the done flags
        rewards = rewards*(1.0-dones)+ self.terminal_reward*dones
        terminal_values = terminal_values*(1.0-dones[:,:,-1:]) + self.terminal_reward/(1-self.gamma*self.lambd)*dones[:,:,-1:]
        paths["discounted_return"] = (self.discount_seq * torch.cat([rewards, terminal_values],dim=-1)).sum(axis=-1)

        # Logging
        paths['shaped_rewards'] = rewards
        paths['value_preds'] = value_preds # model x particles x horizon

    def termination_function(self, paths):
        dones = paths.get('dones', None)
        if callable(self.termination_model):
            extra_dones = self.termination_model(paths)
            dones = extra_dones if dones is None else extra_dones + dones
        # Make sure once the agent is in the absorbing observation, it stays there forever.
        dones = torch.cumsum(dones, dim=-1)
        dones[dones > 0] = 1.0
        return dones

    def check_convergence(self, infos):
        return False

    ## Below are methods that require algorithm-specific instantiations
    @abstractmethod
    def rollout_policy(self, observation, t):
        pass

    @abstractmethod
    def update_policy(self, observation, paths, infos=None):
        pass

    @abstractmethod
    def make_decision(self, observation, infos):
        pass

    # Useful helper methods
    def to(self, device):
        """ Move all torch object to device. """
        self.device = device
        to_device(self.__dict__, device)

    def _get_buffers(self, num_particles):
        key = str(num_particles)
        if key not in self._buffers:
            buffer = (torch.zeros(self.num_models, num_particles, self.horizon+1, self.observation_dim, device=self.device),  # observation
                      torch.zeros(self.num_models, num_particles, self.horizon, self.action_dim, device=self.device))  # action
            self._buffers[key] =  buffer
        return self._buffers[key]