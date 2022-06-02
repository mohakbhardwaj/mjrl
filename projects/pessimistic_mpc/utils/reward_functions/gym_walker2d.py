import torch
import numpy as np

# observaion mask for scaling
# 1.0 for positions and dt=0.02 for velocities
obs_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.02,
                    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])



def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_models, num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_models, num_traj, horizon)
    # obs = torch.clip(paths["observations"], -10.0, 10.0)
    # act = paths["actions"].clip(-1.0, 1.0)
    obs = paths["observations"]
    act = paths["actions"]
    vel_x = obs[:, :, :, -9] / 0.02
    power = torch.square(act).sum(axis=-1)
    height = obs[:, :, :, 0]
    ang = obs[:, :, :, 1]
    alive_bonus = 1.0 * (height > 0.8) * (height < 2.0) * (torch.abs(ang) < 1.0)
    rewards = vel_x + alive_bonus - 1e-3*power

    return rewards





def termination_function(paths):
    # path has 2 keys: observations, actions
    # path["observations"] : (num_models, num_traj, horizon, obs_dim)
    # return paths that contain terminals in path["terminals"]
    obs = paths["observations"]
    height = obs[:,:,:,0]
    angle = obs[:,:,:,1]
    dones = torch.zeros(obs.shape[0], obs.shape[1], obs.shape[2], device=obs.device)
    dones[height <= 0.8] = 1
    dones[height >= 2.0] = 1
    dones[torch.abs(angle) >= 1.0] = 1
    
    #set all states after the first terminal state to 
    #terminal
    dones = torch.cumsum(dones, dim=-1)
    dones[dones > 0] = 1.0

    # paths['dones'] = dones
    # paths['terminated'] = torch.any(dones, dim=-1)
    return dones

reward_function2 = None
termination_function2 = None
