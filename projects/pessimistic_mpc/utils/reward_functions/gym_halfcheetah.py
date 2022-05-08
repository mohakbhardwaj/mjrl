import torch
import numpy as np

# observaion mask for scaling
# 1.0 for positions and dt=0.02 for velocities
obs_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 0.02, 0.02, 0.02, 0.02,
                     0.02, 0.02, 0.02, 0.02, 0.02])


def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_models, num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_models, num_traj, horizon)
    # obs = torch.clip(paths["observations"], -10.0, 10.0)
    obs = paths["observations"]#.clip(-10.0, 10.0)
    act = paths["actions"].clip(-1.0, 1.0)
    reward_run = obs[:, :, :, -9] / 0.02 #x velocity
    reward_ctrl = -0.1 * torch.square(act).sum(axis=-1)
    rewards = reward_run + reward_ctrl
    paths["rewards"] = rewards  # if rewards.shape[0] > 1 else rewards.ravel()

    return paths


def termination_function(paths):
    # path has 2 keys: observations, actions
    # path["observations"] : (num_models, num_traj, horizon, obs_dim)
    # return paths that contain terminals in path["terminals"]
    obs = paths["observations"]
    dones = torch.zeros(obs.shape[0], obs.shape[1],
                        obs.shape[2], device=obs.device)

    paths['dones'] = dones
    paths['terminated'] = torch.any(dones, dim=-1)

    return paths

termination_function2 = None

if __name__ == "__main__":
    #test the above functions
    pass
