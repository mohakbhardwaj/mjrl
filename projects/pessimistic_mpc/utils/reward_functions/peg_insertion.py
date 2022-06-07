import torch
import numpy as np

# observaion mask for scaling
# 1.0 for positions and dt=0.02 for velocities
# obs_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#                      1.0, 1.0, 1.0, 1.0, 1.0])
obs_mask = np.ones(20)
# obs_mask[7:14] = 0.01

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_models, num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_models, num_traj, horizon)
    obs = paths["observations"]
    act = paths["actions"]  # .clip(-1.0, 1.0)
    hand_pos = obs[:, :, :, -6:-3]
    goal_pos = obs[:, :, :, -3:]
    disp = hand_pos - goal_pos

    l1_dist = torch.sum(torch.abs(disp), dim=-1)
    l2_dist = torch.norm(disp, dim=-1)
    bonus = 5.0 * (l2_dist < 0.06)

    # reward_dist = -torch.sum(torch.abs(vec_2), dim=-1)
    rewards = -l1_dist - 5.0*l2_dist + bonus

    return rewards  # if rewards.shape[0] > 1 else rewards.ravel()


def termination_function(paths):
    # path has 2 keys: observations, actions
    # path["observations"] : (num_models, num_traj, horizon, obs_dim)
    # return paths that contain terminals in path["terminals"]
    num_models, num_traj, horizon, _ = paths["observations"].shape
    dones = torch.zeros(num_models, num_traj, horizon,
                        device=paths["observations"].device)
    return dones


def reward_function2(paths):
    return None


def termination_function2(paths):
    return None
