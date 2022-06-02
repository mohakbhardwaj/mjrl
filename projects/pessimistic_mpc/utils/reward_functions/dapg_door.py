import numpy as np
import torch 

# obs_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.02,
#                     0.02, 0.02, 0.02, 0.02, 0.02])
obs_mask = np.ones(39)

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_models, num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_models, num_traj, horizon)
    # obs = torch.clip(paths["observations"], -10.0, 10.0)
    # def get_obs(self):
    obs = paths["observations"]
    act = paths["actions"]
    horizon = obs.shape[2]
    obj_pos = obs[:,:,:,24:27]
    obj_vel = obs[:,:,:,27:33]
    obj_orien = obs[:,:,:,33:36]
    desired_orien = obs[:,:,:,36:39]
    obj_pos_goal_disp = obs[:,:,:,39:42]
    obj_orien_goal_disp = obs[:,:,:,42:]

    #ignore desired orien predictions for now
    # start_desired_orien = desired_orien[:,:,0].unsqueeze(-2)
    # desired_orien = start_desired_orien.repeat(1,1,horizon,1)
    
    dist = torch.norm(obj_pos_goal_disp, dim=-1)
    # pos cost
    dist_reward = -1.0 * dist
    # orien cost
    orien_similarity = torch.sum(obj_orien * desired_orien, dim=-1)
    
    rewards = dist_reward + orien_similarity

    # bonus for being close to desired orientation
    bonus_mask_1 = torch.logical_and(dist < 0.075 , orien_similarity > 0.9)
    bonus_mask_2 = torch.logical_and(dist < 0.075, orien_similarity > 0.95)
    rewards[bonus_mask_1] += 10.0
    rewards[bonus_mask_2] += 50.0
 
    #penalty for dropping pen
    obj_z = obj_pos[:,:,:,2]
    rewards[obj_z < 0.075] -= 5.0

    # if ADD_BONUS_REWARDS:
    #     # bonus for being close to desired orientation
    #     if dist < 0.075 and orien_similarity > 0.9:
    #         reward += 10
    #     if dist < 0.075 and orien_similarity > 0.95:
    #         reward += 50

    # # penalty for dropping the pen
    # done = False
    # if obj_pos[2] < 0.075:
    #     reward -= 5
    #     done = True if not starting_up else False


    # paths["rewards"] = rewards  # if rewards.shape[0] > 1 else rewards.ravel()

    return rewards


def termination_function(paths):
    # path has 2 keys: observations, actions
    # path["observations"] : (num_models, num_traj, horizon, obs_dim)
    # return paths that contain terminals in path["terminals"]
    obs = paths["observations"]
    dones = torch.zeros(obs.shape[0], obs.shape[1],
                        obs.shape[2], device=obs.device)
    
    obj_pos = obs[:,:,:,24:27]
    obj_z = obj_pos[:,:,:,2]
    dones[obj_z < 0.075] = 1.0
    dones = torch.cumsum(dones, dim=-1)
    dones[dones > 0] = 1.0
    # paths['dones'] = dones
    # paths['terminated'] = torch.any(dones, dim=-1)

    return dones










