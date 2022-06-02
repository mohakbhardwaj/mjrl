import numpy as np
import torch 

# obs_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.02,
#                     0.02, 0.02, 0.02, 0.02, 0.02])
obs_mask = np.ones(45)

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_models, num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_models, num_traj, horizon)
    # obs = torch.clip(paths["observations"], -10.0, 10.0)
    # def get_obs(self):

    #     qp = self.data.qpos.ravel()
    #     obj_vel = self.data.qvel[-6:].ravel()
    #     obj_pos = self.data.body_xpos[self.obj_bid].ravel()
    #     desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
    #     obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
    #     desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
    #     return np.concatenate([qp[:-6], obj_pos, obj_vel, obj_orien, desired_orien,
    #                            obj_pos-desired_pos, obj_orien-desired_orien])


        # obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        # desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        # obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
        # desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length

    obs = paths["observations"]
    act = paths["actions"]
    obj_pos = obs[:,:,:,24:27]
    obj_vel = obs[:,:,:,27:33]
    obj_orien = obs[:,:,:,33:36]
    desired_orien = obs[:,:,:,36:39]
    obj_pos_goal_disp = obs[:,:,:,39:42]
    obj_orien_goal_disp = obs[:,:,:,42:]

    dist = torch.norm(obj_pos_goal_disp, dim=-1)
    # pos cost
    dist_reward = -1.0 * dist
    # orien cost
    orien_similarity = torch.sum(obj_orien * desired_orien, dim=-1)
    
    rewards = dist_reward + orien_similarity

    # bonus for being close to desired orientation
    #     if dist < 0.075 and orien_similarity > 0.9:
    #         reward += 10
    #     if dist < 0.075 and orien_similarity > 0.95:
    #         reward += 50
    bonus_mask_1 = torch.logical_and(dist < 0.075 , orien_similarity > 0.9)
    bonus_mask_2 = torch.logical_and(dist < 0.075, orien_similarity > 0.95)
    rewards[bonus_mask_1] += 10.0
    rewards[bonus_mask_2] += 50.0
 
    #penalty for dropping pen
    obj_z = obj_pos[:,:,:,2]
    rewards[obj_z < 0.075] -= 5.0

    # # pos cost
    # dist = np.linalg.norm(obj_pos-desired_loc)
    # reward = -dist
    # # orien cost
    # orien_similarity = np.dot(obj_orien, desired_orien)
    # reward += orien_similarity

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

    # goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False

    # return self.get_obs(), reward, done, dict(goal_achieved=goal_achieved)


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










