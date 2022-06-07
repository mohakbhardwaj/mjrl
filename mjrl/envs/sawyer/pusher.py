from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer

class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, randomize_goal=False):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.randomize_goal = randomize_goal
        self.cylinder_pos, self.goal_pos = np.zeros(3), np.ones(3)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/pusher.xml', 5)

        utils.EzPickle.__init__(self)
        self.reset_model()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obj_pos = self.get_body_com("object"),
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        # reward_near = -np.sum(np.abs(vec_1))
        # reward_dist = -np.sum(np.abs(vec_2))
        reward_near = -np.linalg.norm(vec_1)
        goal_dist = np.linalg.norm(vec_2)
        reward_dist = -goal_dist
        reward_ctrl = 0.0 #-np.square(a).sum()
        bonus = 5.0 * (goal_dist < 0.05)
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near + bonus
        ob = self._get_obs()
        done = False
        return ob, reward, done, self.get_env_infos()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        if not self.randomize_goal:
            self.goal_pos = np.asarray([0., 0.])
        else:
            self.goal_pos = self.np_random.uniform(low=-0.2, high=0.2, size=2)

        # self.cylinder_pos = np.array(
        #     [-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        while True:
            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=-0.3, high=0, size=1),
                self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        # self.ac_goal_pos = self.get_body_com("goal")

        return self._get_obs()

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat[:7]
        qvel = self.sim.data.qvel.flat[:7]
        obs =  np.concatenate([
            qpos,
            qvel,
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal")
        ])
        # if self.randomize_goal:
            # obs = np.concatenate([obs, self.get_body_com("goal")])
        return obs
    
    def get_obs(self):
        return self._get_obs()

    # def get_env_state(self):
    #     return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
    #                 cylinder_pos=self.cylinder_pos.copy(), goal_pos=self.goal_pos.copy())

   # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    cylinder_pos=self.cylinder_pos.copy(), goal_pos=self.goal_pos.copy())

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        self.cylinder_pos = state['cylinder_pos'].copy()
        self.goal_pos = state['goal_pos'].copy()
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        dist = np.linalg.norm(self.get_body_com("object") - self.get_body_com("goal"))
        goal_achieved = (dist < 0.05)
        return dict(state=self.get_env_state(), goal_achieved=goal_achieved)

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.sim.forward()
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to goal for at least 5 steps
        for path in paths:
            # print('sum', path['env_infos']['goal_achieved'], sum(path['env_infos']['goal_achieved']))
            # input('....')
            if sum(path['env_infos']['goal_achieved']) > 5: num_success += 1
        # print('num success', num_success)
        success_percentage = num_success*100.0/num_paths
        return success_percentage 
        
    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for env")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)




if __name__ == "__main__":
    from mjrl.utils.gym_env import GymEnv
    import mjrl.envs
    e = GymEnv('pusher-v0')
    e.reset()
    while True:
        obs, reward, done, info = e.step(e.action_space.sample())
        e.render()
        if done:
            e.reset()
