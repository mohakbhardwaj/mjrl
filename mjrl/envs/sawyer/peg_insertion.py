import numpy as np
import os
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer


class PegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, sensor_noise=False):
        self.peg_sid = -2
        self.target_sid = -1
        self.true_target_sid = -3

        #used to simulate noise in hole location if needed
        self.sensor_noise = sensor_noise

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self, curr_dir+'/assets/peg_insertion.xml', 2)
        utils.EzPickle.__init__(self)
        self.peg_sid = self.model.site_name2id("peg_bottom")
        self.target_sid = self.model.site_name2id("target")
        self.true_target_sid = self.model.site_name2id("true_target")
        self.init_body_pos = self.model.body_pos.copy()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self.get_obs()
        reward = self.get_reward(obs, a)
        if self.sensor_noise:
            return self.get_noisy_obs(), reward, False, self.get_env_infos()
        return obs, reward, False, self.get_env_infos()

    def get_obs(self):
        target_pos = self.data.site_xpos[self.target_sid]
        hand_pos = self.data.site_xpos[self.peg_sid]
        # target_to_hand = hand_pos - target_pos
        # l1_dist = np.sum(np.abs(target_to_hand))
        # l2_dist = np.linalg.norm(target_to_hand)
        # self.data.qvel.copy().flat,
        qpos = self.sim.data.qpos.copy().flat
        qvel = self.sim.data.qvel.copy().flat
        return np.concatenate([
            qpos,
            qvel,
            # target_to_hand.copy(),
            # [l1_dist], [l2_dist],
            hand_pos.copy(),
            target_pos.copy(),
        ])

    def _get_obs(self):
        return self.get_obs()


    def get_noisy_obs(self):
        target_pos = self.data.sensordata  # self.biased_target_pos.copy()
        # if self.obs_error: target_pos += self.np_random.normal(0.0, self.noise_std, size=target_pos.shape)
        hand_pos = self.data.site_xpos[self.peg_sid]
        qpos = self.sim.data.qpos.copy().flat
        qvel = self.sim.data.qvel.copy().flat

        # target_to_hand = hand_pos - target_pos
        # l1_dist = np.sum(np.abs(target_to_hand))
        # l2_dist = np.linalg.norm(target_to_hand)
        # self.data.qvel.copy().flat,

        return np.concatenate([
            qpos,
            qvel,
            # target_to_hand.copy(),
            # [l1_dist], [l2_dist],
            hand_pos.copy(),
            target_pos.copy(),
        ])

    def get_reward(self, obs, act=None):
        obs = np.clip(obs, -10.0, 10.0)
        if len(obs.shape) == 1:
            # vector obs, called when stepping the env
            hand_pos = obs[-6:-3]
            target_pos = obs[-3:]
            l1_dist = np.sum(np.abs(hand_pos - target_pos))
            l2_dist = np.linalg.norm(hand_pos - target_pos)
        else:
            obs = np.expand_dims(obs, axis=0) if len(obs.shape) == 2 else obs
            hand_pos = obs[:, :, -6:-3]
            target_pos = obs[:, :, -3:]
            l1_dist = np.sum(np.abs(hand_pos - target_pos), axis=-1)
            l2_dist = np.linalg.norm(hand_pos - target_pos, axis=-1)

        bonus = 5.0 * (l2_dist < 0.06)  # 10.0, 5.0
        reward = - l1_dist - 5.0 * l2_dist + bonus
        return reward

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        rewards = self.get_reward(obs)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self):
        # Randomize goal position
        goal_y = self.np_random.uniform(
            low=0.3, high=0.5)  # (low=0.1, high=0.3)#
        try:
            self.model.body_pos[-1,
                                1] = self.init_body_pos[-1, 1] + (goal_y-0.29)
            self.model.body_pos[-2,
                                1] = self.init_body_pos[-2, 1] + (goal_y-0.29)
            self.model.body_pos[-3,
                                1] = self.init_body_pos[-3, 1] + (goal_y-0.29)
            self.sim.forward()
            # self.model.site_pos[self.model.site_name2id("true_target")] = self.data.site_xpos[self.target_sid].copy() #only used for rendering purpose
            # only used for rendering purpose
            self.model.site_pos[self.target_sid] = self.data.site_xpos[self.true_target_sid].copy(
            )

        except:
            pass

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        self.target_reset()
        return self.get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        true_target_pos = self.model.body_pos[-1].copy()
        if self.sensor_noise:
            target_pos = self.data.sensordata
        else:
            # self.model.body_pos[-1].copy()
            target_pos = self.data.site_xpos[self.target_sid]
        # if self.obs_error: target_pos += self.np_random.normal(0.0, self.noise_std, size=target_pos.shape)
        # if self.obs_error: target_pos = self.biased_target_pos.copy()

        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos, true_target_pos=true_target_pos)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        true_target_pos = state['true_target_pos']

        self.model.body_pos[-1] = true_target_pos.copy()
        goal_y = true_target_pos[1]
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        self.model.body_pos[-1, 1] = self.init_body_pos[-1, 1] + (goal_y-0.29)
        self.model.body_pos[-2, 1] = self.init_body_pos[-2, 1] + (goal_y-0.29)
        self.model.body_pos[-3, 1] = self.init_body_pos[-3, 1] + (goal_y-0.29)

        self.model.site_pos[self.true_target_sid] = true_target_pos.copy()
        self.model.site_pos[self.target_sid] = target_pos.copy()
        # self.data.site_xpos[self.target_sid] = target_pos.copy()

        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        # l2_dist = np.linalg.norm(self.data.site_xpos[self.peg_sid] - self.data.site_xpos[self.target_sid])
        l2_dist = np.linalg.norm(
            self.data.site_xpos[self.peg_sid] - self.data.site_xpos[self.true_target_sid])
        goal_achieved = True if (l2_dist < 0.06) else False
        return dict(state=self.get_env_state(), goal_achieved=goal_achieved)

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth += 200
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent*2.0

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if peg in hole for at least 5 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 5:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage

