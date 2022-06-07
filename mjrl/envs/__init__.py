from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    id='mjrl_point_mass-v0',
    entry_point='mjrl.envs:PointMassEnv',
    max_episode_steps=25,
)

register(
    id='mjrl_swimmer-v0',
    entry_point='mjrl.envs:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id='mjrl_reacher_7dof-v0',
    entry_point='mjrl.envs:Reacher7DOFEnv',
    max_episode_steps=50,
)

register(
    id='mjrl_peg_insertion-v0',
    entry_point='mjrl.envs:PegEnv',
    max_episode_steps=50,
)


register(
    id='reacher-v0',
    entry_point='mjrl.envs.sawyer:ReacherEnv',
    max_episode_steps=50,
)

register(
    id='sawyer-pusher-v0',
    entry_point='mjrl.envs.sawyer:PusherEnv',
    max_episode_steps=50,
)

register(
    id='sawyer-pusher-v1',
    entry_point='mjrl.envs.sawyer:PusherEnv',
    max_episode_steps=50,
    kwargs={'randomize_goal': True}
)

register(
    id='sawyer-peg-insertion-v0',
    entry_point='mjrl.envs.sawyer:PegEnv',
    max_episode_steps=50,
)

from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from mjrl.envs.point_mass import PointMassEnv
from mjrl.envs.swimmer import SwimmerEnv
from mjrl.envs.reacher_sawyer import Reacher7DOFEnv
from mjrl.envs.peg_insertion_sawyer import PegEnv
from mjrl.envs.sawyer.pusher import PusherEnv
