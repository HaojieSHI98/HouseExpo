"""Set up gym interface for locomotion environments."""
import gym
from gym.envs.registration import registry, make, spec

def register(env_id, *args, **kvargs):
  if env_id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(env_id, *args, **kvargs)

register(
    env_id='RobotExploration-v0',
    entry_point='pseudoslam.envs.robot_exploration_v0:RobotExplorationT0',
    max_episode_steps=200,
)