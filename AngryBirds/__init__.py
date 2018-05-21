from AngryBirds.angrybirds import AngryBirdEnv
from gym.envs.registration import registry, register, make, spec

register(
    id='AngryBirds-v0',
    entry_point='AngryBirds.angrybirds:AngryBirdEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
    nondeterministic=False,
)

register(
    id='AngryBirds-v1',
    entry_point='AngryBirds.angrybirds:AngryBirdEnvNoDisp',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
    nondeterministic=False,
)
