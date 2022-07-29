
from gym.envs.registration import register

register(id='TrafficIntersectionEnv-v1',
    entry_point='envs.custom_env_dir:TrafficIntersectionEnv'
)