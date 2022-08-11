
from gym.envs.registration import register

register(id='TrafficIntersectionEnvDoubleLane-v1',
    entry_point='envs.custom_env_dir:TrafficIntersectionEnvDoubleLane'
)

register(id='TrafficIntersectionEnvSingleLane-v1',
    entry_point='envs.custom_env_dir:TrafficIntersectionEnvSingleLane'
)