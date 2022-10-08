from stable_baselines3 import PPO
from stable_baselines3 import A2C
import stable_baselines3.common.env_checker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

import gym

import os
import sys
from pathlib import Path

from datetime import date, datetime

from custom_gym.envs.custom_env_dir import TafficIntersectionEnvDoubleLane

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # traffic-intersection-rl-environment-cli root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

startTime = datetime.now()

# constants
TRAFFIC_INTERSECTION_TYPE = "double"
TIMESTEP=10000

# stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)

modelType = "ppo"
models_path = Path(str(ROOT) + "/models").resolve()
log_path = Path(str(ROOT) + "/logs/{}-trafficintersection-{}-lane-{}/".format(modelType, TRAFFIC_INTERSECTION_TYPE, "CLI")).resolve()

env = gym.make('TrafficIntersectionEnv{}Lane-v1'.format(TRAFFIC_INTERSECTION_TYPE.capitalize()))

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

# model = PPO.load("./models/2022-08-12 22:31:22.631886-TrafficIntersection-DoubleLane-{}-1140000".format(modelType), env=env)

count = 1
while count < 30:
    model.learn(total_timesteps=TIMESTEP, reset_num_timesteps=False, tb_log_name=f"{modelType}-{startTime}")
    save_dir = Path(str(models_path) + "/{}-TrafficIntersection-{}Lane-{}-{}".format(startTime, TRAFFIC_INTERSECTION_TYPE.capitalize(), modelType, count * TIMESTEP))
    model.save(str(save_dir))
    count += 1

for i in range(3):
    obs = env.reset()
    
    net_reward = 0
    total_steps = 0
    done = False

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        net_reward += reward
        total_steps += 1
        print("Observation: {} \n Action: {} \nReward: {} \nDone: {}".format(obs,action,reward,done))
        print("\n")
    
    Mean_Reward = net_reward/total_steps

    print("Mean Reward: {}".format(Mean_Reward))
    print("\n")