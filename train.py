from multiprocessing import popen_fork
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import stable_baselines3.common.env_checker
from stable_baselines3.common.env_util import make_vec_env
from custom_gym.envs.custom_env_dir import TrafficIntersectionEnvDoubleLane
from stable_baselines3.common.vec_env import DummyVecEnv
import gym


env = gym.make('TrafficIntersectionEnvDoubleLane-v1')

# stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="../logs/ppo-trafficintersection/")

model.learn(total_timesteps=200000)

model.save("../models/TrafficIntersection-DoubleLane-ppo")


# model = PPO.load("../models/TrafficIntersection-singleLane-ppo", env=env)


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
