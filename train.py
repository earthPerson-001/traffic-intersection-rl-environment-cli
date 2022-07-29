from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from custom_gym.envs.custom_env_dir import TrafficIntersectionEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import gym


env = gym.make('TrafficIntersectionEnv-v1')

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo-trafficintersection/")
model.learn(total_timesteps=200000)
model.save("./models/TrafficIntersection-ppo")

# model = PPO.load("./models/TrafficIntersection-ppo", env=env)

obs = env.reset()
for i in range(10):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    print("Observation: {}\n Action: {}\n Reward: {} \n Done: {} ".format(obs, action, reward, done))
    print("\n")