from datetime import date, datetime
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import stable_baselines3.common.env_checker
from stable_baselines3.common.env_util import make_vec_env
from custom_gym.envs.custom_env_dir import TrafficIntersectionEnvDoubleLane
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

startTime = datetime.now()

env = gym.make('TrafficIntersectionEnvSingleLane-v1')

# stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)

modelType = "ppo"
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/{}-trafficintersection/".format(modelType))

# model = PPO.load("./models/2022-08-12 22:31:22.631886-TrafficIntersection-DoubleLane-{}-1140000".format(modelType), env=env)


TIMESTEP=10000
count = 1
while count < 30:
    model.learn(total_timesteps=TIMESTEP, reset_num_timesteps=False, tb_log_name=f"{modelType}-{startTime}")
    model.save("./models/{}-TrafficIntersection-SingleLane-{}-{}".format(startTime, modelType, count * TIMESTEP))
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