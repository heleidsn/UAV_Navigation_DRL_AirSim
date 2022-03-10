import gym
import gym_airsim_multirotor
import datetime
import os

import torch as th
import numpy as np

from stable_baselines3 import TD3, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

env = gym.make('airsim-env-v0')

model_path = r'C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\logs\2022_03_03_23_26__no_cnn_fixed_wing_test\models\pure_rl_td3_2d_no_cnn_fixed_wing_test.zip'
model = TD3.load(model_path)

env.model = model

obs = env.reset()

while True:
    action = model.predict(obs)
    obs, rewards, done, info = env.step(action[0])
    if done:
        obs = env.reset()