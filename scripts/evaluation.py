import gym
import gym_airsim_multirotor
import datetime
import os

import torch as th
import numpy as np

from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from utils.custom_policy_sb3 import CustomCNN, CustomCNN_GAP, CustomCNN_fc, CustomCNN_mobile
from stable_baselines3.common.callbacks import BaseCallback

env = gym.make('airsim-simple-dynamics-v0')

model_path = r'C:\Users\helei\OneDrive - mail.nwpu.edu.cn\Github\UAV_Navigation_DRL_AirSim\logs\2022_01_22_21_53__no_cnn_acc_NH_env_3d\models\pure_rl_td3_2d_no_cnn_acc_NH_env_3d.zip'
model = TD3.load(model_path)

env.model = model

obs = env.reset()

while True:
    action = model.predict(obs)
    obs, rewards, done, info = env.step(action[0])
    if done:
        obs = env.reset()