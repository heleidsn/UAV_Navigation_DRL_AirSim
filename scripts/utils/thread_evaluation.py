import datetime
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
sys.path.append(r"C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\scripts")

import gym
import gym_env
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from torch.utils.tensorboard import SummaryWriter
# import wandb

from utils.custom_policy_sb3 import CustomNoCNN, CustomCNN_GAP, CustomCNN_fc, CustomCNN_mobile
import torch as th
from configparser import ConfigParser

from PyQt5 import QtWidgets, QtCore

class EvaluateThread(QtCore.QThread):
    # signals
    def __init__(self, config, model_file, total_eval_episodes):
        super(EvaluateThread, self).__init__()
        print("init training thread")

        # config 
        self.cfg = ConfigParser()
        self.cfg.read(config)

        self.env = gym.make('airsim-env-v0')
        self.env.set_config(self.cfg)

        self.model_file = model_file
        self.total_eval_episodes = total_eval_episodes

    def terminate(self):
        print('Evaluation terminated')

    def run(self):
        print('start evaluation')

        model = TD3.load(self.model_file, env=self.env)
        self.env.model = model

        obs = self.env.reset()
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        episode_successes = []

        while episode_num < self.total_eval_episodes:
            unscaled_action, _ = model.predict(obs)
            time_step += 1
            new_obs, reward, done, info, = self.env.step(unscaled_action)

            obs = new_obs
            reward_sum[-1] += reward

            if done:
                episode_num += 1
                maybe_is_success = info.get('is_success')
                print('episode: ', episode_num, ' reward:', reward_sum[-1], 'success:', maybe_is_success)
                episode_successes.append(float(maybe_is_success))
                reward_sum = np.append(reward_sum, .0)
                obs = self.env.reset()

        print('Average episode reward: ', reward_sum[:self.total_eval_episodes].mean(), 'Success rate:', np.mean(episode_successes))
        
def main():
    config_file = r'C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\logs\2022_03_10_16_06_test\config\config.ini'
    model_file = r'C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\logs\2022_03_10_16_06_test\models\2022_03_10_16_06_test_50000.zip'
    total_eval_episodes = 10
    evaluate_thread = EvaluateThread(config_file, model_file, total_eval_episodes)
    evaluate_thread.run()
    # evaluate_thread.terminate()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
    