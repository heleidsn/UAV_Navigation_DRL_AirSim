from cmath import inf
import datetime
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
sys.path.append(r"C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\scripts")

import gym
import gym_env
import numpy as np
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from torch.utils.tensorboard import SummaryWriter
# import wandb

from utils.custom_policy_sb3 import CNN_FC, CNN_GAP, CNN_GAP_BN, No_CNN, CNN_MobileNet
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
        algo = self.cfg.get('options', 'algo')
        if algo == 'TD3':
            model = TD3.load(self.model_file, env=self.env)
        elif algo == 'SAC':
            model = SAC.load(self.model_file, env=self.env)
        elif algo == 'PPO':
            model = PPO.load(self.model_file, env=self.env)
        else:
            raise Exception('algo set error {}'.format(algo))
        self.env.model = model

        obs = self.env.reset()
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        episode_successes = []
        traj_list_all = []
        action_list_all = []
        state_list_all = []
        
        traj_list = []
        action_list = []
        state_raw_list = []
        step_num_list = []

        while episode_num < self.total_eval_episodes:
            unscaled_action, _ = model.predict(obs)
            time_step += 1
            new_obs, reward, done, info, = self.env.step(unscaled_action)
            pose = self.env.dynamic_model.get_position()
            traj_list.append(pose)
            action_list.append(unscaled_action)
            state_raw_list.append(self.env.dynamic_model.state_raw)

            obs = new_obs
            reward_sum[-1] += reward

            if done:
                episode_num += 1
                maybe_is_success = info.get('is_success')
                print('episode: ', episode_num, ' reward:', reward_sum[-1], 'success:', maybe_is_success)
                episode_successes.append(float(maybe_is_success))
                reward_sum = np.append(reward_sum, .0)
                obs = self.env.reset()
                if info.get('is_success'):
                    traj_list.append(1)
                    action_list.append(1)
                    step_num_list.append(info.get('step_num'))
                elif info.get('is_crash'):
                    traj_list.append(2)
                    action_list.append(2)
                else:
                    traj_list.append(3)
                    action_list.append(3)
                # traj_list.append(info)
                traj_list_all.append(traj_list)
                action_list_all.append(action_list)
                state_list_all.append(state_raw_list)
                traj_list = []
                action_list = []
                state_raw_list = []

        np.save('traj_eval_{}'.format(self.total_eval_episodes), traj_list_all)
        np.save('action_eval_{}'.format(self.total_eval_episodes), action_list_all)
        np.save('state_eval_{}'.format(self.total_eval_episodes), state_list_all)
        print('Average episode reward: ', reward_sum[:self.total_eval_episodes].mean(), 'Success rate:', np.mean(episode_successes), 'average step num: ', np.mean(step_num_list))
        
def main():
    config_file = r'C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\logs\NH_center_SimpleMultirotor_3D\2022_03_23_14_58_No_CNN_PPO\config\config.ini'
    model_file = r'C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\logs\NH_center_SimpleMultirotor_3D\2022_03_23_14_58_No_CNN_PPO\models\model_240000.zip'
    total_eval_episodes = 50
    evaluate_thread = EvaluateThread(config_file, model_file, total_eval_episodes)
    evaluate_thread.run()
    # evaluate_thread.terminate()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
    