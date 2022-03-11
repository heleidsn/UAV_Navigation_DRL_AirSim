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

import wandb
from wandb.integration.sb3 import WandbCallback

class TrainingThread(QtCore.QThread):
    # signals
    def __init__(self, config):
        super(TrainingThread, self).__init__()
        print("init training thread")

        # config 
        self.cfg = ConfigParser()
        self.cfg.read(config)

        self.objective = self.cfg.get('options', 'objective')

        # make gym environment
        self.env = gym.make('airsim-env-v0')
        self.env.set_config(self.cfg)

        # wandb
        if self.cfg.getboolean('options', 'use_wandb'):
            project_name = self.cfg.get('options', 'env_name') + '_' + self.cfg.get('options', 'dynamic_name') + '_' + self.cfg.get('options', 'navigation_3d')
            wandb.init(
                project=project_name,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=True,  # optional
            )

    def terminate(self):
        print('TrainingThread terminated')

    def run(self):
        print("run training thread")
        
        # init folders
        now = datetime.datetime.now()
        now_string = now.strftime('%Y_%m_%d_%H_%M_' + self.objective)
        file_path = 'logs/' + now_string
        log_path = file_path + '/tb_logs'
        model_path = file_path + '/models'
        config_path = file_path + '/config'
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(config_path, exist_ok=True)

        # save config file
        with open(config_path + '\config.ini', 'w') as configfile:
            self.cfg.write(configfile)
        
        # The noise objects for TD3
        n_actions = self.env.action_space.shape[-1]
        noise_sigma = self.cfg.getfloat('TD3', 'action_noise_sigma') * np.ones(n_actions)
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma)
        
        # set policy
        feature_num_state = self.env.dynamic_model.state_feature_length
        policy_name = self.cfg.get('policy', 'policy_name')
        if policy_name == 'CustomCNN_fc':
            feature_num_cnn = 25
            policy_used = CustomCNN_fc
        elif policy_name == 'CustomCNN_GAP':
            feature_num_cnn = 16
            policy_used = CustomCNN_GAP
        elif policy_name == 'CustomCNN_mobile':
            feature_num_cnn = 32
            policy_used = CustomCNN_mobile
        elif policy_name == 'CustomNoCNN':
            feature_num_cnn = 25
            policy_used = CustomNoCNN
        else:
            print('policy select error')
        policy_kwargs = dict(
            features_extractor_class=policy_used,
            features_extractor_kwargs=dict(features_dim=feature_num_state+feature_num_cnn,
                                           state_feature_dim=feature_num_state),  # 指定最后total feature 数目 应该是CNN+state
            activation_fn=th.nn.ReLU
        )
        policy_kwargs['net_arch']=[64, 32]

        # TODO create eval_callback
        # eval_freq = self.cfg.getint('TD3', 'eval_freq')
        # n_eval_episodes = self.cfg.getint('TD3', 'n_eval_episodes')
        # eval_callback = EvalCallback(self.env, best_model_save_path= file_path + '/eval',
        #                      log_path= file_path + '/eval', eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        #                      deterministic=True, render=False)
        model = TD3('CnnPolicy', self.env, 
                    action_noise=action_noise,
                    learning_rate=self.cfg.getfloat('TD3', 'learning_rate'),
                    gamma=self.cfg.getfloat('TD3', 'gamma'),
                    policy_kwargs=policy_kwargs,
                    learning_starts=self.cfg.getint('TD3', 'learning_starts'),
                    batch_size=self.cfg.getint('TD3', 'batch_size'),
                    train_freq=(self.cfg.getint('TD3', 'train_freq'), 'step'),
                    gradient_steps=self.cfg.getint('TD3', 'gradient_steps'),
                    buffer_size=self.cfg.getint('TD3', 'buffer_size'),
                    tensorboard_log=log_path,
                    seed=0,
                    verbose=1,
                    )
        
        # train
        print('start training model')
        total_timesteps = self.cfg.getint('TD3', 'total_timesteps')
        self.env.model = model

        if self.cfg.getboolean('options', 'use_wandb'):
            model.learn(
                total_timesteps,
                callback=WandbCallback(
                    model_save_freq=2000,
                    gradient_save_freq=100,
                    model_save_path=model_path,
                    verbose=2,
                )
            )
        else:
            model.learn(total_timesteps)
    
        # self.run.finish()
        # model_name = 'model_sb3'
        # model.save(model_path + '/' + model_name)
        
        print('training finished')
        print('model saved to: {}'.format(model_path))

        
def main():
    config_file = 'configs/config_new.ini'
    training_thread = TrainingThread(config_file)
    training_thread.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
    