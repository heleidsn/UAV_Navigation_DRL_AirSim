import gym
import gym_airsim_multirotor
import datetime
import os

import torch as th
import numpy as np

from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from utils.custom_policy_sb3 import CustomNoCNN, CustomCNN_GAP, CustomCNN_fc, CustomCNN_mobile
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback

# wandb.init(project="MR_NH", entity="heleidsn")

HOME_PATH = os.pwd()
print(HOME_PATH)

#! ---------------step 0: custom your training process-------------------------
method = 'pure_rl'      # 1-pure_rl 2-generate_expert_data 3-bc_rl 4-offline_rl
policy = 'no_cnn'       # 1-cnn_fc 2-cnn_gap 3-no_cnn 4-cnn_mobile
env_name = 'airsim_city'      # 1-trees  2-cylinder
algo = 'td3'            # 1-ppo 2-td3
action_num = '2d'       # 2d or 3d
purpose = 'fixed_wing_test'        # input your training purpose

noise_type = 'NA'
goal_distance = 70
noise_intensity = 0.1
gamma = 0.99
learning_rate = 5e-4
total_steps = 3e5

config = {
    "policy_type": policy,
    "total_timesteps": total_steps,
    "env_name": env_name,
}
# run = wandb.init(
#     project="sb3_MR_NH",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     # monitor_gym=True,  # auto-upload the videos of agents playing the game
#     save_code=True,  # optional
# )

# init folders
now = datetime.datetime.now()
now_string = now.strftime('%Y_%m_%d_%H_%M_')
file_path = HOME_PATH + '/logs/' + now_string + '_' + policy + '_' + purpose
log_path = file_path + '/log'
model_path = file_path + '/models'
config_path = file_path + '/config'
os.makedirs(log_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
os.makedirs(config_path, exist_ok=True)

env = gym.make('airsim-env-v0')

#! --------------step 2: create models------------------------------------------
feature_num_state = env.dynamic_model.state_feature_length  # state feature num
if policy == 'cnn_fc':
    feature_num_cnn = 25
    policy_used = CustomCNN_fc
elif policy == 'cnn_gap':
    feature_num_cnn = 16
    policy_used = CustomCNN_GAP
elif policy == 'cnn_mobile':
    feature_num_cnn = 32
    policy_used = CustomCNN_mobile
elif policy == 'no_cnn':
    feature_num_cnn = 25
    policy_used = CustomNoCNN
else:
    print('policy select error')

policy_kwargs = dict(
    features_extractor_class=policy_used,
    # 指定最后total feature 数目 应该是CNN+state
    features_extractor_kwargs=dict(
        features_dim=feature_num_state+feature_num_cnn),
    activation_fn=th.nn.ReLU
)

if algo == 'ppo':
    # policy_kwargs['net_arch']=[dict(pi=[32, 32], vf=[32, 32])]
    policy_kwargs['net_arch'] = [64, 32]

    model = PPO('CnnPolicy', env,
                n_steps=2048,
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_path + '_' + env_name + '_' + algo,
                seed=0, verbose=2)
elif algo == 'td3':
    # policy_kwargs['net_arch']=dict(pi=[32, 32], qf=[32, 32])
    policy_kwargs['net_arch'] = [64, 32]

    n_actions = env.action_space.shape[-1]
    if noise_type == 'NA':
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_intensity * np.ones(n_actions))
    elif noise_type == 'OU':
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_intensity * np.ones(n_actions), theta=5)
    else:
        print("noise_type error")

    model = TD3('CnnPolicy', env,
                action_noise=action_noise,
                learning_rate=learning_rate,
                gamma=gamma,
                policy_kwargs=policy_kwargs, verbose=1,
                learning_starts=2000,
                batch_size=128,
                train_freq=(200, 'step'),
                gradient_steps=200,
                tensorboard_log=log_path + '_' + env_name + '_' + algo,
                buffer_size=50000, seed=0)
else:
    print('algo input error')

#! ----------------step 4: enjoy training process---------------------------------
env.model = model
tb_log_name = algo + '_' + policy + '_' + purpose
model.learn(total_timesteps=total_steps,
            # callback=TensorboardCallback(),
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     model_save_path=f"models/{run.id}",
            #     verbose=2,
            # ),
            log_interval=1,
            tb_log_name=tb_log_name)

# wandb.watch(model)
#! ----------------step 5: save models and training results-----------------------
model_name = method + '_' + algo + '_' + \
    action_num + '_' + policy + '_' + purpose
os.makedirs(model_path, exist_ok=True)
model.save(model_path + '/' + model_name)
print('model is saved to :{}'.format(model_path + '/' + model_name))
print('log is saved to {}'.format(log_path + env_name + '_' + algo))

# run.finish()
