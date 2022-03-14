from os import stat
from random import random
from tracemalloc import start
from webbrowser import Elinks
import gym
from gym import spaces
import airsim
from configparser import ConfigParser
from matplotlib.pyplot import setp
import keyboard

import torch as th
import numpy as np
import math
import cv2

from .dynamics.multirotor_simple import MultirotorDynamicsSimple
from .dynamics.multirotor_airsim import MultirotorDynamicsAirsim
from .dynamics.fixedwing_simple import FixedwingDynamicsSimple

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

class AirsimGymEnv(gym.Env, QtCore.QThread):

    action_signal = pyqtSignal(int, np.ndarray) # action (fixedwing: [roll, pitch, yaw, airspeed], multirotor: [v_xy, v_z, yaw_rate])
    state_signal = pyqtSignal(int, np.ndarray) # state_raw (multirotor: [dist_xy, dist_z, relative_yaw, v_xy, v_z, yaw_rate])
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray) # attitude (pitch, roll, yaw   current and command)
    reward_signal = pyqtSignal(int, float, float) # step_reward, total_reward
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)  # goal_pose start_pose current_pose trajectory

    def __init__(self) -> None:
        """_summary_
        """        
        super().__init__()
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("init airsim-gym-env.")
        self.model = None
        self.data_path = None

    def set_config(self, cfg):
        """get config from .ini file

        Args:
            cfg (_type_): config file
        """
        self.cfg = cfg
        self.env_name = cfg.get('options', 'env_name')
        self.dynamic_name = cfg.get('options', 'dynamic_name')
        self.keyboard_debug = cfg.getboolean('options', 'keyboard_debug')
        self.generate_q_map = cfg.getboolean('options', 'generate_q_map')
        print('Environment: ', self.env_name, "dynamics: ", self.dynamic_name)

        if self.dynamic_name == 'SimpleFixedwing':
            self.dynamic_model = FixedwingDynamicsSimple(cfg)
        elif self.dynamic_name == 'SimpleMultirotor':
            self.dynamic_model = MultirotorDynamicsSimple(cfg)
        elif self.dynamic_name == 'Multirotor':
            self.dynamic_model = MultirotorDynamicsAirsim(cfg)
        else:
            raise Exception("Invalid dynamic_name!", self.dynamic_name)

        # set start and goal position according to different environment
        if self.env_name == 'NH':
            start_position = [110, 180, 5]
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model.set_goal(distance = 90, random_angle=0)
            self.work_space_x = [start_position[0], start_position[0] + goal_distance + 10]
            self.work_space_y = [start_position[1] - 30, start_position[1] + 30]
            self.work_space_z = [0.5, 10]
        elif self.env_name == 'City':
            start_position = [40, -30, 40]
            goal_position = [280, -200, 40]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 350]
            self.work_space_y = [-300, 100]
            self.work_space_z = [0, 100]
        elif self.env_name == 'SimpleAvoid':
            start_position = [0, 0, 5]
            goal_distance = 50
            self.dynamic_model.set_start([0, 0, 5], random_angle=math.pi*2)
            self.dynamic_model.set_goal(distance = goal_distance, random_angle=math.pi*2)
            self.work_space_x = [start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
            self.work_space_y = [start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
            self.work_space_z = [0.5, 50]
        else:
            raise Exception("Invalid env_name!", self.env_name)
            
        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length
        
        # robot state
        # [x,y,z] [vx,vy,vz] [Phi,Theta,Psi] [p,q,r](body frame)
        self.robot_state = {
            'position': np.zeros(3),
            'linear_velocity': np.zeros(3),
            'attitude': np.zeros(3),
            'angular_velocity': np.zeros(3)
        }

        # training state
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0

        # other settings
        self.crash_distance = cfg.getint('environment', 'crash_distance')
        self.accept_radius = cfg.getint('environment', 'accept_radius')
        self.max_episode_steps = cfg.getint('environment', 'max_episode_steps')
        self.max_depth_meters = cfg.getint('environment', 'max_depth_meters')

        self.screen_height = cfg.getint('environment', 'screen_height')
        self.screen_width = cfg.getint('environment', 'screen_width')

        self.trajectory_list = []

        self.observation_space = spaces.Box(low=0, high=255, \
                                            shape=(self.screen_height, self.screen_width, 2),\
                                            dtype=np.uint8)

        self.action_space = self.dynamic_model.action_space


    def reset(self):
        
        # reset state  
        self.dynamic_model.reset()

        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.dynamic_model.goal_distance = self.dynamic_model.get_distance_to_goal_2d()
        self.previous_distance_from_des_point = self.dynamic_model.goal_distance

        self.trajectory_list = []

        obs = self.get_obs()

        return obs

    def step(self, action):
        
        # set action
        self.dynamic_model.set_action(action)
        
        position_ue4 = self.dynamic_model.get_position()
        self.trajectory_list.append(position_ue4)

        # get new obs
        obs = self.get_obs()
        done = self.is_done()
        info = {
            'is_success': self.is_in_desired_pose(),
            'is_crash': self.is_crashed(),
            'is_not_in_workspace': self.is_not_inside_workspace(),
            'step_num': self.step_num
        }
        if done:
            print(info)
        
        if self.dynamic_name == 'SimpleFixedwing':
            reward = self.compute_reward_fixedwing(done, action)
        else:
            reward = self.compute_reward(done, action)
        self.cumulated_episode_reward += reward

        self.print_train_info(action, reward, info)

        if self.cfg.get('options', 'dynamic_name') == 'SimpleFixedwing':
            self.set_pyqt_signal_fixedwing(action, reward, done)
        else:
            self.set_pyqt_signal_multirotor(action, reward)

        if self.keyboard_debug:
            action_copy = np.copy(action)
            action_copy[-1] = math.degrees(action_copy[-1])
            state_copy = np.copy(self.dynamic_model.state_raw)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('=============================================================================')
            print('episode', self.episode_num, 'step', self.step_num, 'total step', self.total_step)
            print('action', action_copy)
            print('state', state_copy)
            print('state_norm', self.dynamic_model.state_norm)
            print('reward {:.3f} {:.3f}'.format(reward, self.cumulated_episode_reward))
            print('done', done)
            keyboard.wait('a')

        if self.generate_q_map and (self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC'):
            if self.model is not None:
                with th.no_grad():
                    # get q-value for td3
                    obs_copy = obs.copy()
                    obs_copy = obs_copy.swapaxes(0, 1)
                    obs_copy = obs_copy.swapaxes(0, 2)
                    q_value_current = self.model.critic(th.from_numpy(obs_copy[tuple([None])]).float().cuda(), th.from_numpy(action[None]).float().cuda())
                    q_1 = q_value_current[0].cpu().numpy()[0]
                    q_2 = q_value_current[1].cpu().numpy()[0]

                    q_value = min(q_1, q_2)[0]

                    self.visual_log_q_value(q_value, action, reward)

        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info

#! -------------------------get obs------------------------------------------
    def get_obs(self):
        '''
        @description: get depth image and target features for navigation
        @param {type}
        @return:
        '''
        # 1. get current depth image and transfer to 0-255  0-20m 255-0m
        image = self.get_depth_image()
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        image_scaled = image_resize * 100
        self.min_distance_to_obstacles = image_scaled.min()
        image_scaled = -np.clip(image_scaled, 0, self.max_depth_meters) / self.max_depth_meters * 255 + 255  
        image_uint8 = image_scaled.astype(np.uint8)

        assert image_uint8.shape[0] == self.screen_height and image_uint8.shape[1] == self.screen_width, 'image size not match'
        
        # 2. get current state (relative_pose, velocity)
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()

        assert (self.state_feature_length == state_feature.shape[0]), 'state_length {0} is not equal to state_feature_length {1}' \
                                                                    .format(self.state_feature_length, state_feature.shape[0])
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        # 3. generate image with state
        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)
        
        return image_with_state

    def get_depth_image(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ])

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages(
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True))

        depth_img = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width, responses[0].height)

        return depth_img

#! ---------------------calculate rewards-------------------------------------
    def compute_reward(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -10
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now)
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                v_z_cost = 0 * abs(action[1]) / self.dynamic_model.v_z_max
                action_cost += v_z_cost
            
            action_cost += yaw_speed_cost

            reward = reward_distance - reward_obs - action_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_fixedwing(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -10
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now)
            self.previous_distance_from_des_point = distance_now

            state_cost = 0
            action_cost = 0

            state_cost += 0.2 * abs(self.dynamic_model.roll) / self.dynamic_model.roll_max

            action_cost += 0.2 * abs(action[0]) /self.dynamic_model.roll_rate_max

            reward = reward_distance - state_cost - action_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

#! ------------------ is done-----------------------------------------------
    def is_done(self):
        episode_done = False

        is_not_inside_workspace_now = self.is_not_inside_workspace()
        has_reached_des_pose    = self.is_in_desired_pose()
        too_close_to_obstable   = self.is_crashed()

        # We see if we are outside the Learning Space
        episode_done = is_not_inside_workspace_now or\
                        has_reached_des_pose or\
                        too_close_to_obstable or\
                        self.step_num >= self.max_episode_steps
    
        return episode_done

    def is_not_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        current_position = self.dynamic_model.get_position()

        if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
            current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):
        in_desired_pose = False
        if self.get_distance_to_goal_3d() < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.crash_distance:
            is_crashed = True

        return is_crashed

#! ----------- useful functions-------------------------------------------
    def get_distance_to_goal_3d(self):
        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))
#! -----------used for plot or show states------------------------------------------------------------------

    def print_train_info(self, action, reward, info):
        if self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC':
            feature_all = self.model.actor.features_extractor.feature_all
        elif self.cfg.get('options', 'algo') == 'PPO':
            feature_all = self.model.policy.features_extractor.feature_all
        
        self.client.simPrintLogMessage('feature_all: ', str(feature_all))
        
        msg_train_info = "EP: {} Step: {} Total_step: {}".format(self.episode_num, self.step_num, self.total_step)

        self.client.simPrintLogMessage('Train: ', msg_train_info)
        self.client.simPrintLogMessage('Action: ', str(action))
        self.client.simPrintLogMessage('reward: ', "{:4.4f} total: {:4.4f}".format(reward, self.cumulated_episode_reward))
        self.client.simPrintLogMessage('Info: ', str(info))
        self.client.simPrintLogMessage('Feature_norm: ', str(self.dynamic_model.state_norm))
        self.client.simPrintLogMessage('Feature_raw: ', str(self.dynamic_model.state_raw))

    def set_pyqt_signal_fixedwing(self, action, reward, done):
        """
        emit signals for pyqt plot
        """
        action_plot = np.array([math.degrees(action[0]), 0, 0, 0])
        euler_angle = self.dynamic_model.get_attitude() # pitch, roll yaw
        state_plot = np.array([math.degrees(euler_angle[0]), math.degrees(euler_angle[1]), math.degrees(euler_angle[2]), self.dynamic_model.v_xy])

        self.action_signal.emit(int(self.total_step), action_plot, state_plot)
        # self.robot_state_signal.emit(int(self.total_step), self.robot_state)
        self.state_signal.emit(int(self.total_step), self.dynamic_model.state_raw)
        self.state_norm_signal.emit(int(self.total_step), self.dynamic_model.state_norm)
        self.reward_signal.emit(int(self.total_step), reward, self.cumulated_episode_reward)

        # emit signals for shap explain
        self.training_info_signal.emit(self.episode_num, self.step_num, self.total_step, reward, self.cumulated_episode_reward, done) # episode, timestep, total timestep, reward, accumulate reward, done
        self.state_raw_norm_signal.emit(self.dynamic_model.state_raw, self.dynamic_model.state_norm)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position), np.asarray(self.dynamic_model.start_position), np.asarray(self.dynamic_model.get_position()), np.asarray(self.trajectory_list))

    def set_pyqt_signal_multirotor(self, action, reward):
        step = int(self.total_step)

        # transfer 2D state and action to 3D
        state = self.dynamic_model.state_raw
        if self.dynamic_model.navigation_3d:
            action_output = action
            state_output = state
        else:
            action_output = np.array([action[0], 0, action[1]])
            state_output = np.array([state[0], 0, state[1], state[2], 0, state[3]])
        self.action_signal.emit(step, action_output)
        self.state_signal.emit(step, state_output)

        # other values
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude()), np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position), np.asarray(self.dynamic_model.start_position), np.asarray(self.dynamic_model.get_position()), np.asarray(self.trajectory_list))

    def visual_log_q_value(self, q_value, action, reward):
        '''
        Create grid map (map_size = work_space)
        Log Q value and the best action in grid map
        At any grid position, record:
        1. Q value
        2. action 0
        3. action 1
        4. steps
        5. reward
        Save image every 10k steps
        Used only for 2D explanation
        '''

        # create init array if not exist 
        map_size_x = self.work_space_x[1] - self.work_space_x[0]
        map_size_y = self.work_space_y[1] - self.work_space_y[0]
        if not hasattr(self, 'q_value_map'):
            self.q_value_map = np.full((9, map_size_x+1, map_size_y+1), np.nan)
        
        # record info
        position = self.dynamic_model.get_position()
        pose_x = position[0]
        pose_y = position[1]

        index_x = int(np.round(pose_x) + self.work_space_x[1])
        index_y = int(np.round(pose_y) + self.work_space_y[1])

        # check if index valid
        if index_x in range(0, map_size_x) and index_y in range(0, map_size_y):
            self.q_value_map[0, index_x, index_y] = q_value
            self.q_value_map[1, index_x, index_y] = action[0]
            self.q_value_map[2, index_x, index_y] = action[-1]
            self.q_value_map[3, index_x, index_y] = self.total_step
            self.q_value_map[4, index_x, index_y] = reward
            self.q_value_map[5, index_x, index_y] = q_value
            self.q_value_map[6, index_x, index_y] = action[0]
            self.q_value_map[7, index_x, index_y] = action[-1]
            self.q_value_map[8, index_x, index_y] = reward
        else:
            print('Error: X:{} and Y:{} is outside of range 0~mapsize (visual_log_q_value)')

        # save array every record_step steps 
        record_step = self.cfg.getint('options', 'q_map_save_steps')
        if (self.total_step+1) % record_step == 0:
            if self.data_path is not None:
                np.save(self.data_path + '/q_value_map_{}'.format(self.total_step+1), self.q_value_map)
                # refresh 5 6 7 8 to record period data
                self.q_value_map[5, :, :] = np.nan
                self.q_value_map[6, :, :] = np.nan
                self.q_value_map[7, :, :] = np.nan
                self.q_value_map[8, :, :] = np.nan