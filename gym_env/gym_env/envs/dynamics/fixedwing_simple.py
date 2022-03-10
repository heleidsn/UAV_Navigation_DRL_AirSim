from threading import main_thread
from unittest.mock import patch
from webbrowser import Elinks
import airsim
import numpy as np
import math
from gym import spaces

class FixedwingDynamicsSimple():
    '''
    A simple dynamics used for vision based fixed wing navigation
    It has position (x, y, z, yaw) in local frame and v_xy v_z yaw_rate as states
    '''

    def __init__(self) -> None:
        # config
        self.dt = 0.1

        # AirSim Client
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()

        # states
        self.x = 0
        self.y = 0
        self.z = 0
        self.v_xy = 10
        self.v_z = 0

        # angular in radians
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.pitch_rate = 0
        self.roll_rate = 0
        self.yaw_rate = 0

        self.roll_max = math.radians(40)

        # control command
        self.roll_rate_max = math.radians(40)

        # action space
        self.action_space = spaces.Box(low=np.array([-self.roll_rate_max]), \
                                       high=np.array([self.roll_rate_max]), dtype=np.float32)

        
        self.state_feature_length = 3
        self.goal_distance = 0

    def reset(self):
        self.x = self.start_position[0]
        self.y = self.start_position[1]
        self.z = self.start_position[2]
        self.v_xy = 10
        self.v_z = 0
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.pitch_rate = 0
        self.roll_rate = 0
        self.yaw_rate = 0

        # reset pose
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.x
        pose.position.y_val = self.y
        pose.position.z_val = -self.z
        pose.orientation = airsim.to_quaternion(0, 0, self.yaw)

        self.client.simSetVehiclePose(pose, False)

    def set_action(self, action, robot_state):
        """
        更新动力学
        前向速度保持不变
        通过滚转来实现偏航运动
        """

        self.roll_rate = action[0]

        self.roll = self.roll + self.roll_rate * self.dt

        if self.roll > self.roll_rate_max:
            self.roll = self.roll_rate_max
        elif self.roll < -self.roll_rate_max:
            self.roll = -self.roll_rate_max
        
        # 滚转提供法向加速度，根据法向加速度a_n和飞行速度v_xy可以得到偏航角速度yaw_rate
        a_n = math.tan(self.roll) * 9.8
        self.yaw_rate = a_n / self.v_xy

        self.yaw = self.yaw + self.yaw_rate * self.dt
        if self.yaw > math.radians(180):
            self.yaw -= math.pi * 2
        elif self.yaw < math.radians(-180):
            self.yaw += math.pi * 2
        
        self.x += self.v_xy * math.cos(self.yaw) * self.dt
        self.y += self.v_xy * math.sin(self.yaw) * self.dt

        # set airsim pose
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.x
        pose.position.y_val = self.y
        pose.position.z_val = - self.z
        pose.orientation = airsim.to_quaternion(0, self.roll, self.yaw)
        self.client.simSetVehiclePose(pose, False)

        robot_state[0] = self.x
        robot_state[1] = self.y
        robot_state[2] = self.z
        
        robot_state[3] = self.v_xy * math.cos(self.yaw)
        robot_state[4] = self.v_xy * math.sin(self.yaw)
        robot_state[5] = 0

        robot_state[6] = self.pitch
        robot_state[7] = self.roll
        robot_state[8] = self.yaw

        robot_state[9] = self.pitch_rate
        robot_state[10] = self.roll_rate
        robot_state[11] = self.yaw_rate
        
        return 0

    def _get_state_feature(self):
        '''
        @description: state_feature: 选择目标距离，相对角度，滚转角作为state feature
        @param {type} 
        @return: state_norm
                    normalized state range 0-255
        '''
        distance = self._get_2d_distance_to_goal()
        relative_yaw = self._get_relative_yaw()  # return relative yaw -pi to pi 

        distance_norm = distance / self.goal_distance * 255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5 ) * 255
        roll_norm = (self.roll / self.roll_max / 2 + 0.5) * 255

        self.state_raw = np.array([distance, math.degrees(relative_yaw), self.roll])
        self.state_norm = np.array([distance_norm, relative_yaw_norm, roll_norm])
    
        return self.state_norm
    
    def get_position(self):
        return [self.x, self.y, self.z]

    def get_attitude(self):
        return [self.pitch, self.roll, self.yaw]

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle
    
    def _set_goal_pose_single(self, goal):
        self.goal_position = [goal[0], goal[1], goal[2]]

    def _get_2d_distance_to_goal(self):
        return math.sqrt(pow(self.get_position()[0] - self.goal_position[0], 2) + pow(self.get_position()[1] - self.goal_position[1], 2))

    def _get_relative_yaw(self):
        '''
        @description: get relative yaw from current pose to goal in radian
        @param {type} 
        @return: 
        '''
        current_position = self.get_position()
        # get relative angle
        relative_pose_x = self.goal_position[0] - current_position[0]
        relative_pose_y = self.goal_position[1] - current_position[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = self.yaw

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

