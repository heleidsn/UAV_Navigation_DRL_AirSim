import airsim
import numpy as np
import math
from gym import spaces


class FixedwingDynamicsSimple():
    '''
    A simple dynamics used for vision based fixed wing navigation
    state:
        x, y, z, yaw
    action:
        roll
    '''

    def __init__(self, cfg) -> None:
        # config
        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')
        self.dt = cfg.getfloat('fixedwing', 'dt')

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

        # control command
        self.roll_max_deg = cfg.getfloat('fixedwing', 'roll_max_deg')
        self.roll_max = math.radians(self.roll_max_deg)

        # 动力学约束
        self.roll_rate_max_deg = cfg.getfloat('fixedwing', 'roll_rate_max_deg')
        self.roll_rate_max = math.radians(self.roll_rate_max_deg)

        # flapping config
        self.pitch_flap_hz = cfg.getfloat('fixedwing', 'pitch_flap_hz')
        self.pitch_flap_deg = cfg.getfloat('fixedwing', 'pitch_flap_deg')

        # action space
        self.action_space = spaces.Box(low=np.array([-self.roll_max]),
                                       high=np.array([self.roll_max]),
                                       dtype=np.float32)

        self.state_feature_length = cfg.getint('options', 'state_feature_num')
        self.goal_distance = 0

        self.env_name = cfg.get('options', 'env_name')

    def reset(self):

        if self.env_name == 'City_400':
            self.update_start_goal_rect(size=200)
        if self.env_name == 'Tree_200':
            self.update_start_goal_rect(size=80)
        if self.env_name == 'Forest':
            self.update_start_goal_rect(size=90)

        self.x = self.start_position[0]
        self.y = self.start_position[1]
        self.z = self.start_position[2]
        self.v_xy = 10
        self.v_z = 0
        self.pitch = 0
        self.roll = 0
        # self.yaw = 0
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

    def update_start_goal_rect(self, size):

        rect = [-size, -size, size, size]
        noise = np.random.random() * 2 - 1  # (-1,1)
        angle = noise * math.pi  # -pi~pi

        if abs(angle) == math.pi/2:
            goal_x = 0
            if angle > 0:
                goal_y = rect[3]
            else:
                goal_y = rect[1]
        if abs(angle) <= math.pi/4:
            goal_x = rect[2]
            goal_y = goal_x*math.tan(angle)
        elif abs(angle) > math.pi/4 and abs(angle) <= math.pi/4*3:
            if angle > 0:
                goal_y = rect[3]
                goal_x = goal_y/math.tan(angle)
            else:
                goal_y = rect[1]
                goal_x = goal_y/math.tan(angle)
        else:
            goal_x = rect[0]
            goal_y = goal_x * math.tan(angle)

        self.start_position[0] = -goal_x
        self.start_position[1] = -goal_y
        self.goal_position[0] = goal_x
        self.goal_position[1] = goal_y
        self.goal_position[2] = self.start_position[2]

        self.goal_distance = math.sqrt(goal_x*goal_x + goal_y*goal_y) * 2

        yaw_noise = (np.random.random() * 2 - 1) * math.radians(40)
        self.yaw = angle + yaw_noise

        if self.yaw > math.pi:
            self.yaw = self.yaw - math.pi * 2
        elif self.yaw < -math.pi:
            self.yaw = self.yaw + math.pi * 2

    def set_action(self, action, step):
        """
        更新动力学
        前向速度保持不变
        通过滚转来实现偏航运动
        """
        self.roll_cmd = action[0]

        # roll rate limitation
        if (self.roll_cmd - self.roll) > self.roll_rate_max * self.dt:
            self.roll += self.roll_rate_max*self.dt
        elif (self.roll_cmd - self.roll) < -self.roll_rate_max * self.dt:
            self.roll -= self.roll_rate_max*self.dt
        else:
            self.roll = self.roll_cmd

        self.roll = self.roll + self.roll_rate * self.dt

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

        # get pitch angle according to pitch_flap config
        time = step * self.dt
        self.pitch = self.pitch_flap_deg * math.sin(2 * math.pi * time *
                                                    self.pitch_flap_hz)
        self.pitch = math.radians(self.pitch)

        # set airsim pose
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.x
        pose.position.y_val = self.y
        pose.position.z_val = - self.z
        pose.orientation = airsim.to_quaternion(
            0, 0, self.yaw)
        self.client.simSetVehiclePose(pose, False)

        return 0

    def _get_state_feature(self):
        '''
        state:
            distance, yaw_error, roll
        normalized
            0-255
        '''
        distance = self.get_distance_to_goal_2d()
        relative_yaw = self._get_relative_yaw()

        # get norm to 0-1
        distance_norm = distance / self.goal_distance
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5)
        roll_norm = (self.roll / self.roll_max / 2 + 0.5)

        self.state_raw = np.array([distance, math.degrees(relative_yaw),
                                   math.degrees(self.roll)])
        self.state_norm = np.array([distance_norm, relative_yaw_norm,
                                    roll_norm])
        # self.state_norm = np.array([distance_norm, relative_yaw_norm, 127])

        if self.state_feature_length == 1:
            self.state_norm = np.array([relative_yaw_norm])
        elif self.state_feature_length == 2:
            self.state_norm = np.array([relative_yaw_norm, roll_norm])
        elif self.state_feature_length == 3:
            self.state_norm = np.array([distance_norm, relative_yaw_norm,
                                        roll_norm])

        self.state_norm = np.clip(self.state_norm, 0, 1) * 255

        return self.state_norm

    def get_position(self):
        return [self.x, self.y, self.z]

    def get_attitude(self):
        return [self.pitch, self.roll, self.yaw]

    def get_attitude_cmd(self):
        return [0.0, 0.0, 0.0]

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle

    def _set_goal_pose_single(self, goal):
        self.goal_position = [goal[0], goal[1], goal[2]]

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

    def get_distance_to_goal_2d(self):
        return math.sqrt(pow(self.get_position()[0] - self.goal_position[0], 2) + pow(self.get_position()[1] - self.goal_position[1], 2))
