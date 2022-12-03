import airsim
import numpy as np
import math
from gym import spaces


class MultirotorDynamicsSimple():
    '''
    A simple dynamics used for vision based navigation
    It has position (x, y, z, yaw) in local frame and v_xy v_z yaw_rate as states
    State:
        dist_xy, dist_z, relative_yaw, v_xy, vz, yaw_rate 
        Note:
            Using only position error as state, the controller will be a nolinear P controller
            Using velocity as state, the controller is similar to a PD controller, but velocity changing fast,
            it will lead to vibration
    Action: 
        v_xy, v_z, yaw_rate
    '''

    def __init__(self, cfg) -> None:

        # config
        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')
        self.using_velocity_state = cfg.getboolean(
            'options', 'using_velocity_state')
        self.dt = cfg.getfloat('multirotor', 'dt')

        # AirSim Client
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()

        # start and goal position
        self.start_position = [0, 0, 0]
        self.start_random_angle = None
        self.goal_position = [0, 0, 0]
        self.goal_distance = None
        self.goal_random_angle = None
        self.goal_rect = None

        # states
        self.x = 0
        self.y = 0
        self.z = 0
        self.v_xy = 0
        self.v_z = 0
        self.yaw = 0
        self.yaw_rate = 0

        self.acc_xy_max = cfg.getfloat('multirotor', 'acc_xy_max')
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')
        self.v_xy_min = cfg.getfloat('multirotor', 'v_xy_min')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.max_vertical_difference = 5

        if self.navigation_3d:
            if self.using_velocity_state:
                self.state_feature_length = 6
            else:
                self.state_feature_length = 3
            self.action_space = spaces.Box(low=np.array([self.v_xy_min, -self.v_z_max, -self.yaw_rate_max_rad]),
                                           high=np.array([self.v_xy_max, self.v_z_max, self.yaw_rate_max_rad]),
                                           dtype=np.float32)
        else:
            if self.using_velocity_state:
                self.state_feature_length = 4
            else:
                self.state_feature_length = 2
            self.action_space = spaces.Box(low=np.array([self.v_xy_min, -self.yaw_rate_max_rad]),
                                           high=np.array([self.v_xy_max, self.yaw_rate_max_rad]),
                                           dtype=np.float32)

    def reset(self):
        # reset goal
        self.update_goal_pose()
        # reset start
        yaw_noise = self.start_random_angle * np.random.random()
        self.x = self.start_position[0]
        self.y = self.start_position[1]
        self.z = self.start_position[2]
        self.yaw = yaw_noise
        self.v_xy = 0
        self.v_z = 0
        self.yaw_rate = 0

        # reset pose
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.x
        pose.position.y_val = self.y
        pose.position.z_val = -self.z
        pose.orientation = airsim.to_quaternion(0, 0, self.yaw)

        self.client.simSetVehiclePose(pose, False)

    def set_action(self, action):
        # ------------update control command---------------
        self.v_xy = action[0]
        self.yaw_rate = action[-1]
        if self.navigation_3d:
            self.v_z = action[1]
        else:
            self.v_z = 0

        # ------------update position-----------------------
        self.x += self.v_xy * math.cos(self.yaw) * self.dt
        self.y += self.v_xy * math.sin(self.yaw) * self.dt
        self.z += self.v_z * self.dt

        # ------------update yaw----------------------------
        self.yaw += self.yaw_rate * self.dt
        if self.yaw > math.radians(180):
            self.yaw -= math.pi * 2
        elif self.yaw < math.radians(-180):
            self.yaw += math.pi * 2

        # ------------update AirSim-------------------------
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.x
        pose.position.y_val = self.y
        pose.position.z_val = - self.z
        pose.orientation = airsim.to_quaternion(0, 0, self.yaw)
        self.client.simSetVehiclePose(pose, False)

        return 0

    def update_goal_pose(self):
        # if goal is given by rectangular mode
        if self.goal_rect is None:
            distance = self.goal_distance
            noise = np.random.random()
            angle = noise * self.goal_random_angle  # (0~2pi)
            goal_x = distance * math.cos(angle) + self.start_position[0]
            goal_y = distance * math.sin(angle) + self.start_position[1]
        else:
            goal_x, goal_y = self.get_goal_from_rect(
                self.goal_rect, self.goal_random_angle)
            self.goal_distance = math.sqrt(goal_x*goal_x + goal_y*goal_y)
        self.goal_position[0] = goal_x
        self.goal_position[1] = goal_y
        self.goal_position[2] = self.start_position[2]
        # print('New goal pose: ', self.goal_position)

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle

    def set_goal(self, distance=None, random_angle=0, rect=None):
        if distance is not None:
            self.goal_distance = distance
        self.goal_random_angle = random_angle
        if rect is not None:
            self.goal_rect = rect

    def get_goal_from_rect(self, rect_set, random_angle_set):
        rect = rect_set
        random_angle = random_angle_set
        noise = np.random.random()
        angle = random_angle * noise - math.pi   # -pi~pi
        rect = [-128, -128, 128, 128]
        # goal_x = 100*math.sin(angle)
        # goal_y = 100*math.cos(angle)

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

        return goal_x, goal_y

    def _get_state_feature(self):
        '''
        @description: update and get current uav state and state_norm 
        @return: state_norm
                    normalized state to 0-255
                    the stable baselines 3 observation function will normalize image from 0-255 to 0-1
        '''

        distance = self.get_distance_to_goal_2d()
        relative_yaw = self._get_relative_yaw()  # return relative yaw -pi to pi
        # current position z is positive
        relative_pose_z = self.z - self.goal_position[2]
        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255

        distance_norm = distance / self.goal_distance * 255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5) * 255

        # current speed and angular speed
        linear_velocity_xy = self.v_xy
        linear_velocity_norm = (linear_velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) * 255
        linear_velocity_z = self.v_z
        linear_velocity_z_norm = (linear_velocity_z / self.v_z_max / 2 + 0.5) * 255
        angular_velocity_norm = (self.yaw_rate / self.yaw_rate_max_rad / 2 + 0.5) * 255

        # state: distance_h, distance_v, relative yaw, velocity_x, velocity_z, velocity_yaw
        self.state_raw = np.array([distance, relative_pose_z,  math.degrees(
            relative_yaw), linear_velocity_xy, linear_velocity_z,  math.degrees(self.yaw_rate)])
        state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm,
                               linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])
        state_norm = np.clip(state_norm, 0, 255)

        if self.navigation_3d:
            if self.using_velocity_state == False:
                state_norm = state_norm[:3]
        else:
            state_norm = np.array(
                [state_norm[0], state_norm[2], state_norm[3], state_norm[5]])
            if self.using_velocity_state == False:
                state_norm = state_norm[:2]

        self.state_norm = state_norm

        return state_norm

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
        yaw_current = self.get_attitude()[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def get_position(self):
        position = self.client.simGetVehiclePose().position
        return [position.x_val, position.y_val, -position.z_val]

    def get_attitude_cmd(self):
        return [0.0, 0.0, 0.0]

    def get_attitude(self):
        # return current euler angle
        return [0.0, 0.0, self.yaw]

    def get_distance_to_goal_2d(self):
        return math.sqrt(pow(self.get_position()[0] - self.goal_position[0], 2) + pow(self.get_position()[1] - self.goal_position[1], 2))
