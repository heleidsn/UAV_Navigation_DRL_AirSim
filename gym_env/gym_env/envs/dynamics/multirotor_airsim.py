import airsim
import numpy as np
import math
from gym import spaces

class MultirotorDynamicsAirsim():
    '''
    A simple multirotor dynamics used for vision based navigation
    The controller is AirSim Simple Flight (https://microsoft.github.io/AirSim/simple_flight/)
    API: client.moveByVelocityZAsync(v_x, v_y, v_z, yaw_rate) is used to control
    '''
    def __init__(self, cfg) -> None:
        
        # config
        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')
        self.dt = cfg.getfloat('multirotor', 'dt')

        # AirSim Client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # start and goal position
        self.start_position = [0, 0, 0]
        self.start_random_angle = None
        self.goal_position = [0, 0, 0]
        self.goal_distance = None
        self.goal_random_angle = None

        # states
        self.x = 0
        self.y = 0
        self.z = 0
        self.v_xy = 0
        self.v_z = 0
        self.yaw = 0
        self.yaw_rate = 0

        # cmd
        self.v_xy_sp = 0
        self.v_z_sp = 0
        self.yaw_rate_sp = 0
        
        # action space
        self.acc_xy_max = cfg.getfloat('multirotor', 'acc_xy_max')
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')
        self.v_xy_min = cfg.getfloat('multirotor', 'v_xy_min')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.max_vertical_difference = 5
        
        if self.navigation_3d:
            self.state_feature_length = 6
            self.action_space = spaces.Box(low=np.array([self.v_xy_min , -self.v_z_max, -self.yaw_rate_max_rad]), \
                                            high=np.array([self.v_xy_max, self.v_z_max, self.yaw_rate_max_rad]), dtype=np.float32)
        else:
            self.state_feature_length = 4
            self.action_space = spaces.Box(low=np.array([self.v_xy_min , -self.yaw_rate_max_rad]), \
                                                high=np.array([self.v_xy_max, self.yaw_rate_max_rad]), \
                                                dtype=np.float32)


    def reset(self):
        self.client.reset()
        # reset goal
        self.update_goal_pose()

        # reset start
        yaw_noise = self.start_random_angle * np.random.random() 

        # set airsim pose
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = - self.start_position[2]
        pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
        self.client.simSetVehiclePose(pose, True)

        self.client.simPause(False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # take off
        self.client.moveToZAsync(-self.start_position[2], 2).join()

        self.client.simPause(True)

    def set_action(self, action):

        self.v_xy_sp = action[0]
        self.yaw_rate_sp = action[-1]
        if self.navigation_3d:
            self.v_z_sp = action[1]
        else:
            self.v_z_sp = 0

        self.yaw = self.get_attitude()[2]
        self.yaw_sp = self.yaw + self.yaw_rate_sp * self.dt

        if self.yaw_sp > math.radians(180):
            self.yaw_sp -= math.pi * 2
        elif self.yaw_sp < math.radians(-180):
            self.yaw_sp += math.pi * 2
        
        vx_local_sp = self.v_xy_sp * math.cos(self.yaw_sp)
        vy_local_sp = self.v_xy_sp * math.sin(self.yaw_sp)

        self.client.simPause(False)
        if len(action) == 2:
            self.client.moveByVelocityZAsync(vx_local_sp, vy_local_sp, -self.start_position[2], self.dt,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))).join()
            # self.client.moveByVelocityZAsync(vx_local_sp, vy_local_sp, -self.start_position[2], self.dt,
            #                                 drivetrain=airsim.DrivetrainType.ForwardOnly,
            #                                 yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(0))).join()
        elif len(action) == 3:
            self.client.moveByVelocityAsync(vx_local_sp, vy_local_sp, -self.v_z_sp, self.dt,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))).join()
                        
        self.client.simPause(True)

    def update_goal_pose(self):
        distance = self.goal_distance
        noise = np.random.random() * 2 - 1
        angle = noise * self.goal_random_angle
        goal_x = distance * math.cos(angle) + self.start_position[0]
        goal_y = distance * math.sin(angle) + self.start_position[1]
        self.goal_position[0] = goal_x
        self.goal_position[1] = goal_y
        self.goal_position[2] = self.start_position[2]
        # print(self.goal_position)

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle
    
    def set_goal(self, distance, random_angle):
        self.goal_distance = distance
        self.goal_random_angle = random_angle

    def _get_state_feature(self):
        '''
        @description: update and get current uav state and state_norm 
        @param {type} 
        @return: state_norm
                    normalized state range 0-255
        '''
        
        distance = self.get_distance_to_goal_2d()
        relative_yaw = self._get_relative_yaw()  # return relative yaw -pi to pi 
        relative_pose_z = self.get_position()[2] - self.goal_position[2]  # current position z is positive
        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255

        distance_norm = distance / self.goal_distance * 255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5 ) * 255

        # current speed and angular speed
        velocity = self.get_velocity()
        linear_velocity_xy = velocity[0]
        linear_velocity_norm = (linear_velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) * 255
        linear_velocity_z = velocity[1]
        linear_velocity_z_norm = (linear_velocity_z / self.v_z_max / 2 + 0.5) * 255
        angular_velocity_norm = (velocity[2] / self.yaw_rate_max_rad / 2 + 0.5) * 255

        if self.navigation_3d:
            # state: distance_h, distance_v, relative yaw, velocity_x, velocity_z, velocity_yaw
            self.state_raw = np.array([distance, relative_pose_z,  math.degrees(relative_yaw), linear_velocity_xy, linear_velocity_z,  math.degrees(velocity[2])])
            state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm, linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])
            state_norm = np.clip(state_norm, 0, 255)
            self.state_norm = state_norm
        else:
            self.state_raw = np.array([distance, math.degrees(relative_yaw), linear_velocity_xy,  math.degrees(velocity[2])])
            state_norm = np.array([distance_norm, relative_yaw_norm, linear_velocity_norm, angular_velocity_norm])
            state_norm = np.clip(state_norm, 0, 255)
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

    def get_velocity(self):
        states = self.client.getMultirotorState()
        linear_velocity = states.kinematics_estimated.linear_velocity
        angular_velocity = states.kinematics_estimated.angular_velocity

        velocity_xy = math.sqrt(pow(linear_velocity.x_val, 2) + pow(linear_velocity.y_val, 2))
        velocity_z = linear_velocity.z_val
        yaw_rate = angular_velocity.z_val

        return [velocity_xy, -velocity_z, yaw_rate]

    def get_attitude(self):
        self.state_current_attitude = self.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)

    def get_attitude_cmd(self):
        return [0.0, 0.0, self.yaw_sp]

    def get_distance_to_goal_2d(self):
        return math.sqrt(pow(self.get_position()[0] - self.goal_position[0], 2) + pow(self.get_position()[1] - self.goal_position[1], 2))

    