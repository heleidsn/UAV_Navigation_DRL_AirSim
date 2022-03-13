'''
@Author: Lei He
@Date: 2020-06-01 22:52:40
LastEditTime: 2022-03-12 20:26:56
@Description: 
@Github: https://github.com/heleidsn
'''
import sys
import math
from turtle import pen

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PIL import Image
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QWidget

from configparser import ConfigParser


class TrainingUi(QWidget):
    """PyQt5 GUI used to show the data during training

    Args:
        QWidget (_type_): _description_
    """    
    def __init__(self, config):
        super(TrainingUi, self).__init__()
        # config 
        self.cfg = ConfigParser()
        self.cfg.read(config)

        self.init_ui()

    def init_ui(self):
        """ init UI
            Include four parts:
                Action
                State
                Attitude
                Reward and trajectory
        """
        self.setWindowTitle("Training UI")
        # self.showFullScreen()

        pg.setConfigOptions(leftButtonPan=False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major') # best performance

        self.max_len = 100
        self.ptr = -self.max_len

        # action, state_feature, attitude, trajectory
        self.dynamics = self.cfg.get('options', 'dynamic_name')
        if self.dynamics == 'SimpleFixedwing':
            action_plot_gb = self.create_actionPlot_groupBox_fixed_wing()
        else:
            action_plot_gb = self.create_actionPlot_groupBox_multirotor()
        state_plot_gb = self.create_state_plot_groupbox()
        attitude_plot_gb = self.create_attitude_plot_groupbox()
        reward_plot_gb = self.create_reward_plot_groupbox()
        traj_plot_gb = self.create_traj_plot_groupbox()

        right_widget = QWidget()
        vlayout = QVBoxLayout()
        vlayout.addWidget(reward_plot_gb)
        vlayout.addWidget(traj_plot_gb)
        right_widget.setLayout(vlayout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(action_plot_gb)
        main_layout.addWidget(state_plot_gb)
        main_layout.addWidget(attitude_plot_gb)
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)

        self.pen_red = pg.mkPen(color='r', width=2)    # used for cmd
        self.pen_blue = pg.mkPen(color='b', width=1)   # used for real data
        self.pen_green = pg.mkPen(color='g', width=2)

    def update_value_list(self, list, value):
        '''
        @description: update value list for plot
        @param {type} 
        @return: 
        '''
        list[:-1] = list[1:]
        list[-1] = float(value)
        return list

# action plot groupbox
    def create_actionPlot_groupBox_multirotor(self):
        """ groupbox for multirotor action plot
            For each action, cmd and real state should be ploted
            Action: 
                v_xy, v_z, yaw_rate
        """        
        actionPlotGroupBox = QGroupBox('Action (multirotor)')
        self.v_xy_cmd_list = np.linspace(0, 0, self.max_len)
        self.v_xy_real_list = np.linspace(0, 0, self.max_len)

        self.v_z_cmd_list = np.linspace(0, 0, self.max_len)
        self.v_z_real_list = np.linspace(0, 0, self.max_len)

        self.yaw_rate_cmd_list = np.linspace(0, 0, self.max_len)
        self.yaw_rate_list = np.linspace(0, 0, self.max_len)

        layout = QVBoxLayout()

        self.plotWidget_v_xy = pg.PlotWidget(title='v_xy (m/s)')
        self.plotWidget_v_xy.setYRange(max=self.cfg.getfloat('multirotor', 'v_xy_max'), min=self.cfg.getfloat('multirotor', 'v_xy_min'))
        self.plotWidget_v_xy.showGrid(x=True,y=True)
        self.plot_v_xy = self.plotWidget_v_xy.plot()      # get plot object
        self.plot_v_xy_cmd = self.plotWidget_v_xy.plot()

        self.plotWidget_v_z = pg.PlotWidget(title='v_z (m/s)')
        self.plotWidget_v_z.setYRange(max=self.cfg.getfloat('multirotor', 'v_z_max'), min=-self.cfg.getfloat('multirotor', 'v_z_max'))
        self.plotWidget_v_z.showGrid(x=True,y=True)
        self.plot_v_z = self.plotWidget_v_z.plot()
        self.plot_v_z_cmd = self.plotWidget_v_z.plot()

        self.plotWidget_yaw_rate = pg.PlotWidget(title='yaw_rate (deg/s)')
        self.plotWidget_yaw_rate.setYRange(max=self.cfg.getfloat('multirotor', 'yaw_rate_max_deg'), min=-self.cfg.getfloat('multirotor', 'yaw_rate_max_deg'))
        self.plotWidget_yaw_rate.showGrid(x=True,y=True)
        self.plot_yaw_rate = self.plotWidget_yaw_rate.plot()
        self.plot_yaw_rate_cmd = self.plotWidget_yaw_rate.plot()

        layout.addWidget(self.plotWidget_v_xy)
        layout.addWidget(self.plotWidget_v_z)
        layout.addWidget(self.plotWidget_yaw_rate)

        actionPlotGroupBox.setLayout(layout)

        return actionPlotGroupBox

    def create_actionPlot_groupBox_fixed_wing(self):
        """
        state and action groupbox
        state: roll, pitch, yaw, airspeed
        action: roll_cmd, pitch_cmd, yaw_cmd, airspeed_cmd
        """
        actionPlotGroupBox = QGroupBox('Real time action and state')

        self.roll_list = np.linspace(0, 0, self.max_len)
        self.roll_cmd_list = np.linspace(0, 0, self.max_len)

        self.pitch_list = np.linspace(0, 0, self.max_len)
        self.pitch_cmd_list = np.linspace(0, 0, self.max_len)

        self.yaw_list = np.linspace(0, 0, self.max_len)
        self.yaw_cmd_list = np.linspace(0, 0, self.max_len)

        self.airspeed_list = np.linspace(0, 0, self.max_len)
        self.airspeed_cmd_list = np.linspace(0, 0, self.max_len)

        layout = QVBoxLayout()

        self.plotWidget_roll = pg.PlotWidget(title='roll (deg)')
        self.plotWidget_roll.setYRange(max=45, min=-45)
        self.plotWidget_roll.showGrid(x=True,y=True)
        self.plot_roll = self.plotWidget_roll.plot()
        self.plot_roll_cmd = self.plotWidget_roll.plot()
        # self.plot_vel_x_final = self.plotWidget_roll.plot()

        self.plotWidget_pitch = pg.PlotWidget(title='pitch (deg)')
        self.plotWidget_pitch.setYRange(max=25, min=-25)
        self.plotWidget_pitch.showGrid(x=True,y=True)
        self.plot_pitch = self.plotWidget_pitch.plot()
        self.plot_pitch_cmd = self.plotWidget_pitch.plot()
        # self.plot_vel_z_final = self.plotWidget_pitch.plot()

        self.plotWidget_yaw = pg.PlotWidget(title='yaw (deg)')
        # self.plotWidget_yaw.setYRange(max=180, min=-180)
        self.plotWidget_yaw.showGrid(x=True,y=True)
        self.plot_yaw = self.plotWidget_yaw.plot()
        self.plot_yaw_cmd = self.plotWidget_yaw.plot()
        # self.plot_vel_yaw_final = self.plotWidget_yaw.plot()

        self.plotWidget_airspeed = pg.PlotWidget(title='airspeed (m/s)')
        self.plotWidget_airspeed.setYRange(max=20, min=10)
        self.plotWidget_airspeed.showGrid(x=True,y=True)
        self.plot_airspeed = self.plotWidget_airspeed.plot()
        self.plot_airspeed_cmd = self.plotWidget_airspeed.plot()
        # self.plot_vel_yaw_final = self.plotWidget_airspeed.plot()

        layout.addWidget(self.plotWidget_roll)
        layout.addWidget(self.plotWidget_pitch)
        layout.addWidget(self.plotWidget_yaw)
        layout.addWidget(self.plotWidget_airspeed)

        actionPlotGroupBox.setLayout(layout)

        return actionPlotGroupBox

    def action_cb(self, step, action):
        if self.dynamics == 'SimpleFixedwing':
            self.action_cb_multirotor(step, action)
        else:
            self.action_cb_multirotor(step, action)

    def action_cb_multirotor(self, step, action):
        self.update_value_list(self.v_xy_cmd_list, action[0])
        self.update_value_list(self.v_z_cmd_list, action[1])
        self.update_value_list(self.yaw_rate_cmd_list, math.degrees(action[2]))
        
        self.plot_v_xy_cmd.setData(self.v_xy_cmd_list, pen=self.pen_blue)
        self.plot_v_z_cmd.setData(self.v_z_cmd_list, pen=self.pen_blue)
        self.plot_yaw_rate_cmd.setData(self.yaw_rate_cmd_list, pen=self.pen_blue)

    def action_cb_fixed_wing(self, step, action, state):
        """
        call back function used for fixed wing plot
        """

        # plot action
        self.update_value_list(self.roll_cmd_list, action[0])
        self.update_value_list(self.pitch_cmd_list, action[1])
        self.update_value_list(self.yaw_cmd_list, action[2])
        self.update_value_list(self.airspeed_cmd_list, action[3])

        self.plot_roll_cmd.setData(self.roll_cmd_list, pen=self.pen_red)
        self.plot_pitch_cmd.setData(self.pitch_cmd_list, pen=self.pen_red)
        self.plot_yaw_cmd.setData(self.yaw_cmd_list, pen=self.pen_red)
        self.plot_airspeed_cmd.setData(self.airspeed_cmd_list, pen=self.pen_red)
        

        # plot state
        self.update_value_list(self.roll_list, state[0])
        self.update_value_list(self.pitch_list, state[1])
        self.update_value_list(self.yaw_list, state[2])
        self.update_value_list(self.airspeed_list, state[3])

        self.plot_roll.setData(self.roll_list, pen=self.pen_blue)
        self.plot_pitch.setData(self.pitch_list, pen=self.pen_blue)
        self.plot_yaw.setData(self.yaw_list, pen=self.pen_blue)
        self.plot_airspeed.setData(self.airspeed_list, pen=self.pen_blue)

# state feature plot groupbox
    def create_state_plot_groupbox(self):
        state_plot_groupbox = QGroupBox(title='State feature')

        self.distance_list = np.linspace(0, 0, self.max_len)
        self.vertical_dis_list = np.linspace(0, 0, self.max_len)
        self.relative_yaw_list = np.linspace(0, 0, self.max_len)

        layout = QVBoxLayout()

        self.pw1 = pg.PlotWidget(title='distance_xy (m)')
        self.pw1.showGrid(x=True,y=True)
        self.p1 = self.pw1.plot() 

        self.pw2 = pg.PlotWidget(title='distance_z (m)')
        self.pw2.showGrid(x=True,y=True)
        self.p2 = self.pw2.plot() 

        self.pw3 = pg.PlotWidget(title='relative yaw (deg)')
        self.pw3.showGrid(x=True,y=True)
        self.p3 = self.pw3.plot() 

        layout.addWidget(self.pw1)
        layout.addWidget(self.pw2)
        layout.addWidget(self.pw3)
        
        state_plot_groupbox.setLayout(layout)
        return state_plot_groupbox

    def state_cb(self, step, state_raw):
        
        # update state
        self.update_value_list(self.distance_list, state_raw[0])
        self.update_value_list(self.vertical_dis_list, state_raw[1])
        self.update_value_list(self.relative_yaw_list, state_raw[2])

        self.p1.setData(self.distance_list, pen=self.pen_red)
        self.p2.setData(self.vertical_dis_list, pen=self.pen_red)
        self.p3.setData(self.relative_yaw_list, pen=self.pen_red)

        # update action real
        self.update_value_list(self.v_xy_real_list, state_raw[3])
        self.update_value_list(self.v_z_real_list, state_raw[4])
        self.update_value_list(self.yaw_rate_list, state_raw[5])

        self.plot_v_xy.setData(self.v_xy_real_list, pen=self.pen_red)
        self.plot_v_z.setData(self.v_z_real_list, pen=self.pen_red)
        self.plot_yaw_rate.setData(self.yaw_rate_list, pen=self.pen_red)

# attitude plot groupbox
    def create_attitude_plot_groupbox(self):
        plot_gb = QGroupBox(title='Attitude')
        layout = QVBoxLayout()

        self.roll_list = np.linspace(0, 0, self.max_len)
        self.roll_cmd_list = np.linspace(0, 0, self.max_len)

        self.pitch_list = np.linspace(0, 0, self.max_len)
        self.pitch_cmd_list = np.linspace(0, 0, self.max_len)

        self.yaw_list = np.linspace(0, 0, self.max_len)
        self.yaw_cmd_list = np.linspace(0, 0, self.max_len)

        self.pw_roll = pg.PlotWidget(title='roll (deg)')
        self.pw_roll.setYRange(max=45, min=-45)
        self.pw_roll.showGrid(x=True,y=True)
        self.plot_roll = self.pw_roll.plot()
        self.plot_roll_cmd = self.pw_roll.plot()

        self.pw_pitch = pg.PlotWidget(title='pitch (deg)')
        self.pw_pitch.setYRange(max=25, min=-25)
        self.pw_pitch.showGrid(x=True,y=True)
        self.plot_pitch = self.pw_pitch.plot()
        self.plot_pitch_cmd = self.pw_pitch.plot()

        self.pw_yaw = pg.PlotWidget(title='yaw (deg)')
        # self.pw_yaw.setYRange(max=180, min=-180)
        self.pw_yaw.showGrid(x=True,y=True)
        self.plot_yaw = self.pw_yaw.plot()
        self.plot_yaw_cmd = self.pw_yaw.plot()

        layout.addWidget(self.pw_roll)
        layout.addWidget(self.pw_pitch)
        layout.addWidget(self.pw_yaw)

        plot_gb.setLayout(layout)
        return plot_gb

    def attitude_plot_cb(self, step, attitude, attitude_cmd):
        """ plot attitude (pitch, roll, yaw) and the cmd data
        """
        self.update_value_list(self.pitch_list, math.degrees(attitude[0]))
        self.update_value_list(self.roll_list, math.degrees(attitude[1]))
        self.update_value_list(self.yaw_list, math.degrees(attitude[2]))

        self.plot_pitch.setData(self.pitch_list, pen=self.pen_red)
        self.plot_roll.setData(self.roll_list, pen=self.pen_red)
        self.plot_yaw.setData(self.yaw_list, pen=self.pen_red)

# reward plot groupbox
    def create_reward_plot_groupbox(self):
        reward_plot_groupbox = QGroupBox(title='Reward')
        layout = QHBoxLayout()
        reward_plot_groupbox.setFixedHeight(300)
        reward_plot_groupbox.setFixedWidth(600)

        self.reward_list = np.linspace(0, 0, self.max_len)
        self.total_reward_list = np.linspace(0, 0, self.max_len)

        self.rw_pw_1 = pg.PlotWidget(title='reward')
        self.rw_pw_1.showGrid(x=True,y=True)
        self.rw_p_1 = self.rw_pw_1.plot()

        self.rw_pw_2 = pg.PlotWidget(title='total reward')
        self.rw_pw_2.showGrid(x=True,y=True)
        self.rw_p_2 = self.rw_pw_2.plot()

        layout.addWidget(self.rw_pw_1)
        layout.addWidget(self.rw_pw_2)

        reward_plot_groupbox.setLayout(layout)
        return reward_plot_groupbox

    def reward_plot_cb(self, step, reward, total_reward):
        self.update_value_list(self.reward_list, reward)
        self.update_value_list(self.total_reward_list, total_reward)

        self.rw_p_1.setData(self.reward_list, pen=self.pen_red)
        self.rw_p_2.setData(self.total_reward_list, pen=self.pen_red)

# trajectory plot groupbox
    def create_traj_plot_groupbox(self):
        traj_plot_groupbox = QGroupBox('Trajectory')
        traj_plot_groupbox.setFixedHeight(600)
        traj_plot_groupbox.setFixedWidth(600)
        layout = QVBoxLayout()

        self.traj_pw = pg.PlotWidget(title='trajectory')
        self.traj_pw.showGrid(x=True,y=True)
        # self.traj_pw.setXRange(max=350, min=-100)
        # self.traj_pw.setYRange(max=100, min=-300)
        self.traj_pw.setXRange(max=60, min=-60)
        self.traj_pw.setYRange(max=60, min=-60)
        self.traj_plot = self.traj_pw.plot()
        self.traj_pw.invertY()

        if self.cfg.get('options', 'env_name') == 'SimpleAvoid':
            background_image_path = 'resources/env_maps/simple_world_light.png'
            img_data = Image.open(background_image_path)
            image=np.copy(img_data)
            self.background_img = pg.ImageItem(image)
            self.traj_pw.addItem(self.background_img)
            self.background_img.setZValue(-100)  # make sure image is behind other data
            self.background_img.setRect(pg.QtCore.QRectF(-60, -60, 120, 120))

        layout.addWidget(self.traj_pw)

        traj_plot_groupbox.setLayout(layout)
        return traj_plot_groupbox

    def traj_plot_cb(self, goal, start, current_pose, trajectory_list):
        """
        Plot trajectory
        """
        # clear plot
        self.traj_pw.clear()

        # set background image
        if self.cfg.get('options', 'env_name') == 'SimpleAvoid':
            self.traj_pw.addItem(self.background_img)
        
        # plot start, goal and trajectory
        self.traj_pw.plot([start[0]], [start[1]], symbol='o')
        self.traj_pw.plot([goal[0]], [goal[1]], symbol='o')
        self.traj_pw.plot(trajectory_list[..., 0], trajectory_list[..., 1], pen=self.pen_red)


def main():
    config_file = 'configs/config_new.ini'
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config_file)
    gui.show()

    sys.exit(app.exec_())
    print('Exiting program')


if __name__ == "__main__":
    main()
