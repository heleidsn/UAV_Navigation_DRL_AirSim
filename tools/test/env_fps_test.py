# used to check env depth image fps

# results:
# only depth iamge:
#       40fps@80*60
#       33fps@100*80
#       10fps@176*120
# using "ViewMode": "NoDisplay" is very useful
# 感觉fps和地图的大小有关，但是和是否ssd无关

import airsim
import time

import gym
import gym_env

from configparser import ConfigParser

client = airsim.VehicleClient()
client.confirmConnection()

# change camera FoV
camera = client.simGetCameraInfo("0")
client.simSetCameraFov("0", 90)
camera = client.simGetCameraInfo("0")

# print(camera)

env = gym.make('airsim-env-v0')

cfg = ConfigParser()
cfg.read('configs\config_SimpleAvoid_SimpleMultirotor.ini')
env.set_config(cfg)

start_time = time.time()
x = 1  # displays the frame rate every 1 second
counter = 0

while True:
    counter += 1
    depth_meter = env.get_depth_image()

    if (time.time() - start_time) > x:
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
