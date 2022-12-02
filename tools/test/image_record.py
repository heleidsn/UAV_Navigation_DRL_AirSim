# 采集图像，用于测试LGMD算法
import airsim
import os
import math
import cv2
import numpy as np

# 设置前向速度、dt和时间
forward_speed = 10
dt = 0.1
time = 10

data_path = 'img_save/'
description = "forest_pitch_roll_new"
tmp_dir_rgb = os.path.join(data_path, description, "rgb")
tmp_dir_depth = os.path.join(data_path, description, "depth")
os.makedirs(tmp_dir_rgb, exist_ok=True)
os.makedirs(tmp_dir_depth, exist_ok=True)

start_pose = [-100, 0, 8]

# flapping
pitch_flap_hz = 8
pitch_flap_deg = 3
roll_change_deg = 40
roll_change_hz = 1 / 5

x = start_pose[0]
y = start_pose[1]
z = start_pose[2]
pitch = 0
roll = 0
yaw = 0


client = airsim.VehicleClient()
client.confirmConnection()

pose = client.simGetVehiclePose()
pose.position.x_val = x
pose.position.y_val = y
pose.position.z_val = -z
pose.orientation = airsim.to_quaternion(0, 0, yaw)
client.simSetVehiclePose(pose, False)

# change camera FoV
camera = client.simGetCameraInfo("0")
client.simSetCameraFov("0", 90)
camera = client.simGetCameraInfo("0")

for i in range(int(time/dt)):
    x = x + forward_speed * dt
    pose = client.simGetVehiclePose()
    pose.position.x_val = x
    
    # change attitude
    time = i * dt
    
    pitch = pitch_flap_deg * math.sin(2 * math.pi * time * pitch_flap_hz)
    pitch = math.radians(pitch)
    
    roll = roll_change_deg * math.sin(2 * math.pi * time * roll_change_hz)
    roll = math.radians(roll)
    
    a_n = math.tan(roll) * 9.8
    yaw_rate = a_n / forward_speed
    
    yaw = yaw + yaw_rate * dt
    if yaw > math.radians(180):
        yaw -= math.pi * 2
    elif yaw < math.radians(-180):
        yaw += math.pi * 2
    
    x += forward_speed * math.cos(yaw) * dt
    y += forward_speed * math.sin(yaw) * dt
    
    pose = client.simGetVehiclePose()
    pose.position.x_val = x
    pose.position.y_val = y
    pose.position.z_val = - z
    pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
    client.simSetVehiclePose(pose, False)
    
    # save image
    responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
            airsim.ImageRequest("0", airsim.ImageType.Scene),           #scene vision image in png format
            ])
    png_image = responses[1]
    depth_image = responses[0]
    depth_img = airsim.list_to_2d_float_array(depth_image.image_data_float, depth_image.width, depth_image.height)
    depth_meter = depth_img * 100
    depth_meter_clip = np.clip(depth_meter, 0, 100)
    cv2.imshow('depth', depth_meter_clip/100)
    cv2.waitKey(1)
    
    filename_rgb = os.path.join(tmp_dir_rgb, str(i))
    filename_depth = os.path.join(tmp_dir_depth, str(i))
    airsim.write_file(os.path.normpath(filename_rgb + '.png'), png_image.image_data_uint8) # save rgb img as png
    airsim.write_pfm(os.path.normpath(filename_depth + '.pfm'), airsim.get_pfm_array(depth_image))  # save depth image as pfm
    

# client.stopRecording()
print('end')