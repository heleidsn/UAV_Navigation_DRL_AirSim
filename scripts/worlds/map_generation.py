'''
Generate world map according to : https://microsoft.github.io/AirSim/voxel_grid/
'''

import airsim
import os
c = airsim.VehicleClient()
center = airsim.Vector3r(0, 0, 0)

output_path = os.path.join(os.getcwd(), "map.binvox")
c.simCreateVoxelGrid(center, 200, 200, 100, 0.5, output_path)