import time
import numpy as np
import math
import pickle
from rplidar import RPLidar

# Constants
PORT_NAME = '/dev/ttyUSB0'
BAUDRATE = 256000
MAP_SIZE_PIXELS = 500
MAP_SIZE_METERS = 10

# Initialize the LIDAR
lidar = RPLidar(PORT_NAME, BAUDRATE)

# SLAM state
pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
map = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), dtype=np.float32)
point_cloud = []

# Convert LIDAR data to cartesian coordinates
def polar_to_cartesian(angle, distance):
    x = distance * math.cos(angle)
    y = distance * math.sin(angle)
    return x, y

# Update SLAM pose
def update_pose(scan, pose):
    # Dummy update function, replace with actual SLAM algorithm (e.g., EKF, Particle Filter)
    # Example: move forward by a small amount
    delta_theta = 0.1
    delta_x = 0.1 * math.cos(pose[2])
    delta_y = 0.1 * math.sin(pose[2])
    return np.array([pose[0] + delta_x, pose[1] + delta_y, pose[2] + delta_theta])

# Update map with LIDAR data
def update_map(scan, pose, map):
    global point_cloud
    for (_, angle, distance) in scan:
        angle_rad = np.deg2rad(angle) + pose[2]
        x, y = polar_to_cartesian(angle_rad, distance)
        map_x = int((x + MAP_SIZE_METERS / 2) / MAP_SIZE_METERS * MAP_SIZE_PIXELS)
        map_y = int((y + MAP_SIZE_METERS / 2) / MAP_SIZE_METERS * MAP_SIZE_PIXELS)
        if 0 <= map_x < MAP_SIZE_PIXELS and 0 <= map_y < MAP_SIZE_PIXELS:
            map[map_y, map_x] = 1  # Mark as occupied
            point_cloud.append((x, y))
    return map

# Main loop
try:
    for scan in lidar.iter_scans():
        print(f"Got {len(scan)} samples")
        for (_, angle, distance) in scan:
            print(f"Angle: {angle}, distance: {distance / 1000} m")
            
        # Update SLAM pose
        pose = update_pose(scan, pose)
        # Update map
        map = update_map(scan, pose, map)
        # Log or save the map and pose as needed
        print(f"Pose: {pose}")
        time.sleep(0.1)  # Sleep to simulate real-time processing

except KeyboardInterrupt:
    print("Stopping LIDAR scan.")
    lidar.stop()
    lidar.disconnect()

# Save point cloud to file
with open('point_cloud.pkl', 'wb') as f:
    pickle.dump(point_cloud, f)

# Clean up
lidar.stop()
lidar.disconnect()
