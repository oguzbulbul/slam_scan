import rplidar
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import math

# Constants
PORT_NAME = '/dev/ttyUSB0'
BAUDRATE = 256000

# Initialize the LIDAR
lidar = rplidar.RPLidar(PORT_NAME, BAUDRATE)

# Function to process LIDAR data
def process_data(scan):
    distances = []
    angles = []
    for (_, angle, distance) in scan:
        distances.append(distance)
        angles.append(angle)
    return distances, angles

# Function to convert polar to cartesian coordinates
def polar_to_cartesian(distances, angles):
    x = [d * math.cos(math.radians(a)) for d, a in zip(distances, angles)]
    y = [d * math.sin(math.radians(a)) for d, a in zip(distances, angles)]
    return x, y

# Function to update the plot
def update(frame):
    scan = next(lidar.iter_scans())
    distances, angles = process_data(scan)
    x, y = polar_to_cartesian(distances, angles)
    
    ax.clear()
    ax.set_xlim(-8000, 8000)
    ax.set_ylim(-8000, 8000)
    ax.scatter(x, y, s=2)
    ax.add_patch(patches.Circle((0, 0), radius=1000, edgecolor='r', facecolor='none'))

# Initialize plot
fig, ax = plt.subplots()
ax.set_aspect('equal')

# Start animation
ani = animation.FuncAnimation(fig, update, interval=50)

plt.show()

# Stop LIDAR when done
lidar.stop()
lidar.disconnect()
