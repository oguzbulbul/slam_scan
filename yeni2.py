import numpy as np
import matplotlib.pyplot as plt

# Load point cloud from txt file
point_cloud = []
with open('point_cloud.txt', 'r') as f:
    for line in f:
        x, y = map(float, line.strip().split())
        point_cloud.append((x, y))

# Convert to numpy array for easier handling
point_cloud = np.array(point_cloud)

# Plot the point cloud
plt.figure()
plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=1)
plt.title('LIDAR Point Cloud')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.axis('equal')
plt.show()
