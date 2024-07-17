import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load point cloud from file
with open('point_cloud.pkl', 'rb') as f:
    point_cloud = pickle.load(f)

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
