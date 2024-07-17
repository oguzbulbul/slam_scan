import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from rplidar import RPLidar

# EKF SLAM variables
state_size = 3  # [x, y, theta] - State size
control_size = 2  # [v, omega] - Control size
measurement_size = 2  # [range, bearing] - Measurement size

# Initial state [x, y, theta]
state = np.zeros(state_size)
# Initial covariance
covariance = np.eye(state_size) * 0.1  # Initial covariance matrix

# Process noise covariance
process_noise = np.diag([0.1, 0.1, np.deg2rad(5)])  # Control noise

# Measurement noise covariance
measurement_noise = np.diag([0.1, np.deg2rad(1)])  # Measurement noise

# State transition model
def motion_model(state, control, delta_t):
    theta = state[2]
    v = control[0]
    omega = control[1]

    if omega == 0:
        dx = v * delta_t * np.cos(theta)
        dy = v * delta_t * np.sin(theta)
        dtheta = 0
    else:
        dx = -(v / omega) * np.sin(theta) + (v / omega) * np.sin(theta + omega * delta_t)
        dy = (v / omega) * np.cos(theta) - (v / omega) * np.cos(theta + omega * delta_t)
        dtheta = omega * delta_t

    return state + np.array([dx, dy, dtheta])

# Jacobian of the motion model
def jacobian_motion_model(state, control, delta_t):
    theta = state[2]
    v = control[0]
    omega = control[1]

    if omega == 0:
        J = np.array([
            [1, 0, -v * delta_t * np.sin(theta)],
            [0, 1, v * delta_t * np.cos(theta)],
            [0, 0, 1]
        ])
    else:
        J = np.array([
            [1, 0, (v / omega) * (np.cos(theta) - np.cos(theta + omega * delta_t))],
            [0, 1, (v / omega) * (np.sin(theta + omega * delta_t) - np.sin(theta))],
            [0, 0, 1]
        ])

    return J

# Measurement model
def measurement_model(state, landmark):
    dx = landmark[0] - state[0]
    dy = landmark[1] - state[1]
    range_ = np.sqrt(dx**2 + dy**2)
    bearing = math.atan2(dy, dx) - state[2]

    return np.array([range_, bearing])

# Jacobian of the measurement model
def jacobian_measurement_model(state, landmark):
    dx = landmark[0] - state[0]
    dy = landmark[1] - state[1]
    range_ = np.sqrt(dx**2 + dy**2)

    H = np.array([
        [-dx / range_, -dy / range_, 0],
        [dy / (range_**2), -dx / (range_**2), -1]
    ])

    return H

# EKF SLAM algorithm
def ekf_slam(state, covariance, control, measurements, landmarks, delta_t):
    # Prediction step
    state_pred = motion_model(state, control, delta_t)  # State prediction
    F = jacobian_motion_model(state, control, delta_t)  # Jacobian of the motion model
    covariance_pred = F @ covariance @ F.T + process_noise  # Covariance update

    # Update step
    for i, landmark in enumerate(landmarks):
        measurement_pred = measurement_model(state_pred, landmark)  # Predicted measurement
        H = jacobian_measurement_model(state_pred, landmark)  # Jacobian of the measurement model

        S = H @ covariance_pred @ H.T + measurement_noise  # Innovation covariance
        K = covariance_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

        innovation = measurements[i] - measurement_pred  # Innovation
        innovation[1] = ((innovation[1] + np.pi) % (2 * np.pi)) - np.pi  # Normalize angle

        state_pred = state_pred + K @ innovation
        covariance_pred = (np.eye(len(covariance_pred)) - K @ H) @ covariance_pred

    return state_pred, covariance_pred

# Example control input
control = np.array([1.0, 0.1])  # [v, omega]
delta_t = 0.1  # Time step

# Visualization data storage
trajectory = []
landmark_positions = []

# Initialize RPLIDAR
lidar = RPLidar('/dev/ttyUSB0',baudrate=256000)
lidar.reset()
info = lidar.get_info()
print(info)
 
health = lidar.get_health()
print(health)
# Process LIDAR data and run EKF SLAM
try:
    measurements = []
    for i, scan in enumerate(lidar.iter_scans()):
        for (_, angle, distance) in scan:
            print("Angle:{:.2f}, distance:{:.2f}".format(angle, distance))
            #log the data to a txt file
            with open('lidar_data.txt', 'a') as f:
                f.write("{:.2f},{:.2f}\n".format(angle, distance))
        # Convert LIDAR data to measurements
        
            measurements = np.array([[distance, angle] for (_, angle, distance) in scan])
        print('%d: Got %d measurments' % (i, len(scan)))
        if i > 1_000:
            lidar.reset()
            break
        # Landmark coordinates
        landmarks = np.array([[item[2] * np.cos(np.deg2rad(item[1])), item[2] * np.sin(np.deg2rad(item[1]))] for item in scan if item[2] > 0])

        # Run EKF SLAM
        state, covariance = ekf_slam(state, covariance, control, measurements, landmarks, delta_t)

        # Store results
        trajectory.append(state.copy())
        landmark_positions.append(landmarks)

        # Visualization
        plt.figure()
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=50, c='b', marker='.', label='Landmarks')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.title('EKF SLAM')
        plt.show()

except KeyboardInterrupt:
    print('Stopping...')
finally:
    lidar.stop()
    lidar.disconnect()
