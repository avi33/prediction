import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kalman_filter import KalmanFilter
import time
import random

# Parameters
speed_mps = 1  # speed in meters per second
fps = 10  # frames per second
scale = 320.9  # pixels per meter
distance_per_frame = (speed_mps / fps) * scale  # distance in pixels per frame
initial_position_1 = (50, 50)  # starting position for point 1
initial_position_2 = (50, 70)  # starting position for point 2 (after Kalman filter)
num_frames = 50  # number of frames to simulate
use_kalman = True
kf = KalmanFilter(2, 4, 1, 1)
# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 2560)  # set x limits of the image
ax.set_ylim(0, 1440)  # set y limits of the image
point1, = ax.plot([], [], 'ro', markersize=8, label='Original Point')  # point 1 to be animated
point2, = ax.plot([], [], 'bo', markersize=8, label='Kalman Filtered Point')  # point 2 to be animated

# Initialize the points' positions
def init():
    point1.set_data([], [])
    point2.set_data([], [])
    return point1, point2

# Update function for animation
def update(frame):
    # Update the original point (point 1)
    x1, y1 = initial_position_1
    x1 += distance_per_frame * frame
    point1.set_data([x1], [y1])  # x1 and y1 must be sequences

    # Simulate Kalman filter update for the second point (point 2)
    # For example, we could just add a small random noise to the original position
    # x2 = x1 + np.random.normal(0, 1)  # Simulated Kalman filtered x position
    # y2 = y1 + np.random.normal(0, 1)  # Simulated Kalman filtered y position    
    x = np.array((x1, y1))
    t_prev = t=time.time()
    kf.update(x, t_prev)
    t_rand = 1/16 + random.random() * (1/10-1/16)
    time.sleep(t_rand)
    t_now = time.time()
    x_est, P_est = kf.predict(t=t_now)

    point2.set_data([x_est[0]], [x_est[1]])  # x2 and y2 must be sequences
    dx = x_est[:2] - x
    print(dx, t_now-t_prev)
    print(kf.P_est)
    return point1, point2

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=1000/fps)

plt.title('Point Movement with Kalman Filter Simulation')
plt.xlabel('X Position (pixels)')
plt.ylabel('Y Position (pixels)')
plt.grid()
plt.legend()
plt.show()
