import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from kalman_filter import KalmanFilter as KalmanFilter2

def initialize_kalman_filter(x0, dt):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    # dt = 1.0  # time step

    # State transition matrix
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # Measurement function
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # Process noise covariance
    process_noise_std = 10
    q = process_noise_std ** 2

    kf.Q = np.array([[q*dt**3/3, 0, q*dt**2/2, 0],
                     [0, q*dt**3/3, 0, q*dt**2/2],
                     [q*dt**2/2, 0, q*dt, 0],
                     [0, q*dt**2/2, 0, q*dt]])        
    print(kf.Q)
    # Measurement noise covariance
    r = 4.0  # measurement noise magnitude
    kf.R = r * np.eye(2)

    # Initial state covariance
    kf.P[:2, :2] *= 10.0

    # Initial state
    kf.x = np.hstack([x0, 0, 0])
    
    return kf

def simulate_trajectory(initial_position, velocity, steps, noise_std=0):
    trajectory = [initial_position]
    for _ in range(steps - 1):
        new_position = trajectory[-1] + velocity + np.random.normal(0, noise_std, 2)
        trajectory.append(new_position)
    return np.array(trajectory)

def pixelvelocity2velocity(w, h, horizontal_fov_deg, speed_m_s, fps):
    # Calculate pixels per meter
    vertical_fov_deg = 2 * np.arctan(np.tan(np.deg2rad(horizontal_fov_deg / 2)) / (w / h)) * 180 / np.pi
    vertical_fov_rad = np.deg2rad(vertical_fov_deg)
    
    # Distance covered by the vertical FOV at 15 meters
    distance_m = 15
    vertical_distance_covered = 2 * distance_m * np.tan(vertical_fov_rad / 2)
    pixels_per_meter_vertical = h / vertical_distance_covered

    # Speed in pixels per frame
    speed_pixels_per_frame = speed_m_s * pixels_per_meter_vertical / fps
    return speed_pixels_per_frame

def generate_trajectory(frame_shape, fov, fps, trajectory_type, steps=500, noise_std=1):
    frame_width, frame_height = frame_shape

    if trajectory_type == 'far_to_close':
        initial_position = np.array([frame_width / 2, frame_height])        
        velocity = pixelvelocity2velocity(frame_width, frame_height, fov, 2, fps)
        velocity = np.array([0, -velocity])
        trajectory = simulate_trajectory(initial_position, velocity, steps, noise_std)
    
    elif trajectory_type == 'side_to_side':
        initial_position = np.array([0, frame_height / 2])
        velocity = pixelvelocity2velocity(frame_width, frame_height, fov, 2, fps)
        velocity = np.array([velocity, 0])
        trajectory = simulate_trajectory(initial_position, velocity, steps, noise_std)
    
    elif trajectory_type == 'diagonal':
        initial_position = np.array([0, frame_height])
        velocity = pixelvelocity2velocity(frame_width, frame_height, fov, 2, fps)
        velocity = np.array([velocity, -velocity])
        trajectory = simulate_trajectory(initial_position, velocity, steps, noise_std)
    
    elif trajectory_type == 'circular':
        center = np.array([frame_width / 2, frame_height / 2])
        radius = min(frame_width, frame_height) // 4
        circle_points = []
        for angle in np.linspace(0, 2 * np.pi, steps):
            point = center + radius * np.array([np.cos(angle), np.sin(angle)])
            circle_points.append(point)
        trajectory = np.array(circle_points)
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    return trajectory

def plot_trajectory(trajectory, trajectory_est=None, frame_shape=None, save_path='trajectory.png'):
    frame_width, frame_height = frame_shape
    plt.figure(figsize=(12, 9))
    plt.clf()
    plt.xlim(0, frame_width)
    plt.ylim(0, frame_height)

    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
    if trajectory_est is not None:
        plt.plot(trajectory_est[:, 0], trajectory_est[:, 1], label='Trajectory Est', color='r')
    
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.legend()
    plt.title("Point Trajectory")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.grid(True)
    plt.savefig(save_path)  # Save the plot as an image file
    plt.show()
    plt.close()
    print(f"Trajectory plot saved as {save_path}")


if __name__ == "__main__":
    frame_shape = (1280*2, 940*2)
    steps = 500
    fov = 48
    noise_std = 4
    fps = 16
    dt = 1/fps
    t = np.arange(0, steps, 1) / fps

    trajectory_type = 'far_to_close'
    trajectory = generate_trajectory(frame_shape, fov, fps, trajectory_type, steps, noise_std)    
    
    x_est = np.zeros_like(trajectory)    
    kf = initialize_kalman_filter(x0 = trajectory[0], dt = dt)    
    kf2 = KalmanFilter2(dim_meas=2, dim_model=4, meas_noise_std=4, process_noise_std=10)
    kf2.x_pred = np.hstack([trajectory[0], 0, 0])

    x_est[0] = trajectory[0]
    for i, x, in enumerate(trajectory[1:, :]):
        kf.update(x)
        kf.predict()
        kf2.update(x, dt)
        kf2.predict(dt)        
       
        x_est[i+1, :] = kf2.x_est[:2]
    
    plot_trajectory(trajectory, x_est, frame_shape, save_path=r'results/' + trajectory_type + '.png')


    # trajectory_type = 'side_to_side'
    # trajectory = generate_trajectory(frame_shape, fov, fps, trajectory_type, steps, noise_std)    
    
    # x_est = np.zeros_like(trajectory)    
    # kf = initialize_kalman_filter(x0 = trajectory[0], dt = dt)    
    # x_est[0, :] = trajectory[0]
    # for i, x, in enumerate(trajectory[1:, :]):
    #     kf.update(x)
    #     kf.predict()        
    #     x_est[i+1, :] = kf.x[:2]
    
    # plot_trajectory(trajectory, x_est, frame_shape, save_path=r'results/' + trajectory_type + '.png')


    # trajectory_type = 'diagonal'
    # trajectory = generate_trajectory(frame_shape, fov, fps, trajectory_type, steps, noise_std)    
    
    # x_est = np.zeros_like(trajectory)    
    # kf = initialize_kalman_filter(x0 = trajectory[0], dt = dt)
    # x_est[0] = trajectory[0]
    # for i, x, in enumerate(trajectory[1:, :]):
    #     kf.update(x)
    #     kf.predict()
    #     x_est[i+1, :] = kf.x[:2]
    
    # plot_trajectory(trajectory, x_est, frame_shape, save_path=r'results/' + trajectory_type + '.png')

    

    # trajectory_type = 'circular'
    # trajectory = generate_trajectory(frame_shape, fov, fps, trajectory_type, steps, noise_std)    
    
    # x_est = np.zeros_like(trajectory)    
    # kf = initialize_kalman_filter(x0 = trajectory[0], dt = dt)
    # x_est[0] = trajectory[0]
    # for i, x, in enumerate(trajectory[1:, :]):
    #     kf.update(x)
    #     kf.predict()
    #     x_est[i+1, :] = kf.x[:2]
    
    # plot_trajectory(trajectory, x_est, frame_shape, save_path=r'results/' + trajectory_type + '.png')