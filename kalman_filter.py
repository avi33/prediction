import numpy as np
import time
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, dim_meas, dim_model, meas_noise_std, process_noise_std) -> None:        
        self.dim_model = dim_model
        self.process_noise_std = process_noise_std
        self.I = np.eye(dim_model)
        self.R = self.I * (meas_noise_std ** 2)
        self.t_ahead = 0.2
        self.x_est = None
        self.P_est = None
        self.x_pred = None
        self.P_pred = None        
        self.y_prev = None
        self.t_prev_pred = None
        self.t_last_update = None
        
    def _A(self, dt):
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]
                      ])
        return A

    def _Q(self, dt):
        Q = self.process_noise_std * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                                                [0, dt ** 4 / 3, 0, dt ** 3 /2],
                                                [dt ** 3/ 2, 0, dt**2, 0],
                                                [0, dt ** 3 / 2, 0, dt**2]])
        
        return Q

    def _is_initialized(self):
        return self.t_last_update is not None
    
    def predict(self, t):
        dt = t - self.t_prev_pred + self.t_ahead
        A = self._A(dt)
        Q = self._Q(dt)
        self.P_pred = A @ self.P_est @ A.T + Q        
        self.x_pred = A @ self.x_est
        self.t_prev_pred = t
        return self.x_pred[:2]

    def update(self, y, t):
        if not self._is_initialized() or t - self.t_last_update > 1:
            self.reset(y, t)
            return self.x_est, self.P_est
        
        v = (y - self.y_prev) / (t - self.t_last_update)
        z = np.hstack((y, v))
        err = z - self.C @ self.x_pred
        
        S = self.C @ self.P_pred @ self.C.T + self.R

        K = self.P_pred @ self.C.T @ np.linalg.inv(S)

        self.x_est = self.x_pred + K @ err

        self.P_est = (self.I - K @ self.C) @ self.P_pred

        self.y_prev = y
        self.t_last_update = t

        return self.x_est[:2]
    
    def reset(self, y, t):
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           ])
        self.t_last_update = t
        self.t_prev_pred = t
        self.x_est = np.array((y[0], y[1], 0.0001, 0.0001))
        self.x_pred = self.x_est
        self.P_est = np.eye(self.dim_model)
        self.P_pred = self.P_est
        # self.P_est[2:, 2:] *= 10
        self.y_prev = y
    

# Function to simulate 2D point motion with constant velocity
def simulate_2d_motion(noise_std=1):
    # Parameters for the 2D motion
    p0 = np.array([0, 0])  # Initial position (x0, y0)
    v = np.array([1, 1])  # Constant velocity (vx, vy)
    fs = 10  # Frequency of measurements (e.g., 10 Hz)
    time_steps = np.arange(0, 10, 1/fs)  # Time steps from 0 to 5 seconds with fs frequency


    positions = []
    measurements = []
    for t in time_steps:
        # True position without noise
        p_true = p0 + v * t
        positions.append(p_true)
        
        # Measurement with added Gaussian noise
        noise = np.random.randn(*p_true.shape) * noise_std
        p_measured = p_true + noise
        measurements.append(p_measured)
        
    # Simulate the motion and noisy measurements
    return(np.array(positions), np.array(measurements), time_steps)

if __name__ == "__main__":
    # Initialize the RLS system with measurement standard deviation
    noise_std = 1
    KF = KalmanFilter(4, 4, noise_std, 1)

    z_true, z, t = simulate_2d_motion(noise_std=noise_std)
    dt = np.mean(np.diff(t))
    z_pred = []
    for i in range(t.shape[-1]):                        
        KF.update(z[i], t[i])
        x_est = KF.predict(t=t[i])
        z_pred.append(x_est[:2])
    z_pred = np.array(z_pred)

    fig, ax = plt.subplots(3)

    ax[0].plot(t, z_true[:, 0])
    ax[0].plot(t, z[:, 0])        
    ax[0].plot(t, z_pred[:, 0])
    ax[0].legend(['true', 'meas', 'pred'])
    ax[0].set_xlabel('time[sec]')
    ax[0].set_ylabel('x')

    ax[1].plot(t, z_true[:, 1])    
    ax[1].plot(t, z[:, 1])        
    ax[1].plot(t, z_pred[:, 1])
    ax[1].legend(['true', 'meas', 'pred'])
    ax[1].set_xlabel('time[sec]')
    ax[1].set_ylabel('y')

    ax[2].plot(t, z_true[:, 0] - z_pred[:, 0])
    ax[2].plot(t, z_true[:, 1] - z_pred[:, 1])
    ax[2].legend(['x', 'y'])
    ax[2].set_ylabel('err')
    ax[2].set_ylabel('time[sec]')
    plt.show()