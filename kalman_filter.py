import numpy as np
import time
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, dim_meas, dim_model, meas_noise_std, process_noise_std) -> None:
        self.dtype = np.float32

        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype = self.dtype)
                
        self.dim_model = dim_model
        self.q = process_noise_std
        self.I = np.eye(dim_model, dtype=self.dtype)
        self.t_ahead = 0.2 #sec
        self.R = np.eye(dim_meas) * meas_noise_std        
        self.x_est = np.zeros(dim_model, dtype=self.dtype)
        self.P_est = np.eye(dim_model, dtype=self.dtype)        
        self.P_est[2:, 2:] *= 1
        self.x_pred = self.x_est
        self.P_pred = self.P_est
        self.t_prev = None
        

    def _A(self, dt):
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype = self.dtype)
        return A

    def _Q(self, dt):
        Q = self.q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                               [0, dt ** 4 / 3, 0, dt ** 3 /2],
                               [dt ** 3/ 2, 0, dt**2, 0],
                               [0, dt ** 3 / 2, 0, dt**2]])
        return Q

       
    def predict(self, t):
        dt = t - self.t_prev + self.t_ahead
        A = self._A(dt)
        Q = self._Q(dt)
        self.P_pred = A @ self.P_est @ A.T + Q        
        self.x_pred = A @ self.x_est    
        return self.x_pred, self.P_pred    
    
    def update(self, y, t):
        if self.t_prev is None:
            self.reset(y, t)
            return self.x_est, self.P_est
        self.t_prev = t
        err = y - self.C @ self.x_pred
        if np.sqrt(np.mean(err**2)) > 100:
            self.reset(y, t)
            return self.x_est, self.P_est
        S = self.C @ self.P_pred @ self.C.T + self.R
        K = self.P_pred @ self.C.T @ np.linalg.inv(S)
        self.x_est = self.x_pred + K @ err
        self.P_est = (self.I - K @ self.C) @ self.P_pred
    
    def reset(self, y, t):
        self.t_prev = t
        self.x_est = np.array((y[0], y[1], 0, 0))
        self.P_est = np.eye(self.dim_model, dtype=self.dtype)
        self.P_est[2:, 2:] *= 10
    

# Function to simulate 2D point motion with constant velocity
def simulate_2d_motion(noise_std=0.1):
    # Parameters for the 2D motion
    p0 = np.array([0, 0])  # Initial position (x0, y0)
    v = np.array([1, 0.5])  # Constant velocity (vx, vy)
    fs = 10  # Frequency of measurements (e.g., 10 Hz)
    time_steps = np.arange(0, 5, 1/fs)  # Time steps from 0 to 5 seconds with fs frequency


    positions = []
    measurements = []
    for t in time_steps:
        # True position without noise
        p_true = p0 + v * t
        positions.append(p_true)
        
        # Measurement with added Gaussian noise
        noise = np.random.normal(0, noise_std, p_true.shape)
        p_measured = p_true + noise
        measurements.append(p_measured)
        
    # Simulate the motion and noisy measurements
    return(np.array(positions), np.array(measurements), time_steps)

if __name__ == "__main__":
    KF = KalmanFilter(2, 4, 1, 1)
    # Initialize the RLS system with measurement standard deviation
    noise_std=0.01    
    z_true, z, t = simulate_2d_motion(noise_std=noise_std)
    z_pred = []
    for i in range(t.shape[-1]):
        KF.update(z[i], t[i])
        x_est, P_est = KF.predict(t=t[i])        
        z_pred.append(x_est[:2])
    z_pred = np.array(z_pred)

    fig, ax = plt.subplots(2)
    ax[0].plot(t, z_true[:, 0])    
    ax[0].plot(t, z[:, 0])        
    ax[0].plot(t, z_pred[:, 0])    
    ax[1].plot(t, z_true[:, 1])    
    ax[1].plot(t, z[:, 1])        
    ax[1].plot(t, z_pred[:, 1])    
    plt.show()