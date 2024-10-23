import numpy as np
import matplotlib.pyplot as plt


class RLS:
    def __init__(self, noise_std):
        # Initial state (not yet initialized)        
        self.theta = None  # Initial state vector [x0, vx, y0, vy]
        self.t0 = None
        self.P_inv = None  # Inverse covariance matrix
        self.z_init = []  # Store initial measurements
        self.t_init = []  # Store initial time points
        R = np.eye(2)*noise_std**2
        self.R_inv = self._R_inv(R)
    
    def is_initialized(self):
        return self.theta is not None  # Check if initialized from two points

    @staticmethod
    def _R_inv(R):
        """ Compute the inverse of R (measurement noise covariance) """
        return np.linalg.inv(R)
    
    def _H(self, t):
        """ State transition matrix A based on time difference dt """
        dt = t - self.t0
        return np.array([[1, dt, 0,  0],
                         [0, 0, 1, dt],
                         ])

    def init(self):
        """ Initialize theta using two measurement points """
        # Calculate velocities
        H = np.array([
                     [1, self.t_init[0], 0, 0], 
                     [0, 0, 1, self.t_init[0]],
                     [1, self.t_init[1], 0, 0], 
                     [0, 0, 1, self.t_init[1]]
                     ])        
        A = H.T @ H
        z = np.array(self.z_init).ravel()
        b = H.T @ z
        self.theta = np.linalg.solve(A, b)
        self.P_inv = np.linalg.inv(A)

    def update(self, z, t):
        """ Perform RLS update using the new measurement z at time t """
        if not self.is_initialized():
            self.reset(z, t)  # Reset and store the first two samples
            return z  # Return the raw measurement before initialization

        H = self._H(t)

        # Prediction error
        e = z - H @ self.theta
                
        # Compute the Kalman gain using the matrix inversion lemma
        tmp = H.T @ self.R_inv @ H
        K = np.linalg.inv(self.P_inv + tmp) @ H.T @ self.R_inv
        
        # Update the parameter estimate
        self.theta = self.theta + K @ e
        
        # Update the inverse covariance matrix
        self.P_inv = self.P_inv + H.T @ self.R_inv @ H
                        
        return

    def predict(self, t):
        dt = t - self.t0
        """ Predict based on the current state """
        H = np.array([[1, t, 0, 0], 
                      [0, 0, 1, t]
                      ])
        z_pred = H @ self.theta
        return z_pred

    def reset(self, z, t):
        """ Store the first two measurements, and initialize when we have two points """
        if len(self.z_init) < 2:
            self.z_init.append(z)
            self.t_init.append(t)
            self.t0 = t

        if len(self.z_init) == 2:
            # Initialize the system using the first two samples
            self.init()

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

# Example usage
if __name__ == "__main__":
    # Initialize the RLS system with measurement standard deviation
    noise_std=0.01
    rls = RLS(noise_std=noise_std)
    z_true, z, t = simulate_2d_motion(noise_std=noise_std)
    z_pred = []
    for i in range(t.shape[-1]):
        rls.update(z[i], t[i])        
        if len(rls.z_init) < 2:            
            pred = z[i]
        else:
            pred = rls.predict(t[i])
        z_pred.append(pred)
    z_pred = np.array(z_pred)

    fig, ax = plt.subplots(2)
    ax[0].plot(t, z_true[:, 0])    
    ax[0].plot(t, z[:, 0])        
    ax[0].plot(t, z_pred[:, 0])    
    ax[1].plot(t, z_true[:, 1])    
    ax[1].plot(t, z[:, 1])        
    ax[1].plot(t, z_pred[:, 1])    
    plt.show()