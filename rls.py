import numpy as np

class RLS:
    def __init__(self, measurement_std):
        # Initial state (not yet initialized)
        self.t_prev = None
        self.theta = np.zeros(4)  # Initial state vector [x0, vx, y0, vy]
        self.P_inv = np.eye(4) * 1e-3  # Inverse covariance matrix
        self.z_init = []  # Store initial measurements
        self.t_init = []  # Store initial time points
        R = np.eye(2, dtype=np.float32) * measurement_std  # Measurement noise covariance
        self.R_inv = self._R_inv(R)  # Inverse of R
    
    def is_initialized(self):
        return len(self.z_init) >= 2  # Check if initialized from two points

    @staticmethod
    def _R_inv(R):
        """ Compute the inverse of R (measurement noise covariance) """
        return np.linalg.inv(R)
    
    def _A(self, dt):
        """ State transition matrix A based on time difference dt """
        return np.array([[1, dt, 0,  0],
                         [0,  0, 1, dt],
                         ], dtype=np.float32)

    def init(self):
        """ Initialize theta using two measurement points """
        # Calculate velocities
        H = np.array([
                     [self.t_init[0], 1, 0, 0], 
                     [0, 0, self.t_init[0], 1],
                     [self.t_init[1], 1, 0, 0], 
                     [0, 0, self.t_init[1], 1]
                     ])        
        self.theta = np.linalg.solve(H.T @ H, H.T @ np.array(self.z_init).ravel())

    def update(self, z, t):
        """ Perform RLS update using the new measurement z at time t """
        if not self.is_initialized():
            self.reset(z, t)  # Reset and store the first two samples
            return z  # Return the raw measurement before initialization

        dt = t# - self.t_prev
        A = self._A(dt)

        # Prediction error
        e = z - A @ self.theta
                
        # Compute the Kalman gain using the matrix inversion lemma
        At_R_inv_A = A.T @ self.R_inv @ A
        K = np.linalg.inv(self.P_inv + At_R_inv_A) @ A.T @ self.R_inv
        
        # Update the parameter estimate
        self.theta = self.theta + K @ e
        
        # Update the inverse covariance matrix
        self.P_inv = self.P_inv + A.T @ self.R_inv @ A

        # Update time
        self.t_prev = t
    
        return self.theta

    def predict(self, z, t):
        """ Predict based on the current state """
        H = np.array([[t, 1, 0, 0], 
                     [0, 0, t, 1]])
        z_pred = H @ self.theta
        return z_pred

    def reset(self, z, t):
        """ Store the first two measurements, and initialize when we have two points """
        if len(self.z_init) < 2:
            self.z_init.append(z)
            self.t_init.append(t)
        if len(self.z_init) == 2:
            # Initialize the system using the first two samples
            self.init()
        self.t_prev = t  # Set the previous time


# Example usage
if __name__ == "__main__":
    # Initialize the RLS system with measurement standard deviation
    rls = RLS(measurement_std=0.01)
    
    # Simulated time steps and measurements
    time_steps = [0, 1, 2, 3, 4, 5]
    measurements = np.array([[1, 2],   # Measurement at t=0
                             [2, 1.8], # Measurement at t=1
                             [2.5, 1.6], # Measurement at t=2
                             [3.5, 1.5], # Measurement at t=3
                             [4.5, 1.3], # Measurement at t=4
                             [5.5, 1.1]]) # Measurement at t=5
    
    # Iterate through time steps and print the predictions
    for t, z in zip(time_steps, measurements):
        rls.update(z, t)
        if len(rls.z_init) == 2:
            pred = rls.predict(z, t)
        # print(f"Time {t}: Prediction = {prediction}")
