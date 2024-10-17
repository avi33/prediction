import numpy as np
import time

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
        Q = self.q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                               [0, dt ** 3 / 3, 0, dt ** 2 /2],
                               [dt ** 2/ 2, 0, dt, 0],
                               [0, dt ** 2 / 2, 0, dt]])
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
    

if __name__ == "__main__":
    KF = KalmanFilter(2, 4, 1, 1)
    Q = KF._Q(1/10)
    print(Q)