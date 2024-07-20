import numpy as np
import time

class KalmanFilter:
    def __init__(self, dim_meas, dim_model, meas_noise_std, process_noise_std) -> None:
        self.dtype = np.float32

        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype = self.dtype)
                
        self.q = process_noise_std ** 2
        self.I = np.eye(dim_model, dtype=self.dtype)
        self.t_ahead = 0.25 #sec
        self.R = np.eye(dim_meas) * meas_noise_std        
        self.x_est = np.zeros(dim_model, dtype=self.dtype)
        self.P_est = np.eye(dim_model, dtype=self.dtype)
        self.P_est[2:, 2:] *= 10
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
        Q = np.array([[self.q * dt ** 3 / 3, 0, self.q * dt ** 2 / 2, 0],
                      [0, self.q * dt ** 3 / 3, 0, self.q * dt ** 2 /2],
                      [self.q * dt ** 2/ 2, 0, self.q * dt, 0],
                      [0, self.q * dt ** 2 / 2, 0, self.q * dt]])  
        return Q

       
    def predict(self, dt):        
        A = self._A(dt)
        Q = self._Q(dt)
        self.P_pred = A @ self.P_est @ A.T + Q        
        self.x_pred = A @ self.x_est        
    
    def update(self, y, dt):
        err = y - self.C @ self.x_pred
        S = self.C @ self.P_pred @ self.C.T + self.R
        K = self.P_pred @ self.C.T @ np.linalg.inv(S)
        self.x_est = self.x_pred + K @ err
        self.P_est = (self.I - K @ self.C) @ self.P_pred

        # num = self.P_pred @ self.C.T
        # denum = self.C @ self.P_pred @ self.C.T + self.R
        # K = num @ np.linalg.inv(denum)
        # IKC = self.I - K @ self.C
        # KRK = K @ self.R @ K.T
        # self.P_est = IKC @ self.P_pred @ IKC.T + KRK
        
        
        # self.x_est = self.x_pred + K @ err