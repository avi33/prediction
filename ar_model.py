from typing import Any
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class ARModel:
    def __init__(self, p=1):
        pass

    @staticmethod
    def unfold(x, kernel_size, stride):
        input_size = x.shape[0]

        unfolded_size = (input_size - kernel_size) // stride + 1

        shape = (unfolded_size, kernel_size)
        strides = (stride * x.strides[0], x.strides[0])

        unfolded_array = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        
        return unfolded_array
    
    def build_equations(self, x, win_len, step):
        y = x[win_len::step]
        X = np.flip(self.unfold(x[:-step], kernel_size=win_len, stride=step), axis=-1)
        return X, y

    @staticmethod
    def solve_yw(X, y):
        A = X.T @ X
        b = X.T @ y
        theta = np.linalg.lstsq(A, b)[0]
        return theta   
    
    def step(self, x, win_len, step=1):
        X, y = self.build_equations(x, win_len, step)
        theta = self.solve_yw(X, y)
        y_pred = np.concatenate((x[:win_len], X @ theta))
        return theta, y_pred
        

if __name__ == "__main__":
    fs = 20
    f = 12
    t = np.arange(0, 1, 1/fs) * fs
    x = np.sin(2*np.pi*f*t)
    std_noise = 0.01
    AR = ARModel()
    y = t + np.random.randn(*x.shape)*std_noise
    theta, y_pred = AR.step(y, win_len=6, step=1)
    err = y_pred-y
    err_rel = 20*np.log10(np.abs(err).mean()/np.abs(y).mean())
    err_abs = 20*np.log10(np.abs(err).mean())
    print(20*np.log10(std_noise))
    plt.plot(y, '.-')
    plt.plot(y_pred, 'ro-')        
    plt.title("rel err:{:.2f} abs err:{:.2f}".format(err_rel, err_abs))
    plt.grid(True)
    plt.show()    