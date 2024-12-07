from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ar_model import ARModel


class BadFrameDet:
    DEFAULT = -1
    def __init__(self, max_len=16):        
        self.buffer = deque(maxlen=max_len)
        self.score_buffer = deque(maxlen=max_len)
        self.max_len = max_len        
        self.ar_model = ARModel(model_order=2)
        self.threshold = 500        

    def reset(self):
        self.buffer.clear()

    def update_buffer(self, x):
        self.buffer.append(x)

    def step(self, frame):
        s = self.measure_blurriness(frame=frame)
        self.update_buffer(s)
        if len(self.buffer) < self.max_len:            
            return False, self.DEFAULT
        
        x = np.array(self.buffer)
        theta, x_est = self.ar_model.step(x, step=1)
        score = np.mean((x - x_est)**2)        
        bad_frame_ind = score > self.threshold and self.buffer[0] > self.buffer[-1]

        if bad_frame_ind:  
            # plt.plot(x); 
            # plt.plot(x_est); 
            # plt.show()   
            print(np.sum(theta), score)
            self.buffer.pop()                        

        return bad_frame_ind, score

    @staticmethod
    def measure_blurriness(frame):   
        frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))     
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var().item()
        return blur_score