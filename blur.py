from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import cv2


class BadFrameDet:
    DEFAULT = -1
    def __init__(self, max_len=16):        
        self.buffer = deque(max_len)
    
    def reset(self):
        self.buffer.clear()

    def update_buffer(self, x):
        self.buffer.append(x)

    def step(self, frame):
        x = self.measure_blurriness(frame=frame)
        if len(self.buffer) < self.max_len:
            self.update_buffer(x)
            return False, self.DEFAULT
        
        score = self.calc_score()

    @staticmethod
    def measure_blurriness(frame):   
        frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))     
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score