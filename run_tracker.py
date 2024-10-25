import numpy as np
import time
import cv2
from kalman_filter import KalmanFilter
from ultralytics import YOLO


def run():
    fname = r"man.mp4"
    fname_res = r"res.mp4"
    tacker = Tracker()
    cap = cv2.VideoCapture(fname)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{w}/{h} - {fps}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap_out = cv2.VideoWriter(fname_res, fourcc, fps, (w, h))

    # Check if video capture and writer are successfully opened
    if not cap.isOpened():
        print("Error: Could not open video.")
    if not cap_out.isOpened():
        print("Error: Could not open video writer.")

    frame_id = 0
    # Process the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_id += 1
            t = frame_id / fps
            tacker.step(frame, t, frame_id)            
            # Write the fr ame to the output file
            # cap_out.write(frame)
        else:
            # Break the loop if no more frames are available
            break
        
        cv2.imshow('', frame)
        cv2.waitKey(1)
    # Release the VideoCapture and VideoWriter objects
    cv2.destroyAllWindows()
    cap.release()
    cap_out.release()

    print("Video processing completed and saved to", fname_res)

class Tracker:
    def __init__(self):
        self.det = YOLO("yolo11m-pose.pt", verbose=False)  # load an official model
        self.track = KalmanFilter(4, 4, 2, 1)
    
    def step(self, frame, t, frame_id):
        preds = self.det.predict(frame, verbose=False)[0]
        kpts = preds.keypoints.data[0]        
        if kpts.shape[0] > 0:
            ii = preds.boxes.conf.argmax()
            if preds.boxes.conf[ii] > 0.5:            
                com = kpts[[5, 6, 11, 12], :2].mean(dim=0)
                if com.sum() != 0 and t > 2:                
                    com = com.cpu().int().numpy()[:2]
                    frame = cv2.circle(frame, com, radius=1, color=(0, 255, 0), thickness=2)                    
                    _ = self.track.update(com, t)
                    pnt_pred = self.track.predict(t)                                            
                    frame = cv2.circle(frame, pnt_pred.astype(np.int32), radius=1, color=(0, 0, 255), thickness=2)
        return



if __name__ == "__main__":
    run()