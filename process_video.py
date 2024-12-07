import cv2
import numpy as np


def process_video(video_path):
    """
    Measure blurriness for each frame in a video.
    :param video_path: Path to the video file.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_count += 1
        blurriness = measure_blurriness(frame)
        print(f"Frame {frame_count}: Blurriness Score = {blurriness:.2f}")
    
    cap.release()
    print("Finished processing video.")

def process_image(image_path):
    """
    Measure blurriness of a single image.
    :param image_path: Path to the image file.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Cannot load image.")
        return
    
    blurriness = measure_blurriness(image)
    print(f"Image Blurriness Score = {blurriness:.2f}")

# Example usage:
# Uncomment the appropriate line below to test with a video or image

# Process a video file
# process_video("path_to_video.mp4")

# Process a single image
# process_image("path_to_image.jpg")
