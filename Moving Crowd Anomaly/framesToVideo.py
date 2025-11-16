import cv2
import os
import glob

# === CONFIGURATION ===
frame_folder = "H:/Anomaly Detection in cctv/archive (1)/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test003"  # Folder containing .tif images
output_video = "test003.mp4"         # Output video file
fps = 25                                  # Frames per second

# === FETCH ALL TIF FRAMES ===
frame_paths = sorted(glob.glob(os.path.join(frame_folder, "*.tif")))

# Read the first frame to get the frame size
first_frame = cv2.imread(frame_paths[0])
height, width, layers = first_frame.shape

# === VIDEO WRITER SETUP ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === WRITE FRAMES ===
for frame_path in frame_paths:
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

video_writer.release()
print(f"Video saved as: {output_video}")
