import cv2
import os
import json
import numpy as np

# Global ROI point list
roi_points = []

# ========== CONFIG ==========
VIDEO_PATH = "H:/Projects/Anomaly-Detection-In-CCTV-Footage/Moving Crowd Anomaly/test001.mp4"  
FRAME_INDEX = 50
FRAME_SAVE_PATH = "temp_roi_frame.png"
ROI_JSON_PATH = "roi_config/restricted_area.json"

# ========== DRAW ROI FUNCTION ==========
def draw_roi(event, x, y, flags, param):
    global roi_points

    if event == cv2.EVENT_LBUTTONDOWN:
        scale_x, scale_y = param["scale"]
        original_x = int(x / scale_x)
        original_y = int(y / scale_y)
        roi_points.append((original_x, original_y))

# ========== SAVE ROI ==========
def save_roi(path="roi_config/restricted_area.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(roi_points, f)

# ========== STEP 1: Extract Frame ==========
def extract_frame(video_path, frame_index, save_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to extract frame {frame_index} from {video_path}")
    
    cv2.imwrite(save_path, frame)
    print(f"[✓] Frame {frame_index} saved as {save_path}")
    return frame

# ========== STEP 2: Launch ROI Drawing GUI ==========
def launch_roi_gui(image_path):
    global roi_points

    img = cv2.imread(image_path)
    clone = img.copy()

    target_width = 960
    aspect_ratio = img.shape[1] / img.shape[0]
    target_height = int(target_width / aspect_ratio)

    scale_x = target_width / img.shape[1]
    scale_y = target_height / img.shape[0]

    cv2.namedWindow("Draw ROI - Click to Add Points")
    cv2.setMouseCallback("Draw ROI - Click to Add Points", draw_roi, param={"scale": (scale_x, scale_y)})

    while True:
        temp = clone.copy()
        for point in roi_points:
            cv2.circle(temp, point, 5, (0, 0, 255), -1)
        if len(roi_points) > 1:
            cv2.polylines(temp, [np.array(roi_points)], isClosed=True, color=(255, 0, 0), thickness=2)

        
        # cv2.imshow("Draw ROI - Click to Add Points", temp) #for og frames
        
        # for resized frame
        display = cv2.resize(temp, (target_width, target_height))
        cv2.imshow("Draw ROI - Click to Add Points", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            save_roi(ROI_JSON_PATH)
            print(f"[✓] ROI saved to {ROI_JSON_PATH}")
            break
        elif key == ord("q"):
            print("[x] ROI drawing canceled.")
            break

    cv2.destroyAllWindows()

# ========== MAIN ==========
if __name__ == "__main__":
    frame = extract_frame(VIDEO_PATH, FRAME_INDEX, FRAME_SAVE_PATH)
    launch_roi_gui(FRAME_SAVE_PATH)
