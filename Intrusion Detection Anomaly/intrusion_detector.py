import cv2
import json
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import os
import numpy as np
# ========== GLOBAL VARIABLES ==========
drawing = False
roi_points = []
restricted_polygon = None

# ========== MOUSE CALLBACK FUNCTION ==========
def draw_roi(event, x, y, flags, param):
    global drawing, roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# ========== SAVE ROI ==========
def save_roi(path="roi_config/restricted_area.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # ✅ Create folder if missing
    with open(path, "w") as f:
        json.dump(roi_points, f)

# ========== LOAD ROI ==========
def load_roi(path="roi_config/restricted_area.json"):
    with open(path, "r") as f:
        points = json.load(f)
    return [tuple(p) for p in points]

# ========== INTRUSION CHECK ==========
def is_intrusion(bbox_center, polygon):
    return polygon.contains(Point(bbox_center))

# ========== MAIN INTRUSION DETECTOR ==========
def run_intrusion_detector(video_path):
    global restricted_polygon

    # Load model
    model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt, etc.

    # Load ROI
    roi = load_roi()
    restricted_polygon = Polygon(roi)

    # Start video
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if is_intrusion((cx, cy), restricted_polygon):
                color = (0, 0, 255)
                label = "INTRUSION"
            else:
                color = (0, 255, 0)
                label = "Safe"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        # Draw ROI on frame
        cv2.polylines(frame, [np.array(roi, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)


        #for og frames 
        cv2.imshow("Intrusion Detection", frame)

        # for resized frames
        # display = cv2.resize(frame, (500, 200))  # Display only resize
        # cv2.imshow("Intrusion Detection", display)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "H:/Projects/Anomaly-Detection-In-CCTV-Footage/Moving Crowd Anomaly/test001.mp4" 
    run_intrusion_detector(video_path)
