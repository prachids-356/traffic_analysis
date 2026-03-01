"""
diag.py – Quick pipeline diagnostic: prints raw YOLO results from first 20 sampled frames.
"""
import cv2
from config import VIDEO_PATH, FRAME_SKIP, RESIZE_WIDTH, RESIZE_HEIGHT, LINE_Y, CONFIDENCE_THRESHOLD, MODEL_PATH
from ultralytics import YOLO

print(f"Video: {VIDEO_PATH}")
print(f"Line Y: {LINE_Y}, Frame skip: {FRAME_SKIP}, Confidence: {CONFIDENCE_THRESHOLD}")
print(f"Resize: {RESIZE_WIDTH}x{RESIZE_HEIGHT}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Cannot open video!")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video info: {w}x{h}, {fps:.1f} fps, {total_frames} total frames")
cap.release()

print("\nLoading YOLO model...")
model = YOLO(MODEL_PATH)
VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0
sampled = 0
total_detections = 0

while sampled < 20:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue
    sampled += 1
    resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    results = model(resized, conf=0.1, imgsz=640, device="cpu", verbose=False)
    found = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cy = (y1 + y2) // 2
            found.append(f"{name}(conf={conf:.2f}, cy={cy})")
            total_detections += 1
    veh = [f for f in found if any(v in f for v in VEHICLE_IDS.values())]
    print(f"Frame {frame_id:4d}: {len(found)} objects, {len(veh)} vehicles | {', '.join(veh[:5]) or 'none'}")

cap.release()
print(f"\nSummary: {total_detections} total objects across {sampled} sampled frames")
print(f"LINE_Y={LINE_Y} — vehicles with cy near this value would trigger counting")
