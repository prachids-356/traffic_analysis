import cv2
from video_stream import frame_generator
from detector import detect
from tracker import CentroidTracker

tracker = CentroidTracker()
for i, (frame_id, frame) in enumerate(frame_generator()):
    detections = detect(frame)
    if detections:
        print(f"Frame {frame_id}: {len(detections)} detections")
    else:
        print(f"Frame {frame_id}: 0 detections")
    objects, counts = tracker.update(detections)
    if counts:
        print(f"  Crossing! {counts}")
    if i > 50:
        break
