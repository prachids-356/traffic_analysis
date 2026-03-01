import cv2
from video_stream import frame_generator
from detector import _get_model

model = _get_model()
for i, (frame_id, frame) in enumerate(frame_generator()):
    results = model(frame, conf=0.1, imgsz=640, device="cpu", verbose=False)
    has_boxes = False
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f"Frame {frame_id}: Detected {name} (id {cls_id}) with conf {conf:.3f}")
                has_boxes = True
    if not has_boxes:
        print(f"Frame {frame_id}: Nothing detected")
    if i > 5:
        break
