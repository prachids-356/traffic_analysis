import cv2
import time
import numpy as np
from detector import detect
from video_stream import frame_generator
from tracker import CentroidTracker
import annotator

def main():
    print("--- Traffic Analysis Live Camera Test ---")
    print("Press 'q' to quit the window.")
    
    # Use camera index 0 (usually the front/embedded webcam)
    camera_source = "0"
    
    # Initialize tracker
    tracker = CentroidTracker()
    
    # Start the generator (it loops by default, but for live it just reads)
    try:
        last_time = time.time()
        for frame_id, frame in frame_generator(path=camera_source):
            # 1. Detect
            detections = detect(frame)
            
            # 2. Track
            objects, _ = tracker.update(detections)
            
            # 3. Calculate FPS
            now = time.time()
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0
            last_time = now
            
            # 4. Annotate
            # objects is a dict of tracked objects
            # We need to pass the snapshot of objects to annotator.draw
            # annotator.draw returns JPEG bytes, we need to decode for imshow
            jpeg_bytes = annotator.draw(frame, objects, fps=fps)
            
            # Decode JPEG bytes back to BGR for display
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            annotated_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 5. Display
            cv2.imshow("TrafficVision AI - Live Camera Test", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Live test stopped.")

if __name__ == "__main__":
    main()
