import cv2
import time
import argparse
from detector import detect
from video_stream import frame_generator
from metrics import average_confidence

def run_benchmark(source, num_frames=100):
    print(f"--- TrafficVision AI Benchmark ---")
    print(f"Source: {source}")
    print(f"Processing {num_frames} frames...")
    
    total_detections = 0
    confidences = []
    start_time = time.time()
    processed_count = 0
    
    class_counts = {}

    try:
        for i, (frame_id, frame) in enumerate(frame_generator(path=source)):
            if i >= num_frames:
                break
            
            detections = detect(frame)
            processed_count += 1
            
            if detections:
                total_detections += len(detections)
                conf = average_confidence(detections)
                confidences.append(conf)
                
                for d in detections:
                    cls = d['class_name']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
            
            if i % 10 == 0:
                print(f" Progress: {i}/{num_frames} frames", end='\r')

        end_time = time.time()
        elapsed = end_time - start_time
        fps = processed_count / elapsed if elapsed > 0 else 0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        print("\n\n--- Results ---")
        print(f"Processed Frames: {processed_count}")
        print(f"Total Detections: {total_detections}")
        print(f"Average Detections/Frame: {total_detections/processed_count:.2f}")
        print(f"Average Confidence: {avg_conf:.3f}")
        print(f"Estimated Throughput: {fps:.2f} FPS")
        print("\nClass Distribution:")
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count}")
        
        print("-" * 30)

    except Exception as e:
        print(f"\nError during benchmark: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark traffic analysis model accuracy.")
    parser.get_default("source")
    parser.add_argument("--source", type=str, default="traffic.mp4", help="Path to video file or '0' for webcam")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process")
    
    args = parser.parse_args()
    run_benchmark(args.source, args.frames)
