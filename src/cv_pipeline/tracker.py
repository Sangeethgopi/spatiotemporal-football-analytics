import os
import cv2
import argparse
from ultralytics import YOLO

def run_tracking(video_path: str, output_path: str):
    """
    Runs YOLOv8 object detection and ByteTrack multi-object tracking 
    on a football video clip.
    """
    print(f"Loading YOLOv8 model...")
    # Using YOLOv8n (nano) for speed in prototyping.
    # In a real thesis environment, we would use a model fine-tuned on football (e.g., from Roboflow)
    model = YOLO("yolov8n.pt")
    
    print(f"Running ByteTrack on {video_path}...")
    
    # We only want to track 'person' (class 0) and 'sports ball' (class 32)
    # The tracker="bytetrack.yaml" specifies the tracking algorithm
    results_generator = model.track(
        source=video_path,
        conf=0.25, 
        iou=0.5,
        classes=[0, 32], # 0: person, 32: sports ball
        tracker="bytetrack.yaml",
        stream=True
    )
    
    results = []
    for i, r in enumerate(results_generator):
        results.append(r)
        if (i + 1) % 500 == 0:
            print(f"  Tracked {i + 1} frames...")
            
    print(f"Tracking complete: {len(results)} frames processed.")
    
    # We also want to export the tracking data to a CSV for our pipeline
    # The results object contains bounding boxes and tracking IDs for every frame.
    csv_out = output_path.replace('.mp4', '_tracking.csv')
    export_tracking_data(results, csv_out)

def export_tracking_data(results, csv_path):
    import pandas as pd
    
    tracking_data = []
    
    for frame_idx, r in enumerate(results):
        boxes = r.boxes
        for box in boxes:
            # Check if the box has a tracking ID (sometimes they drop)
            if box.id is not None:
                track_id = int(box.id.item())
                cls = int(box.cls.item())
                # Get the bounding box coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get the bottom-center of the bounding box (representing the player's feet)
                foot_x = (x1 + x2) / 2
                foot_y = y2
                
                tracking_data.append({
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class": "player" if cls == 0 else "ball",
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "foot_x": foot_x,
                    "foot_y": foot_y
                })
                
    df = pd.DataFrame(tracking_data)
    df.to_csv(csv_path, index=False)
    print(f"Raw Tracking data exported to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/sample_tactical.mp4", help="Path to input video")
    parser.add_argument("--out", type=str, default="data/sample_tracked.mp4", help="Path to output video")
    args = parser.parse_args()
    
    run_tracking(args.video, args.out)
