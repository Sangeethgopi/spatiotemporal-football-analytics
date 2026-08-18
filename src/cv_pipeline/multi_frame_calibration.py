import cv2
import numpy as np
import json
import argparse
import os
import sys
from camera_tracker import CameraTracker

def run_multi_frame_calibration(video_path: str, clicks_json_path: str, output_matrix_path: str):
    """
    Reads a JSON file containing user clicks across multiple frames.
    Uses optical flow to back-project all clicks to Frame 0's coordinate space.
    Computes and saves the global homography matrix.
    
    Expected JSON structure for clicks:
    [
        {"frame_idx": 15, "img_x": 450, "img_y": 300, "pitch_x": 0.0, "pitch_y": 0.0},
        ...
    ]
    """
    with open(clicks_json_path, 'r') as f:
        clicks = json.load(f)
        
    if len(clicks) < 4:
        raise ValueError("At least 4 points are required for Homography.")
        
    # Sort clicks by frame_idx so we know when to stop processing the video
    clicks.sort(key=lambda x: x['frame_idx'])
    max_frame = clicks[-1]['frame_idx']
    
    # Group clicks by frame for easy lookup during video processing
    clicks_by_frame = {}
    for c in clicks:
        f_idx = c['frame_idx']
        if f_idx not in clicks_by_frame:
            clicks_by_frame[f_idx] = []
        clicks_by_frame[f_idx].append(c)
        
    cap = cv2.VideoCapture(video_path)
    tracker = CameraTracker()
    
    projected_img_pts = []
    absolute_pitch_pts = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update optical flow matrix M_{0 -> t}
        M_0_to_t = tracker.update(frame)
        
        # If the user clicked points on this frame, project them back to frame 0
        if frame_idx in clicks_by_frame:
            # M_0_to_t maps from 0 -> t. To map t -> 0, we need the inverse.
            M_t_to_0 = np.linalg.inv(M_0_to_t)
            
            for click in clicks_by_frame[frame_idx]:
                pt_t = np.array([click['img_x'], click['img_y'], 1.0])
                pt_0 = M_t_to_0 @ pt_t
                pt_0 = pt_0 / pt_0[2] # Normalize homogeneous coordinate
                
                projected_img_pts.append([pt_0[0], pt_0[1]])
                absolute_pitch_pts.append([click['pitch_x'], click['pitch_y']])
                
        if frame_idx >= max_frame:
            break # No need to process the rest of the video
            
        frame_idx += 1
        
    cap.release()
    
    # Compute Homography mapping Frame 0 -> Pitch
    src_pts = np.array(projected_img_pts, dtype=np.float32)
    dst_pts = np.array(absolute_pitch_pts, dtype=np.float32)
    
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is not None:
        np.save(output_matrix_path, H)
        print(f"Calibration successful! Matrix saved to {output_matrix_path}")
    else:
        raise RuntimeError("Homography computation failed. Points might be collinear or invalid.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--clicks", required=True, help="Path to clicks JSON file")
    parser.add_argument("--out", required=True, help="Path to save output homography_matrix.npy")
    args = parser.parse_args()
    
    run_multi_frame_calibration(args.video, args.clicks, args.out)
