import pandas as pd
import numpy as np
import cv2
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cv_pipeline.camera_tracker import CameraTracker

def apply_homography(tracking_csv: str, matrix_path: str, video_path: str, output_csv: str):
    """
    Applies a Hybrid Dynamic Homography transformation.
    Uses the fixed base matrix (interactive calibration) and dynamically 
    shifts it using Optical Flow (Camera Tracker) to handle panning.
    """
    print(f"Loading team-classified tracking data from {tracking_csv}...")
    df = pd.read_csv(tracking_csv)
    
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix file not found at {matrix_path}. Run interactive_calibration.py first!")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}.")
        
    H_base = np.load(matrix_path)
    print("Loaded Interactive Base Homography Matrix:\n", H_base)
    
    # Initialize Camera Tracker
    tracker = CameraTracker()
    cap = cv2.VideoCapture(video_path)
    
    # Group tracking data by frame
    frames_data = df.groupby('frame')
    
    transformed_x = []
    transformed_y = []
    
    # Pre-allocate output arrays
    df['pitch_x'] = np.nan
    df['pitch_y'] = np.nan
    
    current_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing optical flow across {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Get the Camera Motion Matrix M (maps Frame 0 pixels -> Current Frame pixels)
        M = tracker.update(frame)
        
        # 2. Invert M to map Current Frame pixels back to Frame 0 pixels
        M_inv = np.linalg.inv(M)
        
        # 3. Compute dynamic homography for this frame: H_base * M_inv
        H_t = H_base @ M_inv
        
        # Apply this matrix to all players detected in this frame
        if current_frame in frames_data.groups:
            frame_indices = frames_data.groups[current_frame]
            
            for idx in frame_indices:
                row = df.loc[idx]
                pt = np.array([[[row['foot_x'], row['foot_y']]]], dtype=float)
                
                pt_transformed = cv2.perspectiveTransform(pt, H_t)
                
                pitch_x = pt_transformed[0][0][0]
                pitch_y = pt_transformed[0][0][1]
                
                # Clip to pitch boundaries
                pitch_x = np.clip(pitch_x, 0, 105)
                pitch_y = np.clip(pitch_y, 0, 68)
                
                df.at[idx, 'pitch_x'] = pitch_x
                df.at[idx, 'pitch_y'] = pitch_y
                
        current_frame += 1
        
    cap.release()
    
    # Save the final tracking data
    df.to_csv(output_csv, index=False)
    print(f"Successfully transformed pixels to 2D pitch coordinates! Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", type=str, default="data/sample_tracked_teams.csv")
    parser.add_argument("--matrix", type=str, default="data/homography_matrix.npy")
    parser.add_argument("--video", type=str, default="data/sample_tactical_wide.mp4")
    parser.add_argument("--out", type=str, default="data/sample_final_2d.csv")
    args = parser.parse_args()
    
    apply_homography(args.tracking, args.matrix, args.video, args.out)
