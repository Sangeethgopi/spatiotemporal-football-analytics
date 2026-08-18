import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
import os

def classify_teams(video_path: str, tracking_csv: str, output_csv: str):
    """
    Extracts jersey colors from bounding boxes and uses K-Means to cluster 
    players into Home, Away, and Referee groups.
    """
    print(f"Loading tracking data from {tracking_csv}...")
    df = pd.read_csv(tracking_csv)
    
    # We only want to classify players, not the ball
    player_df = df[df["class"] == "player"].copy()
    
    cap = cv2.VideoCapture(video_path)
    
    # Store dominant color for each track_id
    track_colors = {}
    
    # Sample every N frames to speed up processing
    sample_rate = 5
    
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_rate != 0:
            continue
            
        # Get players in this frame
        current_players = player_df[player_df["frame"] == frame_idx]
        
        for _, row in current_players.iterrows():
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = int(row["bbox_x1"]), int(row["bbox_bbox_y1"] if "bbox_bbox_y1" in row else row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])
            
            # Extract bounding box crop
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2]
            
            # To get jersey color, we extract the top half of the player
            # to avoid shorts/socks which might be the same color across teams
            jersey_crop = crop[0:int(crop.shape[0]/2), :]
            
            if jersey_crop.size == 0:
                continue
                
            # Convert to HSV which is better for color clustering
            hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
            
            # Get mean color
            mean_color = np.mean(hsv, axis=(0, 1))
            
            if track_id not in track_colors:
                track_colors[track_id] = []
            track_colors[track_id].append(mean_color)
            
    cap.release()
    
    # Average the colors over all sampled frames for each track_id
    final_colors = []
    track_ids = []
    for tid, colors in track_colors.items():
        if colors:
            avg_color = np.mean(colors, axis=0)
            final_colors.append(avg_color)
            track_ids.append(tid)
            
    if not final_colors:
        print("No players found to cluster.")
        return
        
    X = np.array(final_colors)
    
    print("Clustering into 3 groups (Home, Away, Referee/Goalkeepers)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Create a mapping from track_id to team
    team_mapping = dict(zip(track_ids, labels))
    
    # Apply mapping to the dataframe
    df["team_cluster"] = df["track_id"].map(team_mapping)
    
    # Save the updated tracking data
    df.to_csv(output_csv, index=False)
    print(f"Team classification complete! Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/sample_tactical.mp4")
    parser.add_argument("--tracking", type=str, default="data/sample_tracked_tracking.csv")
    parser.add_argument("--out", type=str, default="data/sample_tracked_teams.csv")
    args = parser.parse_args()
    
    classify_teams(args.video, args.tracking, args.out)
