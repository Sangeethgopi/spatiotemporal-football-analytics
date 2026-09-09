"""
Investigate why Metrica Match 2 derived zero turnovers.
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

from kloppy import metrica

PITCH_L = 105.0
PITCH_W = 68.0
HZ = 25

print("Loading Metrica Match 2...")
dataset = metrica.load_open_data(match_id=2, sample_rate=1/25.0, limit=None)

rows = []
for frame in dataset.frames:
    row = {
        "frame": frame.frame_id,
        "ball_x": None, "ball_y": None,
        "home_players": [], "away_players": [],
    }
    if frame.ball_coordinates:
        row["ball_x"] = frame.ball_coordinates.x * PITCH_L
        row["ball_y"] = frame.ball_coordinates.y * PITCH_W
    for player, coord in frame.players_coordinates.items():
        if coord is None: continue
        px = coord.x * PITCH_L
        py = coord.y * PITCH_W
        if player.team.ground.value == "home":
            row["home_players"].append((px, py))
        else:
            row["away_players"].append((px, py))
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Frames: {len(df)}")

# Check player counts per frame
df["n_home"] = df["home_players"].apply(len)
df["n_away"] = df["away_players"].apply(len)
print(f"\nPlayer counts:")
print(f"  Home: median={df['n_home'].median()}, min={df['n_home'].min()}, max={df['n_home'].max()}")
print(f"  Away: median={df['n_away'].median()}, min={df['n_away'].min()}, max={df['n_away'].max()}")
print(f"  Frames with 10+ home AND 10+ away: {((df['n_home']>=10) & (df['n_away']>=10)).sum()}")

# Check possession assignment in detail
print("\nPossession debug - first 200 frames with ball:")
ball_frames = df[df["ball_x"].notna()].head(100)

for i, row in ball_frames.head(10).iterrows():
    bx, by = row["ball_x"], row["ball_y"]
    hps = row["home_players"]
    aps = row["away_players"]
    if not hps or not aps: 
        print(f"  Frame {row['frame']}: no players")
        continue
    hd = min(math.hypot(px-bx, py-by) for px, py in hps)
    ad = min(math.hypot(px-bx, py-by) for px, py in aps)
    print(f"  Frame {row['frame']}: ball=({bx:.1f},{by:.1f}) home_min={hd:.2f}m away_min={ad:.2f}m -> {'Home' if hd<=ad else 'Away'}")

# Check if coordinates are reasonable
print("\nCoordinate sanity:")
print(f"  Ball x range: {df['ball_x'].min():.1f} to {df['ball_x'].max():.1f}")
print(f"  Ball y range: {df['ball_y'].min():.1f} to {df['ball_y'].max():.1f}")

# Check kloppy coordinate system: might be 0-1 normalized
print(f"\nRaw ball x (first 5 non-null):")
for i, row in df[df["ball_x"].notna()].head(5).iterrows():
    print(f"  {row['ball_x']:.3f}")
