"""
Extended Metrica Match 2 possession debug — find actual possession flips
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from kloppy import metrica

PITCH_L = 105.0
PITCH_W = 68.0

print("Loading Metrica Match 2...")
dataset = metrica.load_open_data(match_id=2, sample_rate=1/25.0, limit=None)

rows = []
for frame in dataset.frames:
    row = {"frame": frame.frame_id, "ball_x": None, "ball_y": None,
           "home_players": [], "away_players": []}
    if frame.ball_coordinates:
        row["ball_x"] = frame.ball_coordinates.x * PITCH_L
        row["ball_y"] = frame.ball_coordinates.y * PITCH_W
    for player, coord in frame.players_coordinates.items():
        if coord is None: continue
        px = coord.x * PITCH_L; py = coord.y * PITCH_W
        if player.team.ground.value == "home":
            row["home_players"].append((px, py))
        else:
            row["away_players"].append((px, py))
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Total frames: {len(df)}, ball available: {df['ball_x'].notna().sum()}")

# Step through full possession logic with HOLD=75 and print flips
HOLD = 75
current_poss = None
consecutive = 0
poss = [None] * len(df)
flips = []

for i, row in df.iterrows():
    bx = row["ball_x"]
    if pd.isna(bx):
        poss[i] = current_poss
        continue
    hps = row["home_players"]
    aps = row["away_players"]
    if not hps or not aps:
        poss[i] = current_poss
        continue
    hd = min(math.hypot(px-bx, py-bx) for px, py in hps)
    ad = min(math.hypot(px-bx, py-bx) for px, py in aps)
    closer = "Home" if hd <= ad else "Away"
    if closer != current_poss:
        consecutive += 1
        if consecutive >= HOLD:
            prev = current_poss
            current_poss = closer
            consecutive = 0
            flips.append({"frame_idx": i, "frame": row["frame"], "from": prev, "to": closer})
    else:
        consecutive = 0
    poss[i] = current_poss

# BUG IN ORIGINAL: distance was hypot(x-bx, y-bx) instead of y-by! Let's fix
print(f"\nPossession flips (bugged): {len(flips)}")

# CORRECT version
flips_correct = []
consecutive = 0
current_poss = None
poss_correct = [None] * len(df)

for i, row in df.iterrows():
    bx = row["ball_x"]
    by = row["ball_y"]
    if pd.isna(bx) or pd.isna(by):
        poss_correct[i] = current_poss
        continue
    hps = row["home_players"]
    aps = row["away_players"]
    if not hps or not aps:
        poss_correct[i] = current_poss
        continue
    hd = min(math.hypot(px-bx, py-by) for px, py in hps)
    ad = min(math.hypot(px-bx, py-by) for px, py in aps)
    closer = "Home" if hd <= ad else "Away"
    if closer != current_poss:
        consecutive += 1
        if consecutive >= HOLD:
            prev = current_poss
            current_poss = closer
            consecutive = 0
            flips_correct.append({"frame_idx": i, "frame": row["frame"], "from": prev, "to": closer, "frame_count": i})
    else:
        consecutive = 0
    poss_correct[i] = current_poss

print(f"Possession flips (corrected): {len(flips_correct)}")
for flip in flips_correct[:10]:
    print(f"  Frame {flip['frame']}: {flip['from']} -> {flip['to']}")

# Check team hold durations
poss_arr = np.array(poss_correct)
print(f"\nPossession breakdown:")
print(f"  Home frames: {np.sum(poss_arr == 'Home')}")
print(f"  Away frames: {np.sum(poss_arr == 'Away')}")
print(f"  None frames: {np.sum(poss_arr == None)}")

# How many hold >= 75?
# Count spell lengths
current = poss_arr[0]
spell_len = 1
spells = []
for i in range(1, len(poss_arr)):
    if poss_arr[i] == current:
        spell_len += 1
    else:
        if current is not None:
            spells.append(spell_len)
        current = poss_arr[i]
        spell_len = 1
spells = np.array(spells)
print(f"\nPossession spells: {len(spells)}")
print(f"  >= 75 frames: {np.sum(spells>=75)}")
print(f"  >= 38 frames: {np.sum(spells>=38)}")
print(f"  Median spell: {np.median(spells):.0f}")

# Try with HOLD=1 to see raw switches
raw_switches = 0
prev = None
for p in poss_correct:
    if p != prev and prev is not None and p is not None:
        raw_switches += 1
    prev = p
print(f"\nRaw possession switches (HOLD=1): {raw_switches}")

with open("data/results/external_dataset/match2_possession_debug.json", "w") as f:
    json.dump({
        "flips_bugged": len(flips),
        "flips_correct": len(flips_correct),
        "n_spells": int(len(spells)),
        "spells_ge75": int(np.sum(spells>=75)),
        "spells_ge38": int(np.sum(spells>=38)),
        "median_spell": float(np.median(spells)) if len(spells) else 0,
        "raw_switches_hold1": raw_switches,
        "home_frames": int(np.sum(poss_arr == "Home")),
        "away_frames": int(np.sum(poss_arr == "Away")),
    }, f, indent=4)
