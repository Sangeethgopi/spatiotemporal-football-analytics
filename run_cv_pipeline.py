"""
run_cv_pipeline.py  –  End-to-end CV pipeline for the full video.

Stages:
  1. YOLOv8 + ByteTrack detection & tracking  →  raw tracking CSV
  2. Jersey-colour K-Means team classification →  team-classified CSV
  3. Auto-homography (pitch line detection)    →  2D pitch-coordinate CSV
  4. Metrica-format export                     →  cv_home.csv / cv_away.csv
"""
import os
import sys
import time
import cv2
import numpy as np

# ----- paths ------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA       = os.path.join(ROOT, "data")
VIDEO      = os.path.join(DATA, "sample_tactical_wide.mp4")
RAW_CSV    = os.path.join(DATA, "sample_tactical_wide_tracking.csv")
TEAMS_CSV  = os.path.join(DATA, "sample_tracked_teams.csv")
HOMO_NPY   = os.path.join(DATA, "homography_matrix.npy")
FINAL_CSV  = os.path.join(DATA, "sample_final_2d.csv")
HOME_CSV   = os.path.join(DATA, "cv_home.csv")
AWAY_CSV   = os.path.join(DATA, "cv_away.csv")

sys.path.insert(0, ROOT)

# ==============================================================================
# Stage 0 – Validate
# ==============================================================================
def stage0_validate():
    print("=" * 60)
    print("STAGE 0 – Validate inputs")
    print("=" * 60)
    if not os.path.exists(VIDEO):
        raise FileNotFoundError(f"Video not found: {VIDEO}")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  Video : {VIDEO}")
    print(f"  Size  : {w}x{h} @ {fps} fps, {n} frames ({n/fps:.1f}s)")
    return fps, n

# ==============================================================================
# Stage 1 – YOLOv8 + ByteTrack
# ==============================================================================
def stage1_track():
    print("\n" + "=" * 60)
    print("STAGE 1 – YOLOv8 detection + ByteTrack tracking")
    print("=" * 60)
    if os.path.exists(RAW_CSV):
        print(f"  [SKIP] {RAW_CSV} already exists. Delete it to re-run.")
        return
    from src.cv_pipeline.tracker import run_tracking
    t0 = time.time()
    run_tracking(VIDEO, VIDEO)   # output_path is used to derive csv name
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  →  {RAW_CSV}")

# ==============================================================================
# Stage 2 – Team classification
# ==============================================================================
def stage2_classify():
    print("\n" + "=" * 60)
    print("STAGE 2 – Jersey colour K-Means team classification")
    print("=" * 60)
    if os.path.exists(TEAMS_CSV):
        print(f"  [SKIP] {TEAMS_CSV} already exists. Delete it to re-run.")
        return
    from src.cv_pipeline.team_classifier import classify_teams
    t0 = time.time()
    classify_teams(VIDEO, RAW_CSV, TEAMS_CSV)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  →  {TEAMS_CSV}")

# ==============================================================================
# Stage 3 – Homography  (auto-detect from pitch lines on 1st suitable frame)
# ==============================================================================
def stage3_homography():
    print("\n" + "=" * 60)
    print("STAGE 3 – Auto-homography & pixel → pitch mapping")
    print("=" * 60)

    # --- 3a: Build the homography matrix if missing ---
    if not os.path.exists(HOMO_NPY):
        print("  Auto-detecting pitch landmarks for homography...")
        from src.cv_pipeline.pitch_detector import detect_pitch_homography
        cap = cv2.VideoCapture(VIDEO)
        H = None
        frame_idx = 0
        # Try each frame until we get a valid H
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            H = detect_pitch_homography(frame, H)
            if H is not None:
                print(f"  Homography found at frame {frame_idx}")
                break
            frame_idx += 1
            if frame_idx > 500:
                break
        cap.release()

        if H is None:
            print("  [WARN] Auto-detection failed. Generating default homography.")
            # Default: assume camera is roughly centered on the pitch
            # Map the center of a 1920x1080 frame to center of 105x68 pitch
            cap2 = cv2.VideoCapture(VIDEO)
            ret, frame = cap2.read()
            cap2.release()
            h_f, w_f = frame.shape[:2]
            # Simple 4-point mapping (corners of visible pitch area)
            pts_src = np.array([
                [w_f * 0.1, h_f * 0.3],   # top-left of pitch area
                [w_f * 0.9, h_f * 0.3],   # top-right
                [w_f * 0.9, h_f * 0.85],  # bottom-right
                [w_f * 0.1, h_f * 0.85],  # bottom-left
            ], dtype=np.float32)
            pts_dst = np.array([
                [0.0, 0.0],
                [105.0, 0.0],
                [105.0, 68.0],
                [0.0, 68.0],
            ], dtype=np.float32)
            H, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

        np.save(HOMO_NPY, H)
        print(f"  Homography matrix saved  →  {HOMO_NPY}")
    else:
        print(f"  [SKIP] {HOMO_NPY} already exists.")

    # --- 3b: Apply homography to all tracked positions ---
    if os.path.exists(FINAL_CSV):
        print(f"  [SKIP] {FINAL_CSV} already exists. Delete it to re-run.")
        return

    from src.cv_pipeline.homography import apply_homography
    t0 = time.time()
    apply_homography(TEAMS_CSV, HOMO_NPY, VIDEO, FINAL_CSV)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  →  {FINAL_CSV}")

# ==============================================================================
# Stage 4 – Export to Metrica format
# ==============================================================================
def stage4_export():
    print("\n" + "=" * 60)
    print("STAGE 4 – Export to Metrica Sports format")
    print("=" * 60)
    if os.path.exists(HOME_CSV) and os.path.exists(AWAY_CSV):
        print(f"  [SKIP] Output files already exist. Delete to re-run.")
        return
    from src.cv_pipeline.to_metrica import cv_to_metrica
    t0 = time.time()
    cv_to_metrica(FINAL_CSV, HOME_CSV, AWAY_CSV)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  →  {HOME_CSV}, {AWAY_CSV}")

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    overall_start = time.time()

    fps, total_frames = stage0_validate()
    stage1_track()
    stage2_classify()
    stage3_homography()
    stage4_export()

    total = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"ALL STAGES COMPLETE in {total/60:.1f} min")
    print("=" * 60)
    print(f"  Tracking CSV   : {RAW_CSV}")
    print(f"  Team CSV       : {TEAMS_CSV}")
    print(f"  Pitch 2D CSV   : {FINAL_CSV}")
    print(f"  Metrica Home   : {HOME_CSV}")
    print(f"  Metrica Away   : {AWAY_CSV}")
