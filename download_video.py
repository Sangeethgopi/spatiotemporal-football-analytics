import yt_dlp
import cv2
import os

print("Fetching stream URL...")
ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'}
# Actually OpenCV just wants the video stream, best[ext=mp4] might give 720p or 360p.
# Let's get best video that has format 720p or 1080p.
ydl_opts = {'format': 'best[height<=1080][ext=mp4]'}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info('https://www.youtube.com/watch?v=fg3DmFxMWZ4', download=False)
    video_url = info_dict.get("url", None)

print(f"URL Found. Opening video stream...")
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error opening video stream.")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 0 or fps != fps:
    fps = 25.0

print(f"Video props: {width}x{height} @ {fps}fps")

out_path = 'data/sample_tactical_wide.mp4'
# We want 5 minutes
frames_to_capture = int(5 * 60 * fps)
print(f"Capturing {frames_to_capture} frames to {out_path}...")

out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Fast forward a bit (e.g. skip first 5 minutes of intro)
# 5 mins = 5 * 60 * fps frames
skip_frames = int(5 * 60 * fps)
print(f"Skipping first {skip_frames} frames (match intro)...")
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

count = 0
while count < frames_to_capture:
    ret, frame = cap.read()
    if not ret: 
        print("End of stream reached early.")
        break
    out.write(frame)
    count += 1
    if count % 1000 == 0: 
        print(f"{count}/{frames_to_capture} frames written")

cap.release()
out.release()
print("Download complete.")
