import os
import subprocess
import argparse

def download_video(url: str, output_path: str, duration: int = 15):
    """
    Downloads a short clip of a YouTube video using yt-dlp.
    """
    print(f"Downloading {duration} seconds from {url}...")
    
    # We use yt-dlp to download the video. 
    # To save time, we use ffmpeg to only download a specific segment if possible,
    # or download the best quality mp4.
    
    command = [
        "python", "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args", f"ffmpeg_i:-ss 00:00:10 -t 00:00:{duration:02d}",
        "-o", output_path,
        url
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Video successfully downloaded to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download video. Is ffmpeg installed? Error: {e}")
        # Fallback to direct full download if ffmpeg slicing fails, though we only want a short clip
        print("Attempting to download without slicing...")
        command_fallback = [
            "python", "-m", "yt_dlp",
            "-f", "best[ext=mp4]",
            "-o", output_path,
            url
        ]
        subprocess.run(command_fallback, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=sT-c-zD5g6s", help="YouTube URL to download")
    parser.add_argument("--out", type=str, default="data/sample_tactical.mp4", help="Output path")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds to extract")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    download_video(args.url, args.out, args.duration)
