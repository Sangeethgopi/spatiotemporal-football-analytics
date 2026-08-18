import cv2
import numpy as np

def detect_pitch_homography(frame, prev_H=None):
    """
    Attempts to automatically detect pitch lines (halfway line, center circle)
    and compute a dynamic Homography matrix for the current frame.
    Returns the Homography matrix H, or prev_H if detection fails.
    """
    h, w = frame.shape[:2]
    
    # 1. Color Masking (Isolate White Lines on Green Grass)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Green grass mask to ignore crowd/stadium
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # White line mask
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine masks (White pixels that are near Green pixels)
    kernel = np.ones((15,15), np.uint8)
    dilated_green = cv2.dilate(mask_green, kernel, iterations=2)
    pitch_lines = cv2.bitwise_and(mask_white, dilated_green)
    
    # 2. Edge Detection
    edges = cv2.Canny(pitch_lines, 50, 150, apertureSize=3)
    
    # 3. Line Detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
    
    if lines is None:
        return prev_H
        
    vertical_lines = []
    horizontal_lines = []
    
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        
        # Vertical lines (Halfway line) usually appear between 60 and 120 degrees due to perspective
        if 60 < angle < 120:
            vertical_lines.append((x1, y1, x2, y2))
        # Horizontal lines (Touchlines) usually appear < 30 or > 150 degrees
        elif angle < 30 or angle > 150:
            horizontal_lines.append((x1, y1, x2, y2))
            
    # 4. Center Circle Detection
    circles = cv2.HoughCircles(pitch_lines, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=50, maxRadius=300)
                               
    # If we don't have enough geometric features, fallback to previous frame's matrix
    # (Prevents jitter or complete tracking loss when camera pans away from center)
    if not vertical_lines or len(horizontal_lines) < 2 or circles is None:
        return prev_H
        
    # Heuristics to find the exact intersections
    # Average the vertical lines to find the "Main Halfway Line" equation
    vx_sum = 0
    vy_sum = 0
    for x1, y1, x2, y2 in vertical_lines:
        vx_sum += (x1 + x2) / 2
        vy_sum += (y1 + y2) / 2
    halfway_x = int(vx_sum / len(vertical_lines))
    
    # Circle Center
    cx, cy, r = np.round(circles[0][0]).astype(int)
    
    # Define source points from video frame (assuming Center Circle + Halfway Line view)
    # 1. Top of center circle intersection with halfway line
    # 2. Bottom of center circle intersection with halfway line
    # 3. Center of circle
    # 4. A point on the halfway line further down
    
    # To map to 105x68m pitch:
    # Center spot = (52.5, 34)
    # Top center circle = (52.5, 34 - 9.15)
    # Bottom center circle = (52.5, 34 + 9.15)
    
    # We need 4 points. 
    # Image Points:
    pt_center = [cx, cy]
    pt_top_circle = [cx, cy - r]
    pt_bot_circle = [cx, cy + r]
    # Estimate a touchline intersection for the 4th point to ensure non-collinearity
    # (Since top, bot, center are collinear)
    # Let's use the left-most and right-most points of the circle!
    pt_left_circle = [cx - r, cy]
    pt_right_circle = [cx + r, cy]
    
    pts_src = np.array([
        pt_center,
        pt_top_circle,
        pt_bot_circle,
        pt_left_circle,
        pt_right_circle
    ], dtype=float)
    
    # Real World Points (Meters)
    pts_dst = np.array([
        [52.5, 34.0],              # Center
        [52.5, 34.0 - 9.15],       # Top Circle
        [52.5, 34.0 + 9.15],       # Bottom Circle
        [52.5 - 9.15, 34.0],       # Left Circle
        [52.5 + 9.15, 34.0]        # Right Circle
    ], dtype=float)
    
    # Compute Homography
    H, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    
    # Smooth with previous H (EMA) to prevent jitter
    if prev_H is not None and H is not None:
        H = cv2.addWeighted(H, 0.2, prev_H, 0.8, 0)
        
    return H if H is not None else prev_H
