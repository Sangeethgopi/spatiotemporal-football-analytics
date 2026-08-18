import cv2
import numpy as np

class CameraTracker:
    def __init__(self):
        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=200,
                                   qualityLevel=0.01,
                                   minDistance=30,
                                   blockSize=3)

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.old_gray = None
        self.p0 = None
        
        # Cumulative Affine Transformation Matrix (Starts as Identity)
        self.cumulative_transform = np.eye(3, dtype=np.float32)

    def _get_green_mask(self, frame):
        """Creates a mask that primarily focuses on the pitch (green/white) to avoid tracking the crowd."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broad green mask
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Dilate heavily to include the white lines inside the pitch
        kernel = np.ones((25, 25), np.uint8)
        mask = cv2.dilate(mask_green, kernel, iterations=2)
        return mask

    def update(self, frame):
        """
        Updates the camera tracking with a new frame.
        Returns the accumulated Affine Transformation Matrix (3x3) 
        that maps Frame 0 coordinates to the Current Frame.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.old_gray is None:
            # First frame initialization
            self.old_gray = frame_gray
            mask = self._get_green_mask(frame)
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            return self.cumulative_transform

        if self.p0 is None or len(self.p0) < 10:
            # Need to re-detect features if we lost too many
            mask = self._get_green_mask(frame)
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=mask, **self.feature_params)
            if self.p0 is None:
                self.old_gray = frame_gray
                return self.cumulative_transform

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            if len(good_new) >= 4:
                # Estimate the affine transformation between old and new frame
                # Using RANSAC to ignore players running
                transform_matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                
                if transform_matrix is not None:
                    # Convert 2x3 affine to 3x3 homography-style matrix
                    M = np.eye(3, dtype=np.float32)
                    M[0:2, :] = transform_matrix
                    
                    # Accumulate the transformation
                    self.cumulative_transform = M @ self.cumulative_transform
                    
            # Update previous points and frame for next iteration
            self.p0 = good_new.reshape(-1, 1, 2)
            
            # Periodically refresh features to prevent drift tracking dead pixels
            if len(self.p0) < 100:
                mask = self._get_green_mask(frame)
                new_p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                if new_p0 is not None:
                    self.p0 = new_p0
                    
        self.old_gray = frame_gray
        return self.cumulative_transform
