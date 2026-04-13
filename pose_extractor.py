"""
Pose Extractor Module
Extracts body landmarks from video frames using MediaPipe BlazePose.
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseExtractor:
    """Extracts body keypoints from video frames using MediaPipe BlazePose."""
    # MediaPipe landmark indices 
    LEFT_SHOULDER  = 11 #second priority
    RIGHT_SHOULDER = 12 #second priority
    LEFT_HIP       = 23 #main priority
    RIGHT_HIP      = 24 #main priority
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28

    def __init__(self, model_complexity: int = 2):
        """
        Initialise MediaPipe Pose estimator.

        Args:
            model_complexity: 0=Lite (~5 ms), 1=Full (~15 ms), 2=Heavy (~30 ms).
                              Higher = more accurate but slower.
        """
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,        # Video tracking mode
            model_complexity=model_complexity,
            smooth_landmarks=True,          # Temporal smoothing
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Stores the most recent MediaPipe result for drawing
        self.pose_landmarks = None

    # ------------------------------------------------------------------
    # Core frame processing
    # ------------------------------------------------------------------
    #extracting pose landmarks from each frame
    def process_frame(self, frame: np.ndarray):
        """
        Extract pose landmarks from a single BGR video frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            landmarks : MediaPipe landmark list (33 items), or None if undetected.
            confidence: Mean visibility of the four key landmarks (0–1).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks is None:
            self.pose_landmarks = None
            return None, 0.0

        # Cache full landmark object for later drawing calls
        self.pose_landmarks = results.pose_landmarks
        landmarks = results.pose_landmarks.landmark

        key_indices = [
            self.LEFT_HIP, self.RIGHT_HIP,
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
        ]
        confidence = float(np.mean([landmarks[i].visibility for i in key_indices]))

        return landmarks, confidence

    # ------------------------------------------------------------------
    # Calculate landmark centres
    # ------------------------------------------------------------------

    def get_hip_center(
        self,
        landmarks,
        frame_width: int,
        frame_height: int,
    ):
        """
        Midpoint between the left and right hips in pixel coordinates.

        The hip centre closely approximates barbell position during a back squat

        Returns:
            position  : (x, y) numpy array in pixels.
            confidence: Mean hip landmark visibility.
        """
        lh = landmarks[self.LEFT_HIP]
        rh = landmarks[self.RIGHT_HIP]

        x = (lh.x + rh.x) / 2 * frame_width
        y = (lh.y + rh.y) / 2 * frame_height
        confidence = (lh.visibility + rh.visibility) / 2

        return np.array([x, y]), float(confidence)

    def get_shoulder_center(
        self,
        landmarks,
        frame_width: int,
        frame_height: int,
    ):
        """
        Midpoint between the left and right shoulders in pixel coordinates.

        Returns:
            position  : (x, y) numpy array in pixels.
            confidence: Mean shoulder landmark visibility.
        """
        ls = landmarks[self.LEFT_SHOULDER]
        rs = landmarks[self.RIGHT_SHOULDER]

        x = (ls.x + rs.x) / 2 * frame_width
        y = (ls.y + rs.y) / 2 * frame_height
        confidence = (ls.visibility + rs.visibility) / 2

        return np.array([x, y]), float(confidence)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw_pose(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the pose skeleton onto *frame* using the most recent detection.

        Args:
            frame: BGR frame (modified in-place).

        Returns:
            The same frame with landmarks drawn.
        """
        if self.pose_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                frame,
                self.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
        return frame

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
