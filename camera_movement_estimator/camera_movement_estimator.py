import pickle
import cv2 
import numpy as np
import sys
import os
sys.path.append('../')  # Adds the parent directory to the Python path for importing custom modules.
from utils import measure_distance, measure_xy_distance  # Utility functions for distance calculations.

class CameraMovementEstimator:
    def __init__(self, frame):
        # Initializes parameters for the Lucas-Kanade optical flow.
        self.minimum_distance = 5  # Minimum distance threshold for detecting significant movement.
        
        # Parameters for the Lucas-Kanade optical flow method.
        self.lk_params = dict(
            winSize=(15, 15),  # Size of the window for optical flow calculation.
            maxLevel=2,  # Maximum number of pyramid levels.
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria for iterative search.
        )

        # Converts the first frame to grayscale for feature detection.
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a mask to limit feature detection to specific regions of the frame.
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1  # Mask for the left side of the frame.
        mask_features[:, 900:1050] = 1  # Mask for the right side of the frame.

        # Parameters for detecting good features to track (corners in the image).
        self.features = dict(
            maxCorners=100,  # Maximum number of corners to detect.
            qualityLevel=0.3,  # Minimum quality level of corners to retain.
            minDistance=3,  # Minimum distance between detected corners.
            blockSize=7,  # Size of the neighborhood for corner detection.
            mask=mask_features  # Apply the mask to restrict detection to certain areas.
        )

    def adjust_position_to_tracks(self, tracks, camera_movement_per_frame):
        # Adjusts object positions in the tracks by compensating for camera movement.
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # Adjusts the position by subtracting the camera movement for the given frame.
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = [position[0] - camera_movement[0], position[1] - camera_movement[1]]
                    # Store the adjusted position.
                    tracks[object][frame_num][track_id]["position_adjusted"] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Optionally read precomputed camera movement data from a file (stub).
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize an array to store camera movement for each frame.
        camera_movement = [[0, 0]] * len(frames)

        # Convert the first frame to grayscale and detect initial features to track.
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Loop through the rest of the frames to estimate camera movement.
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Calculate optical flow between the previous and current frames.
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            # Initialize variables to store the maximum movement.
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # Compare the old and new features to calculate movement.
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # Measure the distance between the new and old feature points.
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    # Measure the x and y movement between the points.
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            # If the detected movement exceeds the minimum distance, record it.
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()  # Update the previous frame to the current one.

            # Optionally save the camera movement data to a file (stub).
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(camera_movement, f)

        return camera_movement  # Return the list of camera movements for each frame.

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        # Draws the calculated camera movement on each frame and returns the annotated frames.
        output_frame = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()  # Copy the frame to avoid modifying the original.

            # Create an overlay to display the camera movement information.
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)  # White rectangle for text background.
            alpha = 0.6  # Transparency for overlay.
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blend the overlay with the original frame.

            # Extract the x and y camera movement for the current frame.
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            # Display the x and y movement on the frame.
            frame = cv2.putText(frame, f'Camera Movement X: {x_movement:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f'Camera Movement Y: {y_movement:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            # Append the annotated frame to the output list.
            output_frame.append(frame)

        return output_frame  # Return the list of frames with camera movement annotations.
