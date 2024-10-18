from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')  # Adds parent directory to Python path for custom utility imports.
from utils import get_center_off_bbox, get_bbox_width, get_foot_position  # Utility functions for bounding box operations.

class Tracker:
    def __init__(self, model_path):
        # Initialize the YOLO model for object detection and ByteTrack for object tracking.
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        # Add the position (center or foot position) to each tracked object.
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_off_bbox(bbox)  # Get the center position of the ball.
                    else:
                        position = get_foot_position(bbox)  # Get the foot position of players/referees.
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        # Interpolate missing ball positions to handle incomplete or noisy data.
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Perform linear interpolation for missing values and backfill any remaining gaps.
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Convert the interpolated DataFrame back to list format.
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        # Use YOLO to perform object detection on a batch of frames.
        batch_size = 20  # Process frames in batches of 20.
        detections = [] 
        for i in range(0, len(frames), batch_size):
            # Perform detection with a confidence threshold of 0.1.
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Retrieve object tracks either by detecting objects or reading from a saved stub (cached data).
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            # Load precomputed tracks if available.
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Perform object detection on the input frames.
        detections = self.detect_frames(frames)

        # Initialize tracking dictionaries for players, referees, and the ball.
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            # Get class names from the detection results.
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}  # Reverse lookup dictionary for class names.

            # Convert detection to the supervision format for further processing.
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert 'goalkeeper' class to 'player' class.
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track objects in the frame.
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize empty track entries for the current frame.
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                # Get bounding box and class ID of the detected object.
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Track players.
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                # Track referees.
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
            # Track the ball using its bounding box.
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save tracks to a file for reuse if a stub path is provided.
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Draw an ellipse around a player's foot to indicate their position.
        y2 = int(bbox[3])  # Bottom y-coordinate of the bounding box.
        x_center, _ = get_center_off_bbox(bbox)  # Center x-coordinate of the bounding box.
        width = get_bbox_width(bbox)  # Width of the bounding box.

        # Draw the ellipse with the calculated center and size.
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw a rectangle for displaying the track ID near the player's position.
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            # Draw a filled rectangle for the track ID.
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            # Adjust text position based on track ID size.
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            # Draw the track ID text inside the rectangle.
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        # Draw a triangle (used for indicating the ball's position).
        y = int(bbox[1])  # Top y-coordinate of the bounding box.
        x, _ = get_center_off_bbox(bbox)  # Center x-coordinate.

        # Define the triangle points.
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        # Draw the triangle with the specified color.
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle to show ball possession percentages.
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Get the cumulative ball control data up to the current frame.
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # Display the ball control percentages for both teams.
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        # Annotate video frames with tracked objects and ball control statistics.
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players on the frame.
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # Default player color is red.
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))  # Draw ball possession indicator.

            # Draw referees on the frame.
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))  # Referee color is yellow.

            # Draw the ball on the frame.
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))  # Ball color is green.

            # Draw ball control statistics on the frame.
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # Append the annotated frame to the output list.
            output_video_frames.append(frame)

        return output_video_frames
