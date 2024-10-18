import cv2
import sys 
sys.path.append('../')  # Adds the parent directory to the Python path to import custom utility functions.
from utils import measure_distance, get_foot_position  # Utility functions for measuring distances and getting foot positions from bounding boxes.

class SpeedAndDistance_Estimator:
    def __init__(self):
        # Initialize the frame window and frame rate.
        self.frame_window = 5  # Number of frames between which speed and distance will be calculated.
        self.frame_rate = 24  # Frame rate of the video (24 frames per second).

    def add_speed_and_distance_to_tracks(self, tracks):
        # Dictionary to keep track of the total distance covered by each object.
        total_distance = {}

        # Loop over each tracked object in the video.
        for object, object_tracks in tracks.items():
            # Skip "ball" and "referees" objects as they are not part of the distance/speed calculation.
            if object == "ball" or object == "referees":
                continue 

            # Get the number of frames in the track for the current object.
            number_of_frames = len(object_tracks)

            # Iterate over the frames in steps of 'frame_window' to calculate speed/distance.
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)  # Set the last frame in the current window.

                # Iterate over each track in the current frame.
                for track_id, _ in object_tracks[frame_num].items():
                    # Ensure the same object is present in both the current frame and the last frame in the window.
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Get the transformed positions (e.g., normalized or projected positions) of the object in both frames.
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # If any of the positions are invalid, skip this track.
                    if start_position is None or end_position is None:
                        continue

                    # Calculate the distance covered between the two frames.
                    distance_covered = measure_distance(start_position, end_position)

                    # Calculate the time elapsed based on the frame rate.
                    time_elapsed = (last_frame - frame_num) / self.frame_rate

                    # Calculate speed in meters per second and convert it to km/h.
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Initialize the total distance for this object and track if not already done.
                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    # Add the current distance covered to the total distance for this object.
                    total_distance[object][track_id] += distance_covered

                    # Update each frame in the current window with the calculated speed and distance.
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        # Add the calculated speed and distance to the current track information.
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        # Initialize a list to store frames with drawn speed and distance data.
        output_frames = []
        
        # Iterate over each frame.
        for frame_num, frame in enumerate(frames):
            # Iterate over all objects and their tracks in the current frame.
            for object, object_tracks in tracks.items():
                # Skip "ball" and "referees" as they are not relevant for drawing speed and distance.
                if object == "ball" or object == "referees":
                    continue 
                
                # Iterate over the track information of the current object.
                for _, track_info in object_tracks[frame_num].items():
                    # Check if the speed information is available.
                    if "speed" in track_info:
                        # Get the speed and distance values from the track information.
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)

                        # If speed or distance is missing, skip this track.
                        if speed is None or distance is None:
                            continue

                        # Get the bounding box of the object to determine where to draw the text.
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)  # Get the foot position of the object to draw the text near it.
                        position = list(position)
                        position[1] += 40  # Adjust the position for better text placement.

                        position = tuple(map(int, position))  # Convert position to integers for drawing.

                        # Draw the speed (in km/h) on the frame.
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        # Draw the total distance (in meters) on the frame.
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Add the modified frame to the output list.
            output_frames.append(frame)

        # Return the list of frames with the drawn speed and distance.
        return output_frames
