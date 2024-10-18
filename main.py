from utils.video_utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # 1. Read video input from the specified path
    video_path = 'input_videos/08fd33_4.mp4'
    video_frames = read_video(video_path)  # Loads the video frames into a list

    # 2. Initialize the Tracker with a pre-trained YOLO model
    tracker = Tracker('models/best.pt')  # Model is used for object detection and tracking

    # 3. Retrieve object tracks (players, ball, referees) from video frames
    # The tracks can be read from a saved 'stub' to avoid reprocessing if available
    tracks = tracker.get_object_tracks(
        video_frames, 
        read_from_stub=True, 
        stub_path='stubs/tracks_stubs.pkl'
    )

    # 4. Add positions (such as foot position for players or center for the ball) to the tracked objects
    tracker.add_position_to_tracks(tracks)

    # 5. Initialize the camera movement estimator with the first frame of the video
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    # 6. Retrieve the camera movement for each frame, reading from a saved stub if available
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )

    # 7. Adjust the object positions according to camera movement across frames
    camera_movement_estimator.adjust_position_to_tracks(tracks, camera_movement_per_frame)

    # 8. Use the ViewTransformer to transform object positions from pixel coordinates to real-world coordinates
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    ''' 
    # Uncomment this section to save a cropped image of a player from the first frame
    for track_id, player in tracks["players"][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]

        # Crop the bounding box area from the frame and save it as an image
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        cv2.imwrite(f'output_videos/croped_img.jpg', cropped_image)
        break
    '''

    # 9. Interpolate missing ball positions between frames to ensure smoother tracking
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 10. Estimate and add speed and distance traveled for all tracked objects
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # 11. Initialize the team assigner to determine which team each player belongs to based on color
    team_assigner = TeamAssigner()

    # Assign team color based on the first frame
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    # 12. Loop through all frames to assign teams to players based on the bounding box information
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            # Get the team assignment based on the current frame and player bounding box
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)

            # Store the team and team color in the tracking data
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # 13. Assign ball possession to the nearest player for each frame
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    # Loop through each frame and determine which player has possession of the ball
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]  # Ball bounding box for the current frame
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            # Mark the assigned player as having possession of the ball
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If no player is assigned the ball, keep the previous team's ball control
            team_ball_control.append(team_ball_control[-1])

    # Convert team ball control to a NumPy array for easier handling later
    team_ball_control = np.array(team_ball_control)

    # 14. Draw the tracked objects and annotations (players, ball, teams) on the video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # 15. Draw camera movement annotations on the video frames
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # 16. Draw speed and distance information on the video frames
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # 17. Save the final video with all annotations and drawings applied
    save_video(output_video_frames, 'output_videos/output_video.avi')


# Entry point of the program
if __name__ == "__main__":
    main()
