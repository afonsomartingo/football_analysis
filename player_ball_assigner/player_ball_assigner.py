import sys
sys.path.append('../')  # Adds the parent directory to the Python path for importing custom modules.
from utils import get_center_off_bbox, measure_distance  # Utility functions to get the center of a bounding box and measure distances.

class PlayerBallAssigner:
    def __init__(self):
        # Initialize the maximum allowable distance for assigning the ball to a player.
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        # Calculate the center position of the ball using the bounding box.
        ball_postion = get_center_off_bbox(ball_bbox)

        # Initialize variables to track the closest player to the ball.
        minimum_distance = 99999  # Set an arbitrarily large distance initially.
        assigned_player = -1  # Default player assignment to -1 (no player assigned).

        # Iterate over the players and calculate their distance to the ball.
        for player_id, player in players.items():
            player_bbox = player["bbox"]  # Get the player's bounding box.

            # Measure distances from the left and right corners of the player's bounding box to the ball.
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_postion) 
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_postion)

            # Use the smaller of the two distances to determine proximity to the ball.
            distance = min(distance_left, distance_right)

            # Check if the player is within the allowable distance to be assigned to the ball.
            if distance < self.max_player_ball_distance:
                # If the player is closer than the previously found player, update the assignment.
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id  # Assign this player as the one closest to the ball.

        # Return the ID of the assigned player (or -1 if no player is within the threshold distance).
        return assigned_player
