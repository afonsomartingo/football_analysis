import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        # Dimensions of the sports court in real-world units (meters)
        court_width = 68  # Court width
        court_length = 23.32  # Court length

        # These are the four pixel coordinates (vertices) in the image where the court is visible.
        # They represent the corners of the court in the image.
        self.pixel_vertices = np.array([
            [110, 1035],  # Bottom-left corner
            [265, 275],   # Top-left corner
            [910, 260],   # Top-right corner
            [1640, 915]   # Bottom-right corner
        ])

        # These are the corresponding real-world coordinates of the court (in meters).
        # They represent the actual dimensions of the court.
        self.target_vertices = np.array([
            [0, court_width],            # Bottom-left corner
            [0, 0],                      # Top-left corner
            [court_length, 0],           # Top-right corner
            [court_length, court_width]  # Bottom-right corner
        ])

        # Convert both the pixel and target coordinates to 32-bit floats for OpenCV operations.
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Compute the perspective transformation matrix that maps pixel coordinates to real-world coordinates.
        # This matrix will be used to transform points from the image view to the top-down view.
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        # Convert the point to integer type and check if it lies inside the polygon defined by the pixel vertices.
        p = (int(point[0]), int(point[1]))  # Ensure the point is in integer format.
        
        # Check if the point lies within the court boundaries using OpenCV's pointPolygonTest.
        # The method returns a positive value if inside, 0 if on the edge, and negative if outside.
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0

        if not is_inside:
            # If the point is outside the court boundaries, return None (invalid point).
            return None

        # Reshape the point to the format expected by the perspective transformation function.
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)

        # Apply the perspective transformation to convert the point from pixel coordinates to real-world coordinates.
        tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)

        # Reshape the result back to a 2D point and return it.
        return tranform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        # This method adds the transformed (real-world) positions to the tracked objects.
        # It loops through all tracked objects and their positions in each frame.

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # Retrieve the 'position_adjusted' field for the current object in the current frame.
                    position = track_info['position_adjusted']

                    # Convert the position to a NumPy array.
                    position = np.array(position)

                    # Transform the position from pixel coordinates to real-world coordinates.
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        # If the transformed position is valid, squeeze it to remove unnecessary dimensions
                        # and convert it back to a list for easier processing.
                        position_transformed = position_transformed.squeeze().tolist()

                    # Save the transformed position back into the tracks dictionary under the current object and frame.
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
