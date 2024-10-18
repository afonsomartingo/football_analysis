from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # Reshape image in 2D array
        image_2d = image.reshape(-1, 3)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        # Extract the region of interest (ROI) from the frame using the bounding box (bbox)
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Extract the top half of the ROI
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get clustering model for the top half of the image
        kmeans = self.get_clustering_model(top_half_image)

        # Reshape the top half image to 2D array for prediction
        top_half_image_2d = top_half_image.reshape(-1, 3)

        # Get cluster labels for each pixel in the top half image
        labels = kmeans.predict(top_half_image_2d)

        # Reshape the labels to the original shape of the top half image
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Determine the non-player cluster by examining the corner pixels
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Get the color of the player cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        # Iterate over all player detections
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            # Get the color of the player
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Perform KMeans clustering on the player colors to assign team colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        # Assign team colors based on the cluster centers
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # If the player is already assigned to a team, return the team ID
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # Get the color of the player
        player_color = self.get_player_color(frame, player_bbox)

        # Predict the team ID based on the player color
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id == 82:
            team_id = 2

        # Assign the player to the team
        self.player_team_dict[player_id] = team_id

        return team_id