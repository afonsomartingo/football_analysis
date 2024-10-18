from ultralytics import YOLO
import supervision as sv 
import pickle
import os
import sys
import cv2

sys.path.append('../')
from utils import get_bbox_width, get_center_off_bbox

class Tracker:
    def __init__ (self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self,frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1) # conf is the confidence threshold for the detection
            detections += detections_batch
        return detections
    
    def get_object_tracker(self,frames, read_from_stub=False, stub_path=None):
         
        if read_from_stub and stub_path is not None and os.path.exists(stub_path): # read tracks from stub. Stub is a file that contains the tracks
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
            return tracks
            
        detections = self.detect_frames(frames)

        tracks ={
            "players": [], # List of dictionaries where each dictionary contains the track id and bbox of a player
            "referees": [], 
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Convert Goalkeeper to Player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # Track objects
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:                # save tracks to stub
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)  

        return tracks
    
    def draw_ellipse(self,frame, bbox, color, track_id):
        y2 = int(bbox[3])

        x_center, _ = get_center_off_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, 
            (x_center, y2), 
            axes = (int(width), 
            int(0.35*width)), 
            angle = 0.0, 
            startAngle=45, 
            endAngle=235, 
            color=color, 
            thickness=2,
            lineType=cv2.LINE_4)
        return frame

    def draw_annotations(self, video_frames,tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            # Draw players

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)
            
            output_video_frames.append(frame)
        return output_video_frames
            