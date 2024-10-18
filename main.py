from utils.video_utils import read_video, save_video
from trackers import Tracker

def main():
    # Read video
    video_path = 'input_videos/08fd33_4.mp4'
    video_frames = read_video(video_path)

    #Initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracker(video_frames,
                                        read_from_stub=True,
                                        stub_path='stubs/tracks_stubs.pkl')
    
    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(video_frames, 'output_videos/output_video.avi')


if __name__ == "__main__":
    main()
