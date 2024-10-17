from ultralytics import YOLO

model = YOLO('yolov8x.pt',device='cuda')  # Load model

results = model.predict('input_videos/08fd33_4.mp4',save=True) # Inference on video, save result to 'runs/detect/exp'
print(results[0])  # Print results
print('=============================')
for box in results.boxes[0]:
    print(box)  # Print results individually
