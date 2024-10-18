from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt')

# Set device to 'cuda' for GPU
model.to('cuda')

# Inference on video, save result to 'runs/detect/exp'
results = model.predict('input_videos/08fd33_4.mp4', save=True)

# Print results
print(results[0])

print('=============================')
for box in results[0].boxes:
    print(box)