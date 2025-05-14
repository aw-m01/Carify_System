import psutil
import torch

# Load your model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov8l.pt')

# Check memory usage
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")