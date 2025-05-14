import psutil
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO('models/yolov8l.pt')

# Check memory usage
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024 ** 2  # Convert bytes to MB
print(f"Memory usage: {memory_usage:.2f} MB")