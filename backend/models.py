from __future__ import division
from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import time
import os
import pandas as pd
from tensorflow.keras.models import load_model
import torch
import timm
from torchvision import transforms
import torch.nn as nn
import math
from numba import float64, boolean
from numba.experimental import jitclass
import requests
import time
import os

# ==================== Main Dehazing Function ====================
# ==================== Constants ====================
class Constants:
    patch_size = 7
    sigma_scale = 1
    sigma_default = 1
    learning_rate = 0.1
    epochs = 10
const = Constants()

# ==================== threshold ====================
class threshold:
    support = 0.4
    angle = math.pi / 12
    unimodal = 0.07 * 2  # 0.07 in Fattal
    intersection = 0.1
    shading = 0.02 / 2  # 0.02 in Fattal
    min_transmission = 0.4
    max_transmission = 0.99

thresholds = threshold()

# ==================== Core Classes ====================
class ChannelValue:
    def __init__(self):
        self.val = -1.0
        self.intensity = -1.0

spec = [
    ('point', float64[:]),
    ('direction', float64[:]),
    ('patch', float64[:, :, :]),
    ('transmission', float64),
    ('support_matrix', boolean[:, :])
]

@jitclass(spec)
class ColorLine:
    def __init__(self, point, direction, patch, support_matrix):
        self.point = point
        self.direction = direction
        self.patch = patch
        self.transmission = 0.0
        self.support_matrix = support_matrix
        self.direction_sign()

    def direction_sign(self):
        """ Change sign of the direction vector when it's negative. """

        for elem in self.direction:
            if elem < 0:
                self.direction = -self.direction

    def valid(self, airlight):
        """ Returns True when a color-line passes all quality tests. """
        if not np.any(self.direction):
            return False

        self.calculate_transmission(airlight)
        passed_all_tests = (self.significant_line_support()
                            and self.positive_reflectance()
                            and self.large_intersection_angle(airlight)
                            and self.unimodality()
                            and self.close_intersection(airlight)
                            and self.valid_transmission()
                            and self.sufficient_shading_variability())

        return passed_all_tests

    def significant_line_support(self):
        """ Test whether enough points support a color-line. """
        total_votes = self.support_matrix.size
        threshold = thresholds.support * total_votes

        if self.support_matrix.sum() < threshold:
            return False
        else:
            return True

    def positive_reflectance(self):
        """ Ensure the color-line doesn't have mixed signs in its direction vector. """
        for elem in self.direction:
            if elem < 0:
                return False
        return True

    def large_intersection_angle(self, airlight):
        """ Ensure the angle between the color-line orientation and
            atmospheric light vector is large enough.
        """
        angle = self.angle(airlight, self.direction)
        angle = abs(angle)
        return angle > thresholds.angle

    def unimodality(self):
        """ Ensure the support points projected onto the color-line
            follow a unimodal distribution.
        """
        a, b = self.normalize_coefficients()

        total_score = 0

        for idy, row in enumerate(self.patch):
            for idx, pixel in enumerate(row):
                if self.support_matrix[idy][idx]:
                    direction = pixel - self.point
                    product = np.dot(direction, self.direction)
                    score = math.cos(a * (product + b))
                    total_score += score

        total_score = total_score / np.sum(self.support_matrix)
        total_score = abs(total_score)
        return total_score < thresholds.unimodal

    def normalize_coefficients(self):
        """ Returns the variables a and b needed for normalizing
            the distribution when checking for unimodality.
        """
        max_product = -np.inf
        min_product = np.inf

        for idy, row in enumerate(self.patch):
            for idx, pixel in enumerate(row):
                if self.support_matrix[idy][idx]:
                    direction = pixel - self.point
                    product = np.dot(direction, self.direction)

                    if product < min_product:
                        min_product = product
                    if product > max_product:
                        max_product = product
        a = math.pi / (max_product - min_product)
        b = -min_product
        return a, b

    def close_intersection(self, airlight):
        """ Ensure the airlight and color-line (almost) intersect.

            Algorithm taken from http://geomalgorithms.com/a07-_distance.html,
            See https://www.youtube.com/watch?v=HC5YikQxwZA for algebraic solution.
        """
        v = airlight
        u = self.direction
        w = v
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w)
        e = np.dot(v, w)
        dd = a * c - b * b
        sc = (b * e - c * d) / dd
        tc = (a * e - b * d) / dd
        dp = w + (sc * u) - (tc * v)
        length = np.linalg.norm(dp)
        return length < thresholds.intersection

    def valid_transmission(self):
        """ Ensure the transmission falls within a valid range. """
        return 0 < self.transmission < 1

    def sufficient_shading_variability(self):
        """ Ensure there is sufficient variability in the shading. """
        samples = []
        for idy, row in enumerate(self.patch):
            for idx, pixel in enumerate(row):
                if self.support_matrix[idy][idx]:
                    direction = pixel - self.point
                    dot = np.dot(direction, self.direction)
                    samples.append(dot)

        samples = np.array(samples)
        variance = np.var(samples)
        score = np.sqrt(variance) / self.transmission

        return score > thresholds.shading

    def calculate_transmission(self, airlight):
        """ Determine the transmission, given the color-line and airlight.

            Algorithm taken from appendix in Fattal's paper.
        """
        d_unit = self.direction / np.linalg.norm(self.direction)
        a_unit = airlight / np.linalg.norm(airlight)

        ad = np.dot(a_unit, d_unit)
        dv = np.dot(d_unit, self.point)
        av = np.dot(a_unit, self.point)
        s = (-dv * ad + av) / (1 - ad * ad)
        self.transmission = 1 - s

    def sigma(self, airlight):
        """ Calculates the sigma, i.e., the color-line uncertainty."""
        similarity = np.dot(self.direction, airlight)

        result = thresholds.sigma_scale \
               * np.linalg.norm(airlight - self.direction * similarity) \
               * (1 - similarity ** 2) ** -1

        return result

# ==================== Core Algorithms ====================
def atmospheric_light(img, gray):
    top_num = int(img.shape[0] * img.shape[1] * 0.001)
    toplist = [ChannelValue() for _ in range(top_num)]
    dark_channel = np.argmin(img, axis=2)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            val = img[y, x, dark_channel[y, x]]
            intensity = gray[y, x]
            for t in toplist:
                if t.val < val or (t.val == val and t.intensity < intensity):
                    t.val = val
                    t.intensity = intensity
                    break

    max_channel = max(toplist, key=lambda x: x.intensity)
    return max_channel.intensity

def dehaze(img):
    """Integrated dehazing function"""
    img = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark_channel = np.min(img, axis=2)
    top_percent = int(img.shape[0] * img.shape[1] * 0.001)
    top_indices = np.argpartition(dark_channel.flatten(), -top_percent)[-top_percent:]
    atmospheric_light = np.max(gray.flatten()[top_indices])

    window_size = 20
    w = 0.95
    t0 = 0.55

    transmission = 1 - w * cv2.erode(dark_channel,
                                   np.ones((window_size, window_size)))
    transmission = np.clip(transmission, t0, 1.0)

    outimg = np.zeros_like(img)
    for i in range(3):
        outimg[..., i] = (img[..., i] - atmospheric_light) / np.maximum(transmission, t0) + atmospheric_light

    return (np.clip(outimg, 0, 1) * 255).astype(np.uint8)

# Initialize models with proper loading
def load_models():
    # Vehicle detection
    vehicle_model = YOLO("models/yolov8l.pt")

    # License plate detection
    plate_model = YOLO("models/license_plate.pt")  # Verify path

    # Color classification
    color_model = load_model("models/car_color_model.keras", compile=False)

    # Car model classification
    car_checkpoint = torch.load("models/Car_Model_Detection.pth", map_location='cpu')
    car_model = get_model(num_classes=len(car_checkpoint['class_names']))
    car_model.load_state_dict(car_checkpoint['state_dict'])
    car_model.eval()

    # Character recognition
    char_model = YOLO("models/character_recognition.pt")

    return vehicle_model, plate_model, color_model, car_model, char_model, car_checkpoint

# MODEL ARCHITECTURE
def get_model(num_classes):
    base_model = timm.create_model("legacy_xception", pretrained=False)
    in_features = base_model.fc.in_features
    base_model.global_pool = nn.Identity()
    base_model.fc = nn.Identity()

    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(in_features, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return nn.Sequential(base_model, classifier)

# Configuration mappings
COLOR_MAP = {
    0: 'beige',
    1: 'black',
    2: 'blue',
    3: 'green',
    4: 'grey',
    5: 'red',
    6: 'white',
    7: 'yellow'
}

CAR_MODELS = [
    "Changan Alsvin","Changan CS95","Chevrolet Captiva",
    "Chevrolet Silverado","Honda Accord","Honda Odyssey",
    "Hyundai Accent","Hyundai Elantra", "Hyundai Sonata",
    "Hyundai Tucson","Hyundai ix35","Isuzu D-Max",
    "Jeep Wrangler","KIA Borrego","KIA K5",
    "Kia Cerato","Kia Sportage","Land Rover Discovery",
    "Land Rover Range Rover","Mercedes-Benz C-Class",
    "Mercedes-Benz GLK-Class","Mitsubishi Outlander",
    "Mitsubishi Pajero Sport","Nissan Altima",
    "Nissan Murano","Nissan Patrol","Nissan Qashqai",
    "Suzuki Brezza","Suzuki Dzire","Suzuki Jimny",
    "Suzuki Swift","Toyota Camry","Toyota Corolla",
    "Toyota FJ Cruiser","Toyota Fortuner","Toyota Hilux",
    "Toyota Land Cruiser","Toyota Prado","Toyota Yaris",
    "Volkswagen Tiguan"
]

CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E', 'G',
              'H', 'J', 'K', 'L', 'N', 'R', 'S', 'T', 'U', 'V', 'X', 'Z']

# Initialize models
vehicle_model, plate_model, color_model, car_model, char_model, car_checkpoint = load_models()

# Get preprocessing parameters from checkpoint
data_mean, data_std = car_checkpoint['mean_std']

# Image transformations
car_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std)
])

# Output configuration
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)
csv_path = "detections.csv"

columns = [
    "timestamp", "coordinates", "vehicle_class",
    "car_model_1", "car_model_1_accuracy",
    "car_model_2", "car_model_2_accuracy",
    "confidence",
    "color_1", "color_1_accuracy",
    "color_2", "color_2_accuracy",
    "plate_number", "plate_confidence",
    "char_confidences",
    "vehicle_image", "plate_image"
]

pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
#for camera streaming
#"rtsp://admin:Gr14_0425@192.168.1.74/Preview_01_sub"

# Initialize video capture, 1 for phone, 0 for webcam
cap = cv2.VideoCapture(1)

cap.set(3, 3840)
cap.set(4, 2160)

# Processing functions
def preprocess_color(img):
    # Dehaze first
    dehazed_img = dehaze(img)
    dehazed_img = cv2.resize(dehazed_img, (224, 224))
    return np.expand_dims(dehazed_img / 255.0, axis=0)

def predict_car_model(img):
    tensor = car_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = car_model(tensor)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top2 = torch.topk(probs, 2)
    return [(CAR_MODELS[i], p.item()) for p, i in zip(top2.values, top2.indices)]

def recognize_plate(plate_img):
    if len(plate_img.shape) == 2:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)

    plate_img = cv2.resize(plate_img, (640, 640))
    results = char_model.predict(plate_img, imgsz=640, conf=0.5)

    chars = []
    char_confs = []
    for box in results[0].boxes:
        # Skip low confidence detections
        if box.conf[0] < 0.5:
            continue

        # Get character coordinates and label
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        char = CHARACTERS[int(box.cls[0].item())]

        # Skip non-alphanumeric characters
        if not char.isalnum():
            continue

        # Skip small detections
        if (x2 - x1) < 15 or (y2 - y1) < 15:
            continue

        chars.append((x1, char))
        char_confs.append(float(box.conf[0]))

    # Sort characters by x-coordinate and calculate overall confidence
    chars.sort(key=lambda x: x[0])
    plate_text = "".join([c[1] for c in chars])
    overall_conf = np.mean(char_confs) if char_confs else 0.0

    return plate_text, overall_conf, char_confs
  
FASTAPI_URL_UPLOAD = "http://64.227.128.31:8000/upload_images"

def send_images_to_backend(vehicle_path, plate_path=None):
    """
    Sends vehicle and plate images to the FastAPI backend.
    """
    try:
        files = []
        if os.path.exists(vehicle_path):
            files.append(("files", (os.path.basename(vehicle_path), open(vehicle_path, "rb"))))
        if plate_path and os.path.exists(plate_path):
            files.append(("files", (os.path.basename(plate_path), open(plate_path, "rb"))))
        
        if files:
            response = requests.post(FASTAPI_URL_UPLOAD, files=files)
            if response.status_code == 200:
                print("Images successfully uploaded to backend!")
            else:
                print(f"Failed to upload images. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"Error uploading images to backend: {str(e)}")

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (1920, 1080))
    # Detect only cars (class 2), buses (5), trucks (7) in COCO dataset
    vehicle_results = vehicle_model.predict(
        frame,
        conf=0.5,
        classes=[2, 5, 7],  # COCO class IDs for car/bus/truck
        verbose=False
    )

    for result in vehicle_results:
        for box in result.boxes:
            # Skip if confidence < 50%
            if box.conf[0] < 0.5:
                continue

            cls_id = int(box.cls[0])
            class_name = vehicle_model.names[cls_id]

            # Double-check class names
            if class_name not in ["car", "bus", "truck"]:
                continue

            # Vehicle processing
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            vehicle_path = f"{output_folder}/{timestamp}_vehicle.jpg"
            cv2.imwrite(vehicle_path, vehicle_crop)

            # Color prediction (top 2)
            color_input = preprocess_color(vehicle_crop)
            color_probs = color_model.predict(color_input)[0]
            top2_colors = np.argsort(color_probs)[::-1][:2]
            colors = []
            for idx in top2_colors:
                try:
                    colors.append((COLOR_MAP[idx], float(color_probs[idx])))
                except KeyError:
                    colors.append((f"unknown_{idx}", float(color_probs[idx])))

            # Car model prediction (top 2)
            model_preds = predict_car_model(vehicle_crop)

            # License plate processing
            plate_number = "N/A"
            plate_conf = 0.0
            overall_char_conf = 0.0
            char_confs = []
            plate_path = "N/A"
            plate_results = plate_model(vehicle_crop, conf=0.5)

            if len(plate_results) > 0:
                plate_result = plate_results[0]
                for plate in plate_result.boxes:
                    if plate.conf[0] < 0.5:
                        continue

                    px1, py1, px2, py2 = map(int, plate.xyxy[0])
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        continue

                    plate_path = f"{output_folder}/{timestamp}_plate.jpg"
                    cv2.imwrite(plate_path, plate_crop)
                  
                    
                    plate_conf = float(plate.conf[0])
                    plate_number, overall_char_conf, char_confs = recognize_plate(plate_crop)
                    break

            # Upload images to the backend
            send_images_to_backend(vehicle_path, plate_path if plate_path != "N/A" else None)

            # Prepare data record
            data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "coordinates": f"({x1},{y1})-({x2},{y2})",
                "vehicle_class": vehicle_model.names[int(box.cls[0])],
                "car_model_1": model_preds[0][0],
                "car_model_1_accuracy": model_preds[0][1],
                "car_model_2": model_preds[1][0],
                "car_model_2_accuracy": model_preds[1][1],
                "confidence": float(box.conf[0]),
                "color_1": colors[0][0],
                "color_1_accuracy": colors[0][1],
                "color_2": colors[1][0],
                "color_2_accuracy": colors[1][1],
                "plate_number": plate_number,
                "plate_confidence": plate_conf,
                "overall_char_confidence": overall_char_conf,
                "char_confidences": str(char_confs),
                "vehicle_image": vehicle_path,
                "plate_image": plate_path
            }
          
          
            FASTAPI_URL = "http://64.227.128.31:8000/detections"
          
            def send_to_backend(data):  
                """
                Sends detection data to the FastAPI backend.
                """
                try:
                    response = requests.post(FASTAPI_URL, json=data)
                    if response.status_code == 201:
                        print("Data successfully sent to backend!")
                    else:
                        print(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
                except Exception as e:
                    print(f"Error sending data to backend: {str(e)}")

            
            base_url = "http://64.227.128.31/static-detections/"  # Updated to match the renamed static mount

            # Hardcoded Google Maps link
            google_maps_link = "https://maps.app.goo.gl/T7GdEXA2BUMW9PYr7"
         
            # Prepare data to match DetectionData class
            detection_data = {
                "plate_number": plate_number,
                "color": colors[0][0],
                "model": model_preds[0][0],  # Add this field
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "location": google_maps_link,  # Include the GPS coordinates here
                "Car_Image": f"{base_url}{os.path.basename(vehicle_path)}",  # Full URL for vehicle image
                "plate_image": f"{base_url}{os.path.basename(plate_path)}"  # Full URL for plate image
            }
            # Send data to FastAPI backend
            send_to_backend(detection_data)

            # Save to CSV
            pd.DataFrame([data]).to_csv(csv_path, mode='a', header=False, index=False)

            # Display
            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1))
            text = f"{data['vehicle_class']} {data['confidence']:.2f} | {data['color_1']} | {plate_number}"
            cvzone.putTextRect(frame, text, (x1, max(35, y1)), scale=0.8, thickness=1)

    cv2.imshow("Vehicle Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

