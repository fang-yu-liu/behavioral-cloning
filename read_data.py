import csv
import cv2
import numpy as np
import pickle

with open('./data/driving_log.csv', 'rt', encoding='utf8') as csvfile:
  reader = csv.DictReader(csvfile)
  file_map = {}
  for row in reader:
    center_image_filename = row['center']
    steering = row['steering']
    file_map[center_image_filename] = steering

images = []
measurements = []
for key, value in file_map.items():
  current_path = './data/' + key
  bgr_image = cv2.imread(current_path)
  image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
  images.append(image)
  measurement = float(value)
  measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# Save to pickle file
data = {'features': X_train, 'labels': y_train}

training_data_file = './train.p'

with open(training_data_file, mode='wb') as f:
  pickle.dump(data, f)
