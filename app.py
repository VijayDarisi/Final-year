from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Pillow for image processing
import numpy as np

# === Paths ===
MODEL_PATH = "/Users/vijay/Downloads/keras wound/keras_model.h5"
LABELS_PATH = "/Users/vijay/Downloads/keras wound/labels.txt"
IMAGE_PATH = "/Users/vijay/Downloads/Wound_dataset copy/Cut/cut (5).jpg"

# === Load the model ===
model = load_model(MODEL_PATH, compile=False)

# === Load and process labels ===
with open(LABELS_PATH, "r") as f:
    class_names = {}
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            class_names[int(parts[0])] = parts[1]

# === Create input array ===
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# === Load and preprocess the image ===
image = Image.open(IMAGE_PATH).convert("RGB")
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array

# === Run prediction ===
prediction = model.predict(data)
index = int(np.argmax(prediction))
class_name = class_names.get(index, "Unknown")
confidence_score = prediction[0][index]

# === Print the results ===
print(f"\nPredicted Class: {class_name} (Index: {index})")
print(f"Confidence Score: {confidence_score * 100:.2f}%")
