import sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/best_parking_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_parking(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    free_probability = float(model.predict(array, verbose=0)[0][0])

    if free_probability >= 0.5:
        label = "FREE"
        confidence = free_probability
    else:
        label = "BUSY"
        confidence = 1.0 - free_probability

    print("Prediction:", label)
    print(f"Model confidence: {confidence * 100:.2f}%")
    return label, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/parking_image.jpg")
        sys.exit(1)
    predict_parking(sys.argv[1])
