from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
import requests
from flask_cors import CORS

# ==============================
# Flask Initialization
# ==============================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ Allow Expo / Web / Mobile access

# ==============================
# Model and Directory Setup
# ==============================
MODEL_PATH = "braille_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1P2g8-IU2BM0T3XK_QdWr5whrbM_fkc1N"
TRAIN_DIR = "./Braille Dataset/train"
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# Download Model if Not Exists
# ==============================
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading Braille model from Google Drive...")
    response = requests.get(MODEL_URL, allow_redirects=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully!")
    else:
        raise Exception(f"❌ Failed to download model from Google Drive (status code: {response.status_code})")

# ==============================
# Load Model
# ==============================
model = load_model(MODEL_PATH)

# ==============================
# Create Label Map (from Train Folder)
# ==============================
train_datagen = ImageDataGenerator(rescale=1.0 / 255)

# If folder exists, use it to rebuild labels
if os.path.exists(TRAIN_DIR):
    print("📁 Generating class label mapping from training dataset...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical"
    )
    class_labels = train_generator.class_indices
    labels_map = {v: k for k, v in class_labels.items()}
else:
    print("⚠️ TRAIN_DIR not found — using default label mapping (A–J).")
    labels_map = {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
        5: "F", 6: "G", 7: "H", 8: "I", 9: "J"
    }

print("✅ Label mapping:", labels_map)

# ==============================
# Image Preprocessing Function
# ==============================
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================
# Prediction Endpoint
# ==============================
@app.route("/predict_braille", methods=["POST"])
def predict_braille():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    try:
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        predicted_class_index = int(tf.argmax(predictions, axis=-1).numpy()[0])
        predicted_label = labels_map.get(predicted_class_index, "Unknown")

        confidence = float(predictions[0][predicted_class_index])

        return jsonify({
            "predicted_class": predicted_label,
            "confidence": round(confidence, 4)
        }), 200
    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# ==============================
# Root Test Route
# ==============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Braille Prediction API is running!"})

# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
