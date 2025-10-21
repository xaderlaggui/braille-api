from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
import requests
from flask_cors import CORS  # ✅ Add this

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ Allow Expo or any domain



# Google Drive Model Setup
MODEL_PATH = "braille_model.tflite"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1L8blof1IrLAGydJpba52NsbSzUfKUV0l"
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading Braille model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully!")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Replace this with your actual label list
labels_map = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J"
}

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# Prediction endpoint
@app.route("/predict_braille", methods=["POST"])
def predict_braille():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    try:
        # Preprocess and predict
        input_data = preprocess_image(image_path)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(output_data))
        confidence = float(output_data[predicted_index])
        predicted_label = labels_map.get(predicted_index, "Unknown")

        return jsonify({
            "predicted_class": predicted_label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(image_path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
