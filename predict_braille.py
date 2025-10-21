from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import os
import requests
from flask_cors import CORS

# ==============================
# Flask App Initialization
# ==============================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ Allow all cross-origin requests (Expo, web, mobile)

# ==============================
# Model Setup
# ==============================
MODEL_PATH = "braille_model.tflite"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1L8blof1IrLAGydJpba52NsbSzUfKUV0l"
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Download model if not already available
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading Braille model from Google Drive...")
    response = requests.get(MODEL_URL, allow_redirects=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully!")

# ✅ Load TensorFlow Lite model safely
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Label map
labels_map = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J"
}

# ==============================
# Image Preprocessing Function
# ==============================
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# ==============================
# API Route
# ==============================
@app.route("/predict_braille", methods=["POST"])
def predict_braille():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        # ✅ Preprocess image
        input_data = preprocess_image(image_path)

        # ✅ Run inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])[0]

        # ✅ Process results
        predicted_index = int(np.argmax(output_data))
        confidence = float(output_data[predicted_index])
        predicted_label = labels_map.get(predicted_index, "Unknown")

        # ✅ Return clean JSON response
        return jsonify({
            "predicted_class": predicted_label,
            "confidence": round(confidence, 4)
        }), 200

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        # ✅ Safe cleanup
        if "image_path" in locals() and os.path.exists(image_path):
            os.remove(image_path)

# ==============================
# Root Route for Testing
# ==============================
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "✅ Braille API is running!"})

# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
