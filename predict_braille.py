from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import tensorflow as tf
import os
import requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Model and directories
MODEL_URL = "https://www.dropbox.com/scl/fi/hno5xglstri1m9v3zo613/braille_model.h5?rlkey=fssf1nay26xa31fcen15elpmx&st=dkd2eamu&dl=1"
MODEL_PATH = "braille_model.h5"
TRAIN_DIR = "./Braille Dataset/train"
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Dropbox...")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model downloaded successfully.")
    else:
        raise Exception(f"Failed to download model: {response.status_code}")

# Load trained model
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Generate label mapping
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=42,
    class_mode='categorical'
)
labels_map = {v: k for k, v in train_generator.class_indices.items()}

# Image preprocessing
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

# Prediction route
@app.route('/predict_braille', methods=['POST'])
def predict_braille():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    try:
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        predicted_class_index = tf.argmax(predictions, axis=-1).numpy()[0]
        predicted_label = labels_map.get(predicted_class_index, "Unknown")

        return jsonify({
            'predicted_class': predicted_label,
            'confidence': float(predictions[0][predicted_class_index])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
