from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from tensorflow.keras.models import load_model 
import numpy as np
from PIL import Image
import io
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Enable CORS
CORS(app)

# Enable CSRF protection
csrf = CSRFProtect(app)

# Enable Rate Limiting
limiter = Limiter(get_remote_address, app=app, default_limits=["100 per minute"])

# Load Chest X-ray Classification Model
try:
    model = load_model("Chest-X-Ray-Image-Classification\models\main_model.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

# Define class labels
CLASS_LABELS = {0: "Normal", 1: "Pneumonia"}  # Adjust based on your model



# Preprocessing function for images
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert to RGB (3 channels)
    img = img.resize((256, 256))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction function
def predict_xray(image_data):
    try:
        processed_image = preprocess_image(image_data)
        prediction = model.predict(processed_image)
        
        raw_score = float(prediction[0][0])  # ดึงค่าผลลัพธ์ออกมา
        print(f"Raw Prediction Score: {raw_score}")  # Debugging

        # แปลงค่าเป็น % ตามเกณฑ์ 0.5 เป็นจุดกึ่งกลาง
        if raw_score <= 0.5:
            normal_confidence = (0.5 - raw_score) * 200  # ค่าปรับให้เป็น 0 - 100%
            pneumonia_confidence = 100 - normal_confidence
        else:
            pneumonia_confidence = (raw_score - 0.5) * 200  # ค่าปรับให้เป็น 0 - 100%
            normal_confidence = 100 - pneumonia_confidence

        # ตัดสินผลลัพธ์ว่าเป็น Normal หรือ Pneumonia
        class_index = int(raw_score > 0.5)

        return {
            "result": CLASS_LABELS[class_index],
            "normal_confidence": round(normal_confidence, 2),
            "pneumonia_confidence": round(pneumonia_confidence, 2)
        }
    except Exception as e:
        return {"error": str(e)}






# Routes
@app.route("/")
@limiter.limit("10 per minute")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@csrf.exempt
@limiter.limit("5 per second")
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        image = request.files["image"].read()
        result = predict_xray(image)
        print(f"Result: {result}")  # Debug statement
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
