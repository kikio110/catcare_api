from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models\cat_skin_disease_Model_vgg.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_names = ['jamuran', 'ringworm', 'scabbies']

# Descriptions
disease_info = {
    "jamuran": "Jamur kulit pada kucing merupakan infeksi yang sangat menular...",
    "ringworm": "Ringworm pada kucing merupakan infeksi kulit akibat jamur dermatofita...",
    "scabbies": "Scabies pada kucing adalah penyakit kulit yang disebabkan oleh tungau Sarcoptes scabiei..."
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((160, 160))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        predicted_index = np.argmax(output)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(output))

        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "description": disease_info[predicted_class]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return "Hello, this is the Cat Skin Disease Classification API."

if __name__ == '__main__':
    app.run(debug=True)
