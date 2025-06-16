import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from werkzeug.utils import secure_filename
import numpy as np

try:
    from tensorflow.keras import layers, models
except ImportError:
    print("Failed to import tensorflow.keras. Using tf.keras instead.")
    from tensorflow import keras
    layers = keras.layers
    models = keras.models

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})  # Allow frontend origin

# Configuration
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to your saved model
MODEL_PATH = 'D:/archive (9)/tomato.h5'  # Update if model is elsewhere

# Model parameters
INPUT_SHAPE = (256, 256, 3)
N_CLASSES = 3  # 3 classes: Tomato_Early_Blight, Tomato_Late_Blight, Tomato_Healthy

# Define preprocessing and augmentation layers
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(256, 256),
    layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# Disease actions dictionary
disease_actions = {
    "Tomato_Early_Blight": [
        "Remove Infected Leaves: Carefully remove and destroy affected leaves to reduce fungal spore spread, avoiding contact with healthy plants.",
        "Apply Fungicides: Use fungicides like chlorothalonil or copper-based products, following label instructions (e.g., apply every 7-10 days during humid conditions).",
        "Improve Air Circulation: Space plants adequately and prune lower leaves to enhance airflow, reducing moisture that favors the fungus.",
        "Crop Rotation: Avoid planting tomatoes or potatoes in the same soil for at least 2 years to break the disease cycle.",
        "Monitor Watering: Water at the base, not overhead, to keep foliage dry, ideally in the morning."
    ],
    "Tomato_Late_Blight": [
        "Destroy Infected Plants: Remove and burn or bag severely infected plants immediately, as late blight spreads rapidly.",
        "Apply Fungicides Urgently: Use fungicides like mancozeb or metalaxyl, starting at the first sign of disease, and reapply as directed.",
        "Avoid Wet Conditions: Ensure good drainage and avoid watering during cool, wet weather to limit spore germination.",
        "Sanitize Equipment: Clean tools and stakes with a 10% bleach solution to prevent spreading the pathogen.",
        "Monitor Nearby Crops: Check potatoes or other tomatoes nearby, as late blight can spread across fields."
    ],
    "Tomato_Healthy": [
        "Maintain Good Practices: Continue proper watering, fertilizing, and pruning to keep plants strong.",
        "Monitor Regularly: Inspect plants weekly for early signs of disease, especially during humid or rainy periods.",
        "Preventive Sprays: Consider preventive fungicides (e.g., copper-based) during high-risk seasons (e.g., wet summers).",
        "Support Growth: Use stakes or cages to support plants, improving airflow and reducing pest access.",
        "Soil Health: Test soil and amend with compost or nutrients to sustain plant vigor."
    ]
}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load the model with fallback
def load_model_safely(model_path):
    try:
        # Attempt to load the model directly
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully using load_model.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to manual model definition and weight loading.")
        
        # Define the model architecture manually
        model = models.Sequential([
            resize_and_rescale,
            data_augmentation,
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(N_CLASSES, activation='softmax')
        ])
        
        # Compile the model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        # Verify layer count
        print(f"Model has {len(model.layers)} layers defined.")
        
        # Load weights
        try:
            model.load_weights(model_path)
            print("Weights loaded successfully.")
            return model
        except Exception as weight_error:
            print(f"Failed to load weights: {weight_error}")
            return None

# Load the model when the app starts
model = load_model_safely(MODEL_PATH)
if model is None:
    raise RuntimeError("Failed to load the model. Check the model file and configuration.")

# Preprocess image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx]) * 100
            
            class_names = ['Tomato_Early_Blight', 'Tomato_Late_Blight', 'Tomato_Healthy']
            predicted_label = class_names[predicted_class_idx]
            
            # Get recommended actions
            actions = disease_actions.get(predicted_label, ["No actions available."])
            
            os.remove(file_path)
            return jsonify({
                'predicted_class': predicted_label,
                'confidence': confidence,
                'recommended_actions': actions
            }), 200
        
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

# Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
    print(f"Starting Flask app with TensorFlow {tf.__version__}")
    app.run(debug=True, host='0.0.0.0', port=5000)