import os
import pickle
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNLABEL_DIR = os.path.join(BASE_DIR, "unlabel")
DATA_DIR = os.path.join(BASE_DIR, "clustering", "data")

KMEANS_PATH = os.path.join(DATA_DIR, "kmeans.pkl")
PCA_PATH = os.path.join(DATA_DIR, "pca.pkl")
DOWNSAMPLED_COORDS_PATH = os.path.join(DATA_DIR, "coords_downsampled.pkl")
REPRESENTATIVE_PATH = os.path.join(DATA_DIR, "representatives.pkl")

# Initialize Flask
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

# Device selection for inference (CPU is fine for single images)
device = torch.device("cpu")

# Load models and data
print("Loading model checkpoints...")
kmeans_model = None
pca_model = None
coords_downsampled = None
representatives = None
feature_extractor = None

def load_models():
    global kmeans_model, pca_model, coords_downsampled, representatives, feature_extractor
    
    if os.path.exists(KMEANS_PATH):
        with open(KMEANS_PATH, "rb") as f:
            kmeans_model = pickle.load(f)
            
    if os.path.exists(PCA_PATH):
        with open(PCA_PATH, "rb") as f:
            pca_model = pickle.load(f)
            
    if os.path.exists(DOWNSAMPLED_COORDS_PATH):
        with open(DOWNSAMPLED_COORDS_PATH, "rb") as f:
            coords_downsampled = pickle.load(f)
            
    if os.path.exists(REPRESENTATIVE_PATH):
        with open(REPRESENTATIVE_PATH, "rb") as f:
            representatives = pickle.load(f)
            
    # Load ResNet18 feature extractor
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    print("All models loaded successfully!")

# Define Image Transform
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(UNLABEL_DIR, filename)

@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    if not kmeans_model or not coords_downsampled or not representatives:
        return jsonify({"error": "Clustering data not ready. Please run feature extraction first."}), 503
        
    cluster_sizes = {}
    centroids = kmeans_model.cluster_centers_
    
    # Calculate size of each cluster from the downsampled set or overall assignments if available
    # For simplicity, we can count the distribution in the downsampled coordinates, or we can just send representatives.
    return jsonify({
        "num_clusters": len(centroids),
        "representatives": representatives,
        "coords": coords_downsampled
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if not feature_extractor or not kmeans_model or not pca_model:
        return jsonify({"error": "Models are not fully loaded."}), 503
        
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400
        
    try:
        # Load and preprocess image
        image = Image.open(file.stream).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            feats = feature_extractor(img_tensor)
            feats = feats.squeeze(-1).squeeze(-1).numpy()  # Shape: (1, 512)
            
        # Predict cluster
        distances = np.linalg.norm(kmeans_model.cluster_centers_ - feats, axis=1)
        predicted_cluster = int(np.argmin(distances))
        
        # Project using PCA to 2D
        coords_2d = pca_model.transform(feats)[0]
        
        # Prepare distances dict
        cluster_distances = {}
        for idx, dist in enumerate(distances):
            cluster_distances[idx] = float(dist)
            
        return jsonify({
            "predicted_cluster": predicted_cluster,
            "x": float(coords_2d[0]),
            "y": float(coords_2d[1]),
            "distances": cluster_distances
        })
        
    except Exception as e:
        print(f"Error predicting: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    # Load models before running server
    try:
        load_models()
    except Exception as e:
        print(f"Warning: Models could not be loaded: {e}. If this is initial training, run extract_features.py first.")
        
    app.run(host="0.0.0.0", port=8000, debug=True)
