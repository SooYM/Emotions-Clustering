import os
import sys
import pickle
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, render_template, send_from_directory

# Suppress stderr during mediapipe and protobuf imports to prevent noisy "GetPrototype" warnings
_original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception:
    sys.stderr = _original_stderr
    raise
finally:
    sys.stderr = _original_stderr

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNLABEL_DIR = os.path.join(BASE_DIR, "unlabel")
DATA_DIR = os.path.join(BASE_DIR, "clustering", "data")

FEATURES_PATH = os.path.join(DATA_DIR, "features.npy")
PATHS_PATH = os.path.join(DATA_DIR, "paths.npy")
KMEANS_PATH = os.path.join(DATA_DIR, "kmeans.pkl")
KNN_REGRESSOR_PATH = os.path.join(DATA_DIR, "knn_regressor.pkl")
DOWNSAMPLED_COORDS_PATH = os.path.join(DATA_DIR, "coords_downsampled.pkl")
REPRESENTATIVE_PATH = os.path.join(DATA_DIR, "representatives.pkl")
EMOTION_MODEL_PATH = os.path.join(DATA_DIR, "emotion_model.pkl")
LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

# Initialize Flask
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

# Global states
kmeans_model = None
knn_model = None
emotion_model = None
coords_downsampled = None
representatives = None
feature_names = None
detector = None
features = None
image_names = None

EMOTIONS = ["happy", "sad", "anxiety", "angry", "surprised", "disgust", "neutral"]
FEATURE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

def load_models():
    global kmeans_model, knn_model, emotion_model, coords_downsampled, representatives, feature_names, detector, features, image_names
    
    print("Loading model checkpoints...")
    if os.path.exists(FEATURES_PATH):
        features = np.load(FEATURES_PATH)
        
    if os.path.exists(PATHS_PATH):
        image_names = np.load(PATHS_PATH)
        
    if os.path.exists(KMEANS_PATH):
        with open(KMEANS_PATH, "rb") as f:
            kmeans_model = pickle.load(f)
            
    if os.path.exists(KNN_REGRESSOR_PATH):
        with open(KNN_REGRESSOR_PATH, "rb") as f:
            knn_model = pickle.load(f)
            
    if os.path.exists(EMOTION_MODEL_PATH):
        with open(EMOTION_MODEL_PATH, "rb") as f:
            emotion_model = pickle.load(f)
            
    if os.path.exists(DOWNSAMPLED_COORDS_PATH):
        with open(DOWNSAMPLED_COORDS_PATH, "rb") as f:
            coords_downsampled = pickle.load(f)
            
    if os.path.exists(REPRESENTATIVE_PATH):
        with open(REPRESENTATIVE_PATH, "rb") as f:
            representatives = pickle.load(f)
            
    feature_names = FEATURE_NAMES
        
    # Setup MediaPipe FaceLandmarker
    if os.path.exists(LANDMARKER_MODEL_PATH):
        print("Initializing MediaPipe FaceLandmarker...")
        base_options = python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        
    print("All models loaded successfully!")

# Helper function to extract blendshapes using MediaPipe Python SDK
def extract_blendshapes(pil_image):
    if detector is None:
        print("MediaPipe FaceLandmarker detector is not initialized.")
        return None
        
    try:
        # Convert PIL to RGB numpy array
        img_rgb = np.array(pil_image.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect landmarks
        result = detector.detect(mp_image)
        
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            blendshapes = result.face_blendshapes[0]
            scores = [item.score for item in blendshapes]
            return scores
            
    except Exception as e:
        print(f"Error running MediaPipe: {e}")
        
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/face_landmarker.task")
def serve_landmarker():
    return send_from_directory(BASE_DIR, "face_landmarker.task")

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(UNLABEL_DIR, filename)

@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    if not kmeans_model or not coords_downsampled or not representatives:
        return jsonify({"error": "Clustering data not ready. Please run feature extraction first."}), 503
        
    centroids = kmeans_model.cluster_centers_
    return jsonify({
        "num_clusters": len(centroids),
        "representatives": representatives,
        "coords": coords_downsampled
    })

# Prediction logic handler
def predict_on_features(scores):
    scores_arr = np.array([scores]) # Shape: (1, 52)
    
    # 1. Cluster assignment
    # Calculate distance to centroids
    distances = np.linalg.norm(kmeans_model.cluster_centers_ - scores_arr, axis=1)
    predicted_cluster = int(np.argmin(distances))
    
    # Prepare distances dict
    cluster_distances = {}
    for idx, dist in enumerate(distances):
        cluster_distances[idx] = float(dist)
        
    # 2. Project using KNN Regressor to 2D t-SNE space
    coords_2d = knn_model.predict(scores_arr)[0]
    
    # 3. Predict emotion probabilities using MLP
    emotion_prob = {}
    emotion_label = "unknown"
    if emotion_model is not None:
        probs = emotion_model.predict_proba(scores_arr)[0]
        pred_idx = int(np.argmax(probs))
        emotion_label = EMOTIONS[pred_idx]
        
        for idx, prob in enumerate(probs):
            emotion_prob[EMOTIONS[idx]] = float(prob)
            
    return {
        "predicted_cluster": predicted_cluster,
        "x": float(coords_2d[0]),
        "y": float(coords_2d[1]),
        "distances": cluster_distances,
        "emotion_label": emotion_label,
        "emotion_probabilities": emotion_prob
    }

@app.route("/api/predict", methods=["POST"])
def predict():
    if not kmeans_model or not knn_model or not detector:
        return jsonify({"error": "Models or MediaPipe are not fully loaded."}), 503
        
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400
        
    try:
        image = Image.open(file.stream).convert("RGB")
        scores = extract_blendshapes(image)
        
        if scores is None or len(scores) == 0:
            return jsonify({"error": "No face detected in the image. Please upload a clear face portrait."}), 400
            
        result = predict_on_features(scores)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error predicting: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route("/api/predict_blendshapes", methods=["POST"])
def predict_blendshapes():
    if not kmeans_model or not knn_model:
        return jsonify({"error": "Models are not fully loaded."}), 503
        
    data = request.json
    if not data or "blendshapes" not in data:
        return jsonify({"error": "Missing blendshapes payload."}), 400
        
    try:
        blendshapes_input = data["blendshapes"]
        
        # If it is a dictionary name -> score, rebuild the array in exact training order
        if isinstance(blendshapes_input, dict):
            if feature_names is None:
                return jsonify({"error": "Feature names list not loaded on backend."}), 503
            scores = [blendshapes_input.get(name, 0.0) for name in feature_names]
        else:
            # Assume it's already a sorted list
            scores = blendshapes_input
            
        if len(scores) != 52:
            return jsonify({"error": f"Expected 52 blendshape values, but got {len(scores)}"}), 400
            
        result = predict_on_features(scores)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error predicting blendshapes: {e}")
        return jsonify({"error": f"Failed to process blendshapes: {str(e)}"}), 500

@app.route("/api/recluster", methods=["POST"])
def recluster():
    global kmeans_model, representatives, coords_downsampled
    
    data = request.json or {}
    new_k = int(data.get("k", 7))
    
    if new_k < 2 or new_k > 15:
        return jsonify({"error": "Number of clusters must be between 2 and 15."}), 400
        
    if features is None or image_names is None:
        return jsonify({"error": "Features cache features.npy or paths.npy is missing."}), 503
        
    try:
        from sklearn.cluster import KMeans
        
        # 1. Run K-Means directly on 52-dim blendshapes
        print(f"Re-clustering to K={new_k} clusters...")
        kmeans_model = KMeans(n_clusters=new_k, random_state=42, n_init=10)
        assignments = kmeans_model.fit_predict(features)
        
        # 2. Identify new representative images per cluster (closest 15)
        representatives = {}
        for k in range(new_k):
            cluster_mask = (assignments == k)
            cluster_indices = np.where(cluster_mask)[0]
            cluster_features = features[cluster_indices]
            centroid = kmeans_model.cluster_centers_[k]
            
            # Distance in 52D space
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            sorted_local_indices = np.argsort(distances)
            closest_local_indices = sorted_local_indices[:15]
            
            cluster_reps = []
            for local_idx in closest_local_indices:
                global_idx = cluster_indices[local_idx]
                cluster_reps.append({
                    "name": str(image_names[global_idx]),
                    "distance": float(distances[local_idx])
                })
            representatives[k] = cluster_reps
            
        # 3. Update coords_downsampled mapping
        name_to_index = {name: idx for idx, name in enumerate(image_names)}
        for pt in coords_downsampled:
            name = pt["name"]
            if name in name_to_index:
                pt["cluster"] = int(assignments[name_to_index[name]])
                
        return jsonify({
            "status": "success",
            "num_clusters": new_k
        })
    except Exception as e:
        print(f"Re-clustering failed: {e}")
        return jsonify({"error": f"Failed to re-cluster: {str(e)}"}), 500

if __name__ == "__main__":
    try:
        load_models()
    except Exception as e:
        print(f"Warning: Models could not be loaded: {e}.")
        
    app.run(host="0.0.0.0", port=8000, debug=True)
