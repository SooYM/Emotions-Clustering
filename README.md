# LatentSpace Clustering & Real-Time Webcam Visualizer

An end-to-end unsupervised learning pipeline and web application designed to cluster face expression datasets and run real-time emotion capture. The system processes face images, extracts 52-dimensional facial muscle blendshape features using MediaPipe, groups them using K-Means clustering, projects their coordinates into a 2D space via t-SNE, and serves them on an interactive, glassmorphic dark-theme web dashboard.

---

## Technical Architecture

- **Feature Representation**: 52-dimensional MediaPipe Face Blendshape scores (representing muscle movements like `mouthSmileLeft`, `browDownRight`, etc.).
- **Unsupervised Clustering**: Scikit-Learn `KMeans` ($K=7$ clusters) trained directly on the blendshape coordinates.
- **Dimensionality Reduction**: Non-linear `t-SNE` (t-Distributed Stochastic Neighbor Embedding) used on the coordinates to ensure highly separated, distinct cluster clouds.
- **Latent Projection**: Scikit-Learn `KNeighborsRegressor` ($K=5$, distance-weighted) trained to map 52D blendshapes to the 2D t-SNE coordinate space in real time.
- **Emotion Classifier**: Pre-trained Multi-Layer Perceptron (MLP) `MLPClassifier` (`emotion_model.pkl`) loaded to classify facial blendshape features into 7 basic emotions: `happy, sad, anxiety, angry, surprised, disgust, neutral`.
- **Backend API**: Lightweight `Flask` server serving models and executing real-time predictions.
- **Frontend Dashboard**: Vanilla HTML5, CSS3 (glassmorphic dark theme), and high-performance HTML5 Canvas with custom animations. Includes client-side MediaPipe tasks-vision WebAssembly face meshes rendering the full 478 points point cloud, contours, lips, eye outlines, brows, and pupils with real-time dynamic emotion color-coding.

---

## Installation & Setup

Ensure you have Python 3.9+ installed. Install the necessary packages using `pip`:

```bash
pip install torch torchvision scikit-learn flask pillow numpy requests mediapipe "protobuf==3.20.3"
```
*Note: Pinning `protobuf==3.20.3` is required to ensure compatibility with MediaPipe's Python bindings.*

---

## How to Run

### Step 1: Run the Feature Extraction & Training Pipeline
This script loads the pre-extracted face blendshapes cache from `dataset_features.npz`, filters files matching the `unlabel/` folder, fits K-Means and t-SNE, trains the KNN mapping regressor, and saves checkpoints to `clustering/data/`:

```bash
python3 clustering/extract_features.py
```

### Step 2: Start the Web Application Server
Ensure your terminal is in the project directory, and start the Flask development server:

```bash
python3 server.py
```

### Step 3: Access the Web App
Open your web browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

Click on **"Live Scan"** to start your webcam, grant permissions, and view your expressions mapped dynamically in real-time onto the cluster clouds!

---

## Directory Structure

```text
Clustering/
├── clustering/
│   ├── data/                   # Cached features, KMeans, KNN, and Emotion MLP checkpoints
│   └── extract_features.py     # Training and t-SNE pipeline script
├── static/
│   ├── app.js                  # Canvas rendering, MediaPipe loops, and animations
│   └── style.css               # Futuristic dark glassmorphic styling
├── templates/
│   └── index.html              # Core single page app layout
├── unlabel/                    # Target image directory (32,298 images)
├── face_landmarker.task        # MediaPipe landmarker task bundle file
├── README.md                   # Project documentation
└── server.py                   # Flask server entry point
```

---

## API Endpoints

### 1. Retrieve Cluster Coordinates
- **Route**: `GET /api/clusters`
- **Output**: JSON payload consisting of cluster centroids, representative images, and coordinates for plotting.

### 2. File Upload Inference
- **Route**: `POST /api/predict`
- **Form Data**: `image: <File>`
- **Output**: JSON payload returning the predicted cluster, coordinates `[x, y]`, centroid distances, predicted emotion label, and probabilities.

### 3. Real-Time Blendshapes Inference (Webcam Scan)
- **Route**: `POST /api/predict_blendshapes`
- **JSON Payload**: `{"blendshapes": {"browDownLeft": 0.0, "mouthSmileLeft": 0.8, ...}}`
- **Output**: JSON payload returning predicted cluster, coordinates `[x, y]`, centroid distances, predicted emotion label, and probabilities.
