# LatentSpace Clustering & Real-Time Webcam Visualizer

An end-to-end unsupervised learning pipeline and web application designed to cluster face expression datasets and run real-time emotion capture. The system processes face images, extracts 52-dimensional facial muscle blendshape features using MediaPipe, groups them using K-Means clustering, projects their coordinates into a 2D space via t-SNE, and serves them on an interactive, glassmorphic dark-theme web dashboard.

---

## Technical Architecture

The application combines supervised emotion models with unsupervised clustering and manifold embedding:
1. **Unsupervised K-Means Clustering**: Clusters the 52-dimensional facial muscle blendshape features extracted by MediaPipe. The K-means algorithm groups expressions based strictly on muscle activation similarities. The number of clusters ($K$) is fully configurable by the user at runtime.
2. **t-SNE Dimensionality Reduction**: Manifold learning (t-Distributed Stochastic Neighbor Embedding) translates the high-dimensional (52D) facial muscle scores into a highly-separated 2D space. t-SNE models local similarities to ensure cluster clusters are visually distinct.
3. **K-Neighbors Regression (Point Mapping)**: A distance-weighted KNN regressor ($K=5$) bridges the 52D model features to the 2D t-SNE space, enabling immediate coordinate projection for custom webcam frames and uploads without re-running the heavy t-SNE solver.
4. **Multi-Layer Perceptron (Emotion Classifier)**: A supervised MLP neural network predicts corresponding emotion probabilities, which dynamically color-code the user's face mesh.
5. **Interactive Controls**: Allows live, real-time recalculation of K-Means clusters via the frontend control widget.

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

### 4. Live Custom Re-Clustering
- **Route**: `POST /api/recluster`
- **JSON Payload**: `{"k": 5}` (number of clusters $K$ between 2 and 15)
- **Output**: JSON payload confirming success: `{"status": "success", "num_clusters": 5}`. Centroids, point labels, and representative galleries are immediately re-calculated on the fly.
