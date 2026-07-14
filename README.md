# LatentSpace Clustering & Prediction Web Application

An end-to-end unsupervised learning pipeline and web application designed to cluster face images and provide real-time inference. The system processes 32,298 face images, extracts feature maps using a deep convolutional network (ResNet18), groups them using K-Means clustering, projects their coordinates into a 2D space via PCA, and serves them on an interactive, glassmorphic dark-theme web dashboard.

---

## Technical Architecture

- **Feature Extraction**: PyTorch Pretrained `ResNet18` (512-dimensional output vectors).
- **Unsupervised Clustering**: Scikit-Learn `KMeans` ($K=7$ clusters).
- **Dimensionality Reduction**: Scikit-Learn `PCA` (reducing 512 dimensions to 2D latent coordinates).
- **Backend API**: Lightweight `Flask` server hosting visual metadata endpoints and running real-time prediction pipelines.
- **Frontend Dashboard**: Vanilla HTML5, CSS3 (blur overlays, custom animations), and high-performance interactive HTML5 Canvas.

---

## Installation & Setup

Ensure you have Python 3.9+ installed. You can install all necessary packages using `pip`:

```bash
pip install torch torchvision scikit-learn flask pillow numpy requests
```

---

## How to Run

### Step 1: Run the Feature Extraction & Training Pipeline
This script loads the 32,298 images from the `unlabel` folder, extracts their features, fits K-Means and PCA models, caches data to `clustering/data/`, and selects representative cluster faces:

```bash
python3 clustering/extract_features.py
```
*Note: This utilizes Apple Metal Performance Shaders (MPS) automatically on macOS for GPU acceleration, completing in under 25 seconds.*

### Step 2: Start the Web Application Server
This starts the Flask development server which loads cached models and serves the single-page application:

```bash
python3 server.py
```

### Step 3: Access the Web App
Open your web browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## Directory Structure

```text
Clustering/
├── clustering/
│   ├── data/                   # Cached features, PCA and KMeans checkpoints
│   └── extract_features.py     # Training and extraction pipeline script
├── static/
│   ├── app.js                  # Canvas rendering, drag-drop inputs, animations
│   └── style.css               # Futuristic dark glassmorphic styling
├── templates/
│   └── index.html              # Core single page app layout
├── unlabel/                    # Target image directory (32,298 images)
├── README.md                   # Project documentation
└── server.py                   # Flask server entry point
```

---

## API Endpoints

### 1. Retrieve Cluster Coordinates
- **Route**: `GET /api/clusters`
- **Output**: JSON payload consisting of cluster centroids, representative images, and downsampled coordinates for plotting.

### 2. Real-Time Inference (Predict)
- **Route**: `POST /api/predict`
- **Form Data**: `image: <File>`
- **Output**: JSON payload returning the predicted cluster, its 2D coordinates `[x, y]`, and calculated distances to all 7 cluster centroids.
