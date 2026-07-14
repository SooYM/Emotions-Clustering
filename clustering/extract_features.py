import os
import time
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuration
CLUSTERING_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CLUSTERING_DIR)
UNLABEL_DIR = os.path.join(BASE_DIR, "unlabel")
DATA_DIR = os.path.join(CLUSTERING_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(DATA_DIR, "features.npy")
PATHS_PATH = os.path.join(DATA_DIR, "paths.npy")
KMEANS_PATH = os.path.join(DATA_DIR, "kmeans.pkl")
PCA_PATH = os.path.join(DATA_DIR, "pca.pkl")
DOWNSAMPLED_COORDS_PATH = os.path.join(DATA_DIR, "coords_downsampled.pkl")
REPRESENTATIVE_PATH = os.path.join(DATA_DIR, "representatives.pkl")

BATCH_SIZE = 128
NUM_CLUSTERS = 7
DOWNSAMPLE_LIMIT = 3000

# Device selection (use MPS if available on Mac, otherwise CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Mac GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

class ImageFolderDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.filenames = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.filenames.sort()  # Ensure deterministic order

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.directory, filename)
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            # Return dummy image if loading fails
            print(f"Error loading {filepath}: {e}")
            image = Image.new("RGB", (48, 48), color=0)
            
        if self.transform:
            image = self.transform(image)
        return image, filename

def get_resnet_feature_extractor():
    # Load ResNet18
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()
    # Remove the final classification layer
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    return feature_extractor

def main():
    t_start = time.time()
    
    # 1. Feature Extraction
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolderDataset(UNLABEL_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Loaded dataset: {len(dataset)} images.")
    
    if os.path.exists(FEATURES_PATH) and os.path.exists(PATHS_PATH):
        print("Loading cached features...")
        features = np.load(FEATURES_PATH)
        image_names = np.load(PATHS_PATH)
    else:
        print("Extracting features (this may take 1-2 minutes on CPU)...")
        feature_extractor = get_resnet_feature_extractor()
        
        feature_list = []
        name_list = []
        
        with torch.no_grad():
            for i, (images, filenames) in enumerate(dataloader):
                images = images.to(device)
                feats = feature_extractor(images)  # Shape: (B, 512, 1, 1)
                feats = feats.squeeze(-1).squeeze(-1)  # Shape: (B, 512)
                feature_list.append(feats.cpu().numpy())
                name_list.extend(filenames)
                
                if (i + 1) % 20 == 0:
                    print(f"Processed {len(feature_list) * BATCH_SIZE}/{len(dataset)} images...")
                    
        features = np.concatenate(feature_list, axis=0)
        image_names = np.array(name_list)
        
        # Save features and names cache
        np.save(FEATURES_PATH, features)
        np.save(PATHS_PATH, image_names)
        print("Features and paths cached.")
        
    print(f"Feature shape: {features.shape}")
    
    # 2. K-Means Clustering
    print(f"Running K-means clustering with K={NUM_CLUSTERS}...")
    t0 = time.time()
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    assignments = kmeans.fit_predict(features)
    t1 = time.time()
    print(f"K-means completed in {t1-t0:.2f} seconds.")
    
    # Save K-means model
    with open(KMEANS_PATH, "wb") as f:
        pickle.dump(kmeans, f)
        
    # 3. PCA Dimensionality Reduction
    print("Fitting PCA model...")
    t0 = time.time()
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(features)
    t1 = time.time()
    print(f"PCA completed in {t1-t0:.2f} seconds.")
    
    # Save PCA model
    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)
        
    # 4. Identify Representative Images
    print("Identifying representative images per cluster...")
    representatives = {}  # cluster_idx -> list of dicts with name, dist
    
    # For each cluster, find the 15 closest points to the centroid
    for k in range(NUM_CLUSTERS):
        cluster_mask = (assignments == k)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_features = features[cluster_indices]
        centroid = kmeans.cluster_centers_[k]
        
        # Compute distances from centroid to all features in this cluster
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        
        # Sort indices by distance (ascending)
        sorted_local_indices = np.argsort(distances)
        closest_local_indices = sorted_local_indices[:15]
        
        cluster_reps = []
        for local_idx in closest_local_indices:
            global_idx = cluster_indices[local_idx]
            cluster_reps.append({
                "name": image_names[global_idx],
                "distance": float(distances[local_idx])
            })
        representatives[k] = cluster_reps
        print(f"Cluster {k}: size = {np.sum(cluster_mask)}, top representative distance = {cluster_reps[0]['distance']:.4f}")
        
    with open(REPRESENTATIVE_PATH, "wb") as f:
        pickle.dump(representatives, f)
        
    # 5. Downsample points for UI Visualization
    print(f"Creating downsampled coordinate set (limit={DOWNSAMPLE_LIMIT})...")
    np.random.seed(42)
    total_images = len(dataset)
    
    if total_images <= DOWNSAMPLE_LIMIT:
        sampled_indices = np.arange(total_images)
    else:
        # Sample uniformly across clusters to keep distribution representative
        sampled_indices = []
        per_cluster_limit = DOWNSAMPLE_LIMIT // NUM_CLUSTERS
        for k in range(NUM_CLUSTERS):
            cluster_indices = np.where(assignments == k)[0]
            sampled_cluster_indices = np.random.choice(
                cluster_indices, 
                size=min(per_cluster_limit, len(cluster_indices)), 
                replace=False
            )
            sampled_indices.extend(sampled_cluster_indices)
            
        sampled_indices = np.array(sampled_indices)
        
    # Create clean data structure for UI
    coords_downsampled = []
    for idx in sampled_indices:
        coords_downsampled.append({
            "name": image_names[idx],
            "x": float(coords_2d[idx, 0]),
            "y": float(coords_2d[idx, 1]),
            "cluster": int(assignments[idx])
        })
        
    with open(DOWNSAMPLED_COORDS_PATH, "wb") as f:
        pickle.dump(coords_downsampled, f)
        
    print("Downsampled coordinates saved.")
    
    print(f"All done! Total time: {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    main()
