import os
import time
import pickle
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor

# Configuration
CLUSTERING_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CLUSTERING_DIR)
UNLABEL_DIR = os.path.join(BASE_DIR, "unlabel")
DATA_DIR = os.path.join(CLUSTERING_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_PATH = "/Users/sooyauming/Desktop/Intern/Classification/dataset_features.npz"

FEATURES_PATH = os.path.join(DATA_DIR, "features.npy")
PATHS_PATH = os.path.join(DATA_DIR, "paths.npy")
KMEANS_PATH = os.path.join(DATA_DIR, "kmeans.pkl")
KNN_REGRESSOR_PATH = os.path.join(DATA_DIR, "knn_regressor.pkl")
DOWNSAMPLED_COORDS_PATH = os.path.join(DATA_DIR, "coords_downsampled.pkl")
REPRESENTATIVE_PATH = os.path.join(DATA_DIR, "representatives.pkl")

NUM_CLUSTERS = 7
DOWNSAMPLE_LIMIT = 3000

def main():
    t_start = time.time()
    
    # 1. Load Pre-extracted MediaPipe Blendshapes Features
    print("Loading pre-extracted features cache from Classification directory...")
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cached features file not found at {CACHE_PATH}")
        
    cache = np.load(CACHE_PATH, allow_pickle=True)
    X_train = cache["X_train"]
    paths_train = cache["paths_train"]
    X_test = cache["X_test"]
    paths_test = cache["paths_test"]
    
    print(f"Loaded raw training features: {X_train.shape}, test features: {X_test.shape}")
    
    # Filter features to keep only images that exist in unlabel
    print(f"Filtering features to match files in: {UNLABEL_DIR}...")
    existing_files = set(os.listdir(UNLABEL_DIR))
    
    features_list = []
    names_list = []
    
    # Process train split
    for idx, p in enumerate(paths_train):
        fname = os.path.basename(p)
        if fname in existing_files:
            features_list.append(X_train[idx])
            names_list.append(fname)
            
    # Process test split
    for idx, p in enumerate(paths_test):
        fname = os.path.basename(p)
        if fname in existing_files:
            features_list.append(X_test[idx])
            names_list.append(fname)
            
    features = np.array(features_list)
    image_names = np.array(names_list)
    
    print(f"Filtered dataset size: {len(features)} images (out of {len(existing_files)} in unlabel).")
    if len(features) == 0:
        raise ValueError("No matching images found in unlabel directory. Make sure the dataset is populated.")
        
    # Save a local cache of the filtered features
    np.save(FEATURES_PATH, features)
    np.save(PATHS_PATH, image_names)
    
    # 2. K-Means Clustering on the 52 Blendshape Features
    print(f"Running K-means clustering directly on 52-dim blendshapes (K={NUM_CLUSTERS})...")
    t0 = time.time()
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    assignments = kmeans.fit_predict(features)
    t1 = time.time()
    print(f"K-means completed in {t1-t0:.2f} seconds.")
    
    # Save K-means model
    with open(KMEANS_PATH, "wb") as f:
        pickle.dump(kmeans, f)
        
    # 3. Identify Representative Images
    print("Identifying representative images per cluster (closest 15)...")
    representatives = {}
    for k in range(NUM_CLUSTERS):
        cluster_mask = (assignments == k)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_features = features[cluster_indices]
        centroid = kmeans.cluster_centers_[k]
        
        # Calculate Euclidean distance in 52D space
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
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
        
    # 4. Downsampling & t-SNE for Highly Separated 2D Coordinates
    print(f"Downsampling dataset for t-SNE (limit={DOWNSAMPLE_LIMIT})...")
    np.random.seed(42)
    total_images = len(features)
    
    if total_images <= DOWNSAMPLE_LIMIT:
        sampled_indices = np.arange(total_images)
    else:
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
        
    sampled_features = features[sampled_indices]
    sampled_names = image_names[sampled_indices]
    sampled_assignments = assignments[sampled_indices]
    
    print("Fitting t-SNE on downsampled features for optimal cluster separation...")
    t0 = time.time()
    # Using high perplexity for global structure & good separation
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, init="pca", learning_rate="auto")
    coords_2d_sampled = tsne.fit_transform(sampled_features)
    print(f"t-SNE completed in {time.time() - t0:.2f} seconds.")
    
    # 5. Train KNeighborsRegressor to Map New 52D Inputs into 2D t-SNE Space
    print("Training KNeighborsRegressor (512D -> 2D t-SNE mapping)...")
    t0 = time.time()
    knn_regressor = KNeighborsRegressor(n_neighbors=5, weights="distance")
    knn_regressor.fit(sampled_features, coords_2d_sampled)
    print(f"KNN Regressor trained in {time.time() - t0:.4f} seconds.")
    
    # Save KNN Regressor model
    with open(KNN_REGRESSOR_PATH, "wb") as f:
        pickle.dump(knn_regressor, f)
        
    # 6. Save Coordinates for the UI Map
    coords_downsampled = []
    for i, idx in enumerate(sampled_indices):
        coords_downsampled.append({
            "name": sampled_names[i],
            "x": float(coords_2d_sampled[i, 0]),
            "y": float(coords_2d_sampled[i, 1]),
            "cluster": int(sampled_assignments[i])
        })
        
    with open(DOWNSAMPLED_COORDS_PATH, "wb") as f:
        pickle.dump(coords_downsampled, f)
        
    print("Downsampled coordinates saved.")
    print(f"All pipeline steps complete! Total execution time: {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    main()
