import os
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torchvision import transforms, models

from data import processSolarData, loadDataset

from PIL import Image


def runFeatureExtraction(cfg, num_clusters=6, max_preview_imgs=25):
    print("\n=== Feature Extraction + Clustering ===")

    # -------------------------------
    # Load dataset (same as your main)
    # -------------------------------
    processSolarData("data", rebuildCombinedData=False, showInfo=False)
    data = loadDataset(cfg.csvPath, False)

    imgPaths = [
        os.path.join(cfg.imgFolder, os.path.basename(p))
        for p in data["PictureName"].values
    ]
    irr = data["Irradiance"].values.astype(np.float32)

    # remove missing images
    goodIdx = [i for i, p in enumerate(imgPaths) if os.path.exists(p)]
    imgPaths = [imgPaths[i] for i in goodIdx]
    irr = irr[goodIdx]

    print(f"Loaded {len(imgPaths)} images for MobileNet feature extraction")

    # -------------------------------
    # MobileNetV2 Feature Extractor
    # -------------------------------
    device = cfg.device
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier = torch.nn.Identity()
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(cfg.imgSize),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # -------------------------------
    # Extract features
    # -------------------------------
    all_features = []

    for path in tqdm(imgPaths, desc="Extracting features"):
        img = Image.open(path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(img_t).cpu().numpy().flatten()

        all_features.append(feat)

    all_features = np.array(all_features)

    # -------------------------------
    # Save raw features
    # -------------------------------
    os.makedirs("features", exist_ok=True)

    df = pd.DataFrame(all_features, columns=[f"f{i}" for i in range(all_features.shape[1])])
    df["img_path"] = imgPaths
    df["irradiance"] = irr

    df.to_csv("features/feature_vectors.csv", index=False)
    print("Saved features/feature_vectors.csv")

    # -------------------------------
    # PCA
    # -------------------------------
    pca50 = PCA(n_components=50)
    feat50 = pca50.fit_transform(all_features)

    pca2 = PCA(n_components=2)
    feat2d = pca2.fit_transform(feat50)

    # -------------------------------
    # Clustering
    # -------------------------------
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(feat50)

    df["cluster"] = cluster_ids
    df["pca_x"] = feat2d[:, 0]
    df["pca_y"] = feat2d[:, 1]

    df.to_csv("features/feature_vectors_with_clusters.csv", index=False)
    print("Saved features/feature_vectors_with_clusters.csv")

    # -------------------------------
    # Cluster preview images
    # -------------------------------
    os.makedirs("cluster_previews", exist_ok=True)
    os.makedirs("cluster_averages", exist_ok=True)

    for c in range(num_clusters):
        cluster_imgs = df[df.cluster == c]["img_path"].values

        # PREVIEW GRID
        preview_imgs = cluster_imgs[:max_preview_imgs]
        small_images = []

        for p in preview_imgs:
            img = cv2.imread(p)
            img = cv2.resize(img, (128, 128))
            small_images.append(img)

        if small_images:
            rows = []
            for i in range(0, len(small_images), 5):
                rows.append(np.hstack(small_images[i:i+5]))
            grid = np.vstack(rows)
            cv2.imwrite(f"cluster_previews/cluster_{c}.jpg", grid)

        # AVERAGE IMAGE
        avg_imgs = []
        for p in cluster_imgs:
            img = cv2.imread(p).astype(np.float32)
            img = cv2.resize(img, (128, 128))
            avg_imgs.append(img)

        if avg_imgs:
            avg = np.mean(avg_imgs, axis=0).astype(np.uint8)
            cv2.imwrite(f"cluster_averages/avg_cluster_{c}.jpg", avg)

    print("=== Feature extraction + clustering DONE ===\n")
    return df
