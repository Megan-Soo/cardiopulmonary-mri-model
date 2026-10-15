#!/usr/bin/env python3
import argparse,re, os
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def midpoint_initial_means(k, low=0, high=255):
    width = (high - low) / k
    mids = low + (np.arange(k) + 0.5) * width
    return mids.reshape(-1, 1)

def hierarchical_kmeans_split_c2(X3d, background_mask):
    """
    X3d: (H, W, D) grayscale volume in [0,255]
    background_mask: boolean array, True = foreground, False = background
    Returns: labels_3d with shape (H, W, D)
    Background voxels are assigned label 0.
    """
    H, W, D = X3d.shape

    # Extract only foreground voxels
    fg_indices = np.where(background_mask)
    X_fg = X3d[fg_indices].reshape(-1, 1).astype(np.float32)

    # -------------------------------
    # Level 1: 4 clusters (C2–C5)
    # -------------------------------
    init_centroids_lvl1 = midpoint_initial_means(4)
    km1 = KMeans(n_clusters=4, init=init_centroids_lvl1, n_init=1, random_state=0)
    labels_lvl1 = km1.fit_predict(X_fg)  # 0..3
    # Sort the clusters by intensity to assign C2–C5 in order of increasing intensity
    centroids = km1.cluster_centers_.flatten()  # shape (4,)
    order = np.argsort(centroids)  # e.g., [2, 0, 3, 1]
    mapping = {old: new for new, old in enumerate(order)}
    labels_lvl1_sorted = np.array([mapping[l] for l in labels_lvl1])
    labels_lvl1 = labels_lvl1_sorted + 2        # → 2..5

    # -------------------------------
    # Level 2: refine C2 → C21–C24
    # -------------------------------
    C2_mask = labels_lvl1 == 2
    X_C2 = X_fg[C2_mask]

    km2 = KMeans(n_clusters=4, n_init=10, random_state=0)
    labels_C2 = km2.fit_predict(X_C2)  # 0..3
    # Sort the clusters by intensity to assign C21–C24 in order of increasing intensity
    centroids = km2.cluster_centers_.flatten()  # shape (4,)
    order = np.argsort(centroids)  # e.g., [2, 0, 3, 1]
    mapping = {old: new for new, old in enumerate(order)}
    labels_C2 = np.array([mapping[l] for l in labels_C2]) # 0..3, sorted by intensity

    # -------------------------------
    # Merge: C21-C22 → new C1, C23–C24 → new C2
    # -------------------------------
    final_fg_labels = labels_lvl1.copy()
    C2_indices = np.where(C2_mask)[0]

    final_fg_labels[C2_indices[labels_C2 <= 1]] = 1      # C21, C22 → C1
    final_fg_labels[C2_indices[labels_C2 >= 2]] = 2      # C23, C24 → C2

    # -------------------------------
    # Scatter back into 3D volume
    # -------------------------------
    labels_3d = np.zeros((H, W, D), dtype=np.int32)
    labels_3d[fg_indices] = final_fg_labels

    return labels_3d

def main():
    # Read mask lung
    mask_lung = sitk.ReadImage(mask_path_lung)
    # Read mask vessels
    mask_vessels = sitk.ReadImage(mask_path_vessel)
    
    # erode 3 pixels
    radius = [3,3,3]
    mask_lung = sitk.BinaryThreshold(mask_lung, lowerThreshold=1, upperThreshold=2, insideValue=1, outsideValue=0) # accept labels 1 and 2
    mask_lung = sitk.BinaryErode(sitk.Cast(mask_lung, sitk.sitkUInt8), radius, sitk.sitkBall)

    # Remove vessels from lung mask
    mask_lung = sitk.BinaryThreshold(mask_lung,1,2,1,0)
    mask_lung = sitk.Cast(mask_lung,sitk.sitkUInt8)-sitk.Cast(mask_vessels,sitk.sitkUInt8)
    if outdir:
        mask_dir = os.path.dirname(mask_path_lung)
        mask_name = os.path.basename(mask_path_lung).split('.nii')[0]
        sitk.WriteImage(mask_lung,os.path.join(mask_dir,f'{mask_name}_eroded.nii.gz'))
    mask_arr = sitk.GetArrayFromImage(mask_lung)

    # Read map
    image = sitk.ReadImage(img)
    # Apply median filter to perfusion map
    if 'q' in mtype or 'p' in mtype:
        image = sitk.Median(image,[2,2,2])
        if outdir:
            sitk.WriteImage(image,os.path.join(outdir,'pw_median.nii.gz'))
    image = sitk.GetArrayFromImage(image)
    image = image.astype(np.float32)

    # --- apply mask ---
    image = np.where(mask_arr>0,image,0)

    if outdir:
        mean = np.mean(image[mask_arr>0])
        res = np.where(mask_arr>0,image/mean,0)
        res = sitk.GetImageFromArray(res)
        res.CopyInformation(mask_lung)  # copy spacing, origin, direction from reference image
        if 'q' in mtype or 'p' in mtype:
            sitk.WriteImage(res, os.path.join(outdir,"pw_masked.nii.gz"))
        elif 'v' in mtype:
            sitk.WriteImage(res, os.path.join(outdir,"v_masked.nii.gz"))

    # --- Hierarchical kmeans clustering ---
    labels_3d = hierarchical_kmeans_split_c2(image, mask_arr > 0) # feed binary mask to avoid clustering on background
    res = np.where(mask_arr > 0, labels_3d, 0)  # zero out non-lung voxels
    if 'q' in mtype or 'p' in mtype:
        print(f"QDP %: {np.count_nonzero(labels_3d == 1) / np.count_nonzero(labels_3d>0):.2%}")
    else:
        print(f"VDP %: {np.count_nonzero(labels_3d == 1) / np.count_nonzero(labels_3d>0):.2%}")

    if outdir:
        # --- Write Image ---
        labels_3d_img = sitk.GetImageFromArray(res.astype(np.uint8))
        labels_3d_img.CopyInformation(mask_lung)  # copy spacing, origin, direction from reference image
        if 'q' in mtype or 'p' in mtype:
            sitk.WriteImage(labels_3d_img, os.path.join(outdir,'clustersQ.nii.gz'))
        else:
            sitk.WriteImage(labels_3d_img, os.path.join(outdir,'clustersV.nii.gz'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical K-means clustering for 3D numpy array")
    parser.add_argument("--map", type=str, help="Path to folder containing ventilation/perfusion map .nii.gz")
    parser.add_argument("--mask-lung", type=str, help="Path to mask lung .nii.gz file")
    parser.add_argument("--mask-vessels", type=str, help="Path to mask vessels .nii.gz file")
    parser.add_argument("--type", choices=['v','vent','q','perf'], help="Specify map type for specific processing steps.")
    parser.add_argument("--o", type=str, required=False, help="Directory to save outputs (median, log map, cluster NifTis)")
    args = parser.parse_args()
    img = args.map
    mask_path_lung = args.mask_lung
    mask_path_vessel = args.mask_vessels
    mtype= args.type
    outdir = args.o
    main()
