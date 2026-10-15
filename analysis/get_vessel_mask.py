#!/usr/bin/env python3
import SimpleITK as sitk
from skimage.morphology import ball, white_tophat
from scipy import ndimage
import numpy as np
import os, argparse, time

def apply_white_tophat(unfiltered: np.ndarray, mask:np.ndarray, filter_nbhood: int, thresh:int) -> np.ndarray:
    """
    Filtering determined by structural element.
    Finds bright spots and small details smaller than a structuring element by subtracting an image's morphological opening from the original image.
    """
    if len(unfiltered.shape)==2:
        print('2D white tophat')
        from skimage.morphology import disk
        enhanced = white_tophat(unfiltered,disk(filter_nbhood))
    else:
        enhanced = white_tophat(unfiltered, ball(filter_nbhood))
    # --- threshold ---
    vals = enhanced[mask > 0]
    thresh = np.percentile(vals, thresh)
    bright_mask = enhanced > thresh
    return bright_mask

def apply_enhanced_mean(unfiltered: np.ndarray, mask:np.ndarray, filter_nbhood: int, thresh:int) -> np.ndarray:
    """
    Filtering determined by neighbourhood.
    """
    local_mean = ndimage.uniform_filter(unfiltered.astype(np.float32), size=filter_nbhood)
    enhanced_mean = unfiltered - local_mean
    enhanced_mean[enhanced_mean < 0] = 0 # filter out values above neighbourhood mean
    # --- threshold ---
    vals_mean = enhanced_mean[mask > 0] # get vessel voxels, toss body voxels
    thresh_mean = np.percentile(vals_mean, thresh)
    bright_mask_mean = enhanced_mean > thresh_mean # get vessel voxels above percentile threshold
    return bright_mask_mean

def propagate_labels(partial_labels: np.ndarray, mask: np.ndarray, radius=1, max_iters=None) -> np.ndarray:
    """
    Given a partially labelled mask, propagate each label individually within a full binary mask using iterative dilation with specified kernel size.

    Parameters
    ----------
    partial_labels : ndarray (int)
        Partially labeled array (0 = unlabeled)
    mask : ndarray (binary)
        Vessel mask (1 = inside vessel, 0 = outside)
    radius : int
        Radius for the structuring element (1 = small neighborhood)
    max_iters : int or None
        Maximum number of dilation iterations per label

    Returns
    -------
    output : ndarray (int)
        Array with propagated labels
    """

    # Make sure input arrays are integer/binary
    mask = (mask > 0).astype(np.uint8)
    partial_labels = partial_labels.astype(np.int32)

    # structuring element for dilation
    ndim = mask.ndim
    structure = ndimage.generate_binary_structure(ndim, 1)  # 6-connectivity in 3D, 4-connectivity in 2D
    if radius > 1:
        # iterate dilation on the structuring element itself to enlarge
        selem = structure.copy()
        for _ in range(radius-1):
            selem = ndimage.binary_dilation(selem, structure)
        structure = selem

    output = np.zeros_like(partial_labels, dtype=np.int32)
    unique_labels = np.unique(partial_labels)
    unique_labels = unique_labels[unique_labels > 0]

    for lbl in unique_labels:
        # Seed mask for this label
        region = (partial_labels == lbl)
        grown = region.copy()
        iteration = 0

        while True:
            iteration += 1
            # Dilate within mask
            dilated = ndimage.binary_dilation(grown, structure=structure)
            dilated = dilated & (mask > 0)
            new_voxels = dilated & (~grown)

            if not np.any(new_voxels):
                break
            grown[new_voxels] = True

            if max_iters is not None and iteration >= max_iters:
                break

        # Assign label only to voxels not yet labeled in output
        assignable = grown & (output == 0)
        output[assignable] = lbl

    return output

def clean_segmentation(image, min_size=100):
    """
    Reads a segmentation image, removes small connected components,
    relabels components by ascending size, and writes out the cleaned image.
    
    Parameters:
        input_path (str): Path to input segmentation image.
        output_path (str): Path to save cleaned segmentation image.
        min_size (int): Minimum component size to keep.
    """
    
    # Ensure the image is labeled (integer values)
    image = sitk.Cast(image, sitk.sitkUInt32)
    
    # Connected component analysis
    cc = sitk.ConnectedComponent(image)
    
    # Remove small components
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)  # Sort descending by size
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    
    # Create a mask for components larger than min_size
    large_mask = sitk.Image(cc.GetSize(), sitk.sitkUInt32)
    large_mask.CopyInformation(cc)
    
    for label in stats.GetLabels():
        if stats.GetPhysicalSize(label) >= min_size:
            large_mask = large_mask | sitk.Cast(cc == label, sitk.sitkUInt32) * 1 #*1 for binary, otherwise *label
    
    # Relabel components by ascending size
    # cleaned = sitk.RelabelComponent(large_mask, sortByObjectSize=False)
    return large_mask

def binary_erosion(image, radius=2,out=None):
    """
    Apply binary erosion to a 3D SimpleITK image.

    Parameters:
    - image (sitk.Image): Input 3D volumetric image.
    - radius (int or tuple): Radius of the binary erosion kernel. Can be an integer or a 3D tuple.

    Returns:
    - sitk.Image: Filtered 3D image.
    """
    erosion_filter = sitk.BinaryErodeImageFilter()
    erosion_filter.SetKernelRadius([radius,radius,radius])
    eroded = erosion_filter.Execute(image)
    eroded.CopyInformation(image)
    if out is not None:
        sitk.WriteImage(eroded,out)
    return eroded

def main():
    parser = argparse.ArgumentParser(description="Filter bright voxels from raw image, filter lung voxels w/ mask, and dilate w/ kernel==1 to form complete vessel feature.")
    parser.add_argument("-image", required=True, help="Filepath of lung image (RAS)")
    parser.add_argument("-mask", required=True, help="Path to RAS lung mask file (e.g. closed_mask.nii.gz). Lung mask to remove high intensities from non-lung tissue voxels.")
    parser.add_argument("-o", required=False, help="Path to save the vessel mask")
    parser.add_argument("-nbhood", type=int, default=10, help="Radius of bright filter neighborhood (default: 10). Low res-> small radius (vice versa)")
    parser.add_argument("-min_size", type=int, default=1000, help="Clean up unconnected components after filtering")
    parser.add_argument("-thresh", type=int, default=90, help="Percentile threshold for bright voxel filtering (default: 90)")
    args = parser.parse_args()

    # === Read files ===
    image = sitk.ReadImage(args.image) #Reader(args.image,keyword=args.keyword).image
    mask0 = sitk.ReadImage(args.mask) # Read lung mask
    mask0 = sitk.BinaryThreshold(mask0, lowerThreshold=1, upperThreshold=2, insideValue=1, outsideValue=0)

    mask = binary_erosion(mask0,radius=3) # remove lung periphery

    # === Filter bright voxels ===
    mask_white_tophat = apply_white_tophat(sitk.GetArrayFromImage(image),sitk.GetArrayFromImage(mask),args.nbhood,args.thresh) # Apply white tophat filter
    mask_white_tophat = np.where(sitk.GetArrayFromImage(mask)>0,mask_white_tophat,0)
    mask_enhanced_mean = apply_enhanced_mean(sitk.GetArrayFromImage(image),sitk.GetArrayFromImage(mask),args.nbhood,args.thresh) # Apply enhanced mean filter
    mask_enhanced_mean = np.where(sitk.GetArrayFromImage(mask)>0,mask_enhanced_mean,0)
    mask_vessel = mask_white_tophat + mask_enhanced_mean # Combine and binarise resulting masks

    # Complete the vessel feature using small kernel dilation
    # start=time.time()
    # mask_vessel = propagate_labels(mask_vessel0, mask_vessel0)
    # print(f"Propagating labels took {round(time.time()-start)} seconds")
    if np.count_nonzero(mask_vessel)==0:
        print(f"No vessel feature after propagation")
        exit()

    # Export vessel mask image
    output_image = sitk.GetImageFromArray(mask_vessel)
    output_image.CopyInformation(mask)

    output_image = clean_segmentation(output_image,min_size=args.min_size) # added clean small components

    if args.o:
        try:
            os.makedirs(os.path.dirname(args.o), exist_ok=True)
        except: # makedirs fails if current directory specified
            pass # skip if output in current dir
        sitk.WriteImage(output_image, args.o)
        print(f"Vessel mask saved to: {args.o}")

        steps = {
                "1. Original lung mask":"Binarises labels<=2. Works either when both lungs are assigned 1 or when lungs are assigned 1 and 2.",
                "2. Erode peripheral lung":"sitk.Ball with radius 3 removes the outer lung voxels",
                "3. Apply white tophat filtering":"Filtering determined by structural element. Finds bright spots and small details smaller than a structuring element by subtracting an image's morphological opening from the original image.",
                "4. Apply enhanced mean filtering":"Filtering determined by neighbourhood size.",
                "5. Combine filtered results":"White tophat + enhanced mean",
                "6. Label propagation":"Dilate a little to try to rejoin small broken connections",
                "7. Remove small unconnected components":f"Components <{args.min_size} voxels are removed."
                }

        # # save arrays for step-by-step visualisation
        # np.savez_compressed('visualise_pipeline/get_vessel_mask.npz',
        #                     steps=steps,
        #                     step1=sitk.GetArrayFromImage(mask0),
        #                     step2=sitk.GetArrayFromImage(mask),
        #                     step3=mask_white_tophat,
        #                     step4=mask_enhanced_mean,
        #                     step5=mask_vessel0,
        #                     step6=mask_vessel,
        #                     step7=sitk.GetArrayFromImage(output_image),
        #                     spacing=image.GetSpacing())

if __name__ == "__main__":
    main()