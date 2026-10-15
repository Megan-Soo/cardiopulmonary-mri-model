#!/usr/bin/env python3
import numpy as np
import SimpleITK as sitk
import argparse, os

def max_min_subtraction(arrs):
    """
    arrs: array containing 3D arrays of the same shape
    """

    # example: arrs = [arr1, arr2, arr3]

    # 1. Stack arrays along a new leading axis (creates a 4D array)
    if len(arrs.shape)!=4:
        stacked = np.stack(arrs, axis=0)
    else:
        stacked = arrs

    # 2. Compute elementwise max and min along the stacked axis
    max_vals = np.max(stacked, axis=0)
    min_vals = np.min(stacked, axis=0)

    # 3. Perform elementwise subtraction
    result = max_vals - min_vals

    return result

def main():
    parser = argparse.ArgumentParser(description="Get perfusion-weighted map by element-wise subtraction of max and min signals across cardiac phases.")
    parser.add_argument("--file","-f",required=False, help="Filepath (NifTi or Numpy) containing cardiac phase-resolved image volumes.")
    parser.add_argument("--dir","-d",required=False, help="Directory containing NifTis with keyword 'recon_cardiac'.")
    parser.add_argument("--out","-o",help="Output path of perfusion-weighted map (NifTi or Numpy).")
    parser.add_argument("--mask",help="(optional) Filepath to lung mask nifti to export masked PW map.")
    parser.add_argument("--apply",action="store_true",help="If not apply, mask will only be used as a reference image for NifTi output where applicable.")
    args = parser.parse_args()

    f = args.file
    d = args.dir
    out = args.out
    mask = args.mask
    apply = args.apply

    # load data
    if f:
        ext = os.path.splitext(f)[-1]
        if '.np' in ext:
            ims = np.load(f) # (MS) added
            if ext=='.npz':
                ims = ims['arr_0'] # if .npz, add step to extract array
        else:
            im = sitk.ReadImage(f)
            ims = sitk.GetArrayFromImage(im)
            # Move the last axis (-1) to the first position (0)
            ims = np.moveaxis(ims, -1, 0)
    elif d:
        ims = []
        files=[os.path.join(d,f) for f in os.listdir(d) if 'recon_cardiac' in f]
        for f in files:
            ims.append(sitk.GetArrayFromImage(sitk.ReadImage(f)))
        ims = np.asarray(ims).astype(np.float32)

    print(f"Image shape: {ims.shape}")

    res = max_min_subtraction(ims)

    if mask:
        mask_im = sitk.ReadImage(mask)
        mask = sitk.GetArrayFromImage(mask_im)
    if apply:
        res = np.where(mask>0,res,0)
        
    outext = os.path.splitext(out)[-1]
    if '.np' in outext:
        np.savez_compressed(out,res)
    else:
        res = sitk.GetImageFromArray(res)
        res.SetSpacing(mask_im.GetSpacing())
        res.SetDirection(mask_im.GetDirection())
        res.SetOrigin(mask_im.GetOrigin())
        sitk.WriteImage(res,out)

if __name__=="__main__":
    main()    
