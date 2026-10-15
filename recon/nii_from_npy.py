#!/usr/bin/env python3
import numpy as np
import os
import SimpleITK as sitk # (MS) added
import argparse # (MS) added

def make_isotropic(ims, size):
    """
    Crop and/or pad a 3D or 4D array symmetrically so the spatial dims
    become (size, size, size).

    3D input: (X, Y, Z)
    4D input: (N, X, Y, Z)

    Output is always 4D: (N, size, size, size)
    """
    target = (size, size, size)

    # Normalize to 4D
    if ims.ndim == 3:
        ims = ims[np.newaxis, ...]
    elif ims.ndim != 4:
        raise ValueError("ims must be 3D or 4D")

    spatial = ims.shape[1:]

    if spatial == target:
        print(f"Array matches target size {target}, no padding/cropping done.")
        return ims

    # ----- Crop first -----
    slices = [slice(None)]  # keep batch dimension

    for dim, tgt in zip(spatial, target):
        if dim > tgt:
            start = (dim - tgt) // 2
            end = start + tgt
            slices.append(slice(start, end))
        else:
            slices.append(slice(None))

    ims = ims[tuple(slices)]

    # ----- Then pad if needed -----
    pads = []
    for dim, tgt in zip(ims.shape[1:], target):
        if dim < tgt:
            total = tgt - dim
            left = total // 2
            right = total - left
            pads.append((left, right))
        else:
            pads.append((0, 0))

    pad_width = [(0, 0)] + pads
    ims_out = np.pad(ims, pad_width, mode="constant", constant_values=0)

    print(f"Output shape: {ims_out.shape}")

    return ims_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate .nii.gz from .npy/.npz")
    parser.add_argument("f", type=str, help="Path to .npy file")
    parser.add_argument("orig_nii",type=str, help="Path to original .nii.gz")
    parser.add_argument("dir_out", type=str, help="Directory output")
    parser.add_argument("--fmri",type=int, default=1, help="1: Jac or SV. Set '0' if recon structural lung img from .npy. ")
    parser.add_argument("--mask", type=str, required=False, help="If recon functional lung img, add path to mask .nii.gz file")
    parser.add_argument("--flip_z", type=int, default=0, help="Flip z axis")
    parser.add_argument("--save_npz", type=str, help="Save fixed .npz with shape (phases,256,256,256)")
    parser.add_argument("--combine-nii", action="store_true", help="Combine into a single 4D NiFTi image.")
    args = parser.parse_args()
    f = args.f
    orig_nii = args.orig_nii
    outdir = args.dir_out # (MS) added

    # load original image
    orig_img = sitk.ReadImage(orig_nii)
    # load mask
    if args.mask:
        mask = sitk.ReadImage(args.mask)
        # erode 3 pixels       
        # radius = [3,3,3]
        # mask = sitk.BinaryErode(sitk.Cast(mask, sitk.sitkUInt8), radius, sitk.sitkBall)
        # mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=10000, insideValue=1, outsideValue=0)
        mask = sitk.GetArrayFromImage(mask)
        # assuming label 1 is lung mask, remove any other labels if present
        mask = np.where((mask == 1) | (mask==2), 1, 0)

    # load data
    filename = os.path.splitext(os.path.basename(f))[0]
    ims = np.load(f) # (MS) added
    if os.path.splitext(f)[-1]=='.npz':
        ims = ims['arr_0'] # if .npz, add step to extract array
    print(f"Input shape: {ims.shape}") # (MS) added: assuming recon to [256,256,256]

    ims = make_isotropic(ims, 256)
    ims_shape = np.shape(ims)
    if args.save_npz:
        np.savez_compressed(args.save_npz,ims)
        
    print(f"Pixel spacing: {orig_img.GetSpacing()}") # (MS) added

    ImagePositionPatient_original = orig_img.GetOrigin()
    print(f"Original Image Position Patient: {ImagePositionPatient_original}") # (MS) added

    # flip_z=True to flip images up-down
    if args.flip_z==1:
        ims = np.flip(ims,1)

    try:
        os.makedirs(outdir,exist_ok=True)
    except FileNotFoundError as e: # if outdir is current dir
        pass

    if args.combine_nii: # Export 4D nifti
        im4d = np.moveaxis(ims,0,-1) # make phases last axis [256,256,256,N]
        im4d = np.abs(im4d) / np.amax(np.abs(im4d)) * 4095
        im4d = im4d.astype(np.uint16)
        im4d = sitk.GetImageFromArray(im4d)
        im4d.SetSpacing(orig_img.GetSpacing())
        im4d.SetDirection(orig_img.GetDirection())
        im4d.SetOrigin(orig_img.GetOrigin())
        sitk.WriteImage(im4d,os.path.join(outdir,filename+'.nii.gz'))
    
    phases = ims_shape[0]
    combined = [] # storing all the RV maps to get mean volume change across all phases
    for ph in range(phases):
        im = ims[ph,]

        if args.fmri==0: # (MS) added. idky *4095 but it looks abritrary
            im = np.abs(im) / np.amax(np.abs(im)) * 4095  # 65535
            im = im.astype(np.uint16)
        else: # (MS) added. maybe don't do arb scaling for functional lung values
            im = im.astype(np.float32)
            if args.mask:
                im_masked = np.where(mask > 0, im, 0)  # background must be -1 for jac or else phase_1 will crash itksnap...
                im_masked = im_masked-1
                # combined.append(im) # collapse the RV maps to get mean volume change

        img = sitk.GetImageFromArray(im)
        img.CopyInformation(orig_img) # copy spacing, origin, direction from reference image
        sitk.WriteImage(img, os.path.join(outdir, f"{os.path.basename(os.path.normpath(outdir))}{ph}.nii.gz"))
        print(f"Saved {os.path.join(outdir, f'{os.path.basename(os.path.normpath(outdir))}{ph}.nii.gz')}")

        if args.mask:
            masked_out = os.path.join(outdir,'masked')
            os.makedirs(masked_out,exist_ok=True)
            img = sitk.GetImageFromArray(im_masked)
            img.CopyInformation(orig_img) # copy spacing, origin, direction from reference image
            sitk.WriteImage(img, os.path.join(masked_out, f"{os.path.basename(os.path.normpath(outdir))}{ph+1}.nii.gz"))
            print(f"Saved {os.path.join(masked_out, f'{os.path.basename(os.path.normpath(outdir))}{ph}.nii.gz')}")

    # # get mean volume change across all phases
    # if args.struct==0 and args.mask:
    #     combined = np.array(combined)
    #     mean_volume_change = np.sum(combined, axis=0)
    #     mean_volume_change_img = sitk.GetImageFromArray(mean_volume_change)
    #     mean_volume_change_img.CopyInformation(orig_img)
    #     sitk.WriteImage(mean_volume_change_img, os.path.join(dir_out, "mean_volume_change.nii.gz"))
