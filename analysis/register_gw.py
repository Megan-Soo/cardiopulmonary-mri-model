"""Extracted from https://github.com/PulmonaryMRI/MoCoLoR
"""
#!/usr/bin/env python3
import numpy as np
import reg
import os, argparse, re
import SimpleITK as sitk

def demons_reg(qt, n_ref, out):
    print('Jacobian Determinant and Specific Ventilation...')
    jacs = []
    svs = []
    qt = np.abs(np.squeeze(qt))
    qt = qt/np.max(qt)
    for i in range(qt.shape[0]):
        jac, sv = reg.ANTsJac(np.abs(qt[n_ref]), np.abs(qt[i]),i+1, out) # (MS) added: pass phase to save the registration results
        jacs.append(jac)
        svs.append(sv)
    jacs = np.asarray(jacs)
    svs = np.asarray(svs)
    return jacs, svs

def natural_key(s):
    """
    Split string into list of text and integer chunks for natural sorting.
    e.g. 'img10.nii.gz' -> ['img', 10, '.nii.gz']
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read gradwarped NifTis and perform Demons registration to generate Jac and SV images")
    parser.add_argument("dir", type=str, help="Dir containing struct img NiFTis")
    parser.add_argument("--o", type=str, help="Output directory")
    parser.add_argument("--n_ref", type=str, default=0, help="Reference frame")
    args = parser.parse_args()
    dir = args.dir
    n_ref = args.n_ref
    outdir = args.o
    os.makedirs(outdir, exist_ok=True)

    # Read gradwarped structural img niftis
    image_files = sorted(
        [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(".nii.gz")],
        key=natural_key
    )
    # print(image_files)

    arrs = [] # Stack into np array
    for f in image_files:
        img = sitk.ReadImage(f)
        arrs.append(sitk.GetArrayFromImage(img))
    arrs = np.array(arrs)
    # print(arrs.shape)

    jacs, svs = demons_reg(arrs, n_ref, outdir) # Demons registration on gradwarped img arrays

    np.savez_compressed(os.path.join(outdir, 'gradwarp_jac_vent.npz'), jacs) # (MS) added: compressed npz saves space
    np.savez_compressed(os.path.join(outdir, 'gradwarp_sv_vent.npz'), svs) # (MS) added: compressed npz saves space

    # save NifTis
    for i,(j,s) in enumerate(zip(jacs,svs)):
        j=sitk.GetImageFromArray(j)
        j.CopyInformation(img)
        sitk.WriteImage(j,os.path.join(outdir,f'jac{i}.nii.gz'))
        print(f"Exported jac{i}.nii.gz")

        s=sitk.GetImageFromArray(s)
        s.CopyInformation(img)
        sitk.WriteImage(j,os.path.join(outdir,f'sv{i}.nii.gz'))
        print(f"Exported sv{i}.nii.gz")
