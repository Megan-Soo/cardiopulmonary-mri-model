#!/usr/bin/env python3
import numpy as np
import os,re, argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk

def read_exnodedata(file_path,extn='.exnode'): # returns dict with key:value - (field name, num Components):[array of field values]
    try:
        file_path, ext = os.path.splitext(file_path)
        if bool(ext) is False:
            ext = extn

        with open((file_path+ext), 'r') as file:
            lines = file.readlines()

        results = {}  # Dictionary to store the results
        for line in lines:
            if ')' in line:
                # Find the closing parenthesis
                close_paren_index = line.find(')')
                # Find the next comma after the closing parenthesis
                next_comma_index = line.find(',', close_paren_index)
                if next_comma_index != -1:
                    # Extract the substring between ')' and ','
                    words_between = line[close_paren_index + 1:next_comma_index].strip()
                else:
                    words_between = line[close_paren_index + 1:].strip()  # If no comma, get till the end

                # Look for "Components=" and check the subsequent character
                components_index = line.find("Components=")
                if components_index != -1:
                    start_index = components_index + len("Components=")
                    if start_index < len(line):  # Ensure there's a character after "Components="
                        subsequent_char = line[start_index]
                        if subsequent_char.isdigit():
                            # Create the key as a tuple (word, integer)
                            key = (words_between, int(subsequent_char))
                            # Add the tuple as a key to the dictionary with a placeholder value
                            results[key] = None  # Placeholder value; replace with desired value

        ## Collect and store index of each node
        indices = [i for i, s in enumerate(lines) if 'Node' in s]
        int_pattern = r"\b\d+\b" # Regex pattern to match integer numbers
        float_int_pattern = r'[+-]?\d+(?:\.\d+)?' # Regex pattern to match float or integer numbers

        list_node_num = []
        for idx in range(len(indices)):
            line = lines[indices[idx]]
            node_number = int(re.findall(int_pattern, line)[-1])
            list_node_num.append(node_number)

        prev_num_components = 0
        iterator = iter(results.items())
        for field in range(len(results)):
            entry = next(iterator)
            key = entry[0]
            components = entry[0][1]
            # print("Key (field name, components):", key) # field name, components
            # print("Components:", components)

            my_list = []
            for idx in range(len(indices)):
                for i in range(components):
                    line = lines[indices[idx]+1+prev_num_components+i]
                    value = float(re.findall(float_int_pattern, line)[0])
                    my_list.append(value)

            if components>1:
                # Reshape into a 2D list
                my_list = [my_list[i:i+components] for i in range(0, len(my_list), components)]

            results[key] = my_list

            prev_num_components = prev_num_components + components

        results[("Node number",None)] = list_node_num
        return results

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

def bin_map(signal_map, n_bins, axis=1, mode="None"):
    """
    Split a 3D signal map into n_bins along the y axis and return the
    fraction of the non-zero signal that falls in each bin.

    Parameters
    ----------
    signal_map : 3D array-like
    n_bins     : number of bins along y
    axis       : which array axis is y (default 1, i.e. shape (x, y, z))
    mode       : "None" -> total signal magnitude in each bin
                 "frac"   -> fraction of total signal magnitude in each bin

    Returns
    -------
    fractions : (n_bins,) array that sums to 1 (all zeros if the map is empty)
    """
    arr = np.moveaxis(np.asarray(signal_map), axis, 0)   # y first
    flat = arr.reshape(arr.shape[0], -1)

    nonzero_count = np.count_nonzero(flat, axis=1)
    keep = nonzero_count > 0                             # slices with any signal
    n_kept = int(keep.sum())

    if n_kept == 0:
        return np.zeros(n_bins)
    if not 1 <= n_bins <= n_kept:
        raise ValueError(f"n_bins must be between 1 and {n_kept} "
                         f"(number of non-empty y slices)")

    per_slice = np.abs(flat[keep]).sum(axis=1)

    totals = np.array([b.sum() for b in np.array_split(per_slice, n_bins)], dtype=float)
    if mode=="frac":
        return totals / totals.sum()
    else:
        return totals

def miget_style(vq,v,q,n_bins=50):
    """
    vq: 1D Numpy array
    v: 1D Numpy array
    q: 1D Numpy array
    """
    _, bin_edges = np.histogram(vq, bins=n_bins, range=(0, 5), density=True)

    # Get bin indices for each vq value
    bin_idx = np.digitize(vq, bin_edges) - 1   # convert to 0-based indexing

    # Number of bins
    n_bins = len(bin_edges) - 1

    # Remove values outside the defined bins
    valid = (bin_idx >= 0) & (bin_idx < n_bins)

    # Total ventilation and perfusion in one cycle in each bin (mm3/s)
    v_total = np.bincount(bin_idx[valid], weights=v[valid], minlength=n_bins)
    q_total = np.bincount(bin_idx[valid], weights=q[valid], minlength=n_bins)
    # V/Q ratio per bin
    vq_total = v_total / q_total # essentially the same as bin_centers. nice.
    return v_total, q_total, vq_total

def main():
    # region:parser
    parser = argparse.ArgumentParser(description="Plot V, Q, and V/Q across lung height.")
    parser.add_argument("-v",help="Path to normalised ventilation map")
    parser.add_argument("-q",help="Path to normalised perfusion map")
    parser.add_argument('-vt',type=float,help="Tidal Volume (L)")
    parser.add_argument('-co',type=float,default=4.0,help="Cardiac output (L/min)")
    parser.add_argument('-rr',type=float,help="Respiratory rate (bpm)")
    parser.add_argument('-n_bins',type=int,default=10,help="Number of bins")
    parser.add_argument('-o',type=str,default=None,help="Export directory (parameterised vq_map.nii.gz & vq-plot.png)")
    parser.add_argument("--add-model",action="store_true",help="Add simulation results to plot")
    args = parser.parse_args()
    path_vent = args.v
    path_perf = args.q
    vt = args.vt
    co = args.co
    rr = args.rr
    n_bins = args.n_bins
    outdir = args.o
    add_model = args.add_model
    #endregion

    subject = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path_perf))))
    print(f"Subject {subject}")

    # Read maps
    im = sitk.ReadImage(path_vent) # keep one image object for write out
    vent = sitk.GetArrayFromImage(im)
    perf = sitk.GetArrayFromImage(sitk.ReadImage(path_perf))

    #region: parameterise maps
    # Scale vent to Tidal Vol
    vent_sum = np.sum(vent[vent!=0]) # background=0
    v_rate = vt*rr # L/min
    scale_factor = v_rate/vent_sum
    vent = vent*scale_factor

    # Scale perf to blood vol
    perf_sum = np.sum(perf[perf!=0]) # background=0
    scale_factor = co/perf_sum
    perf = perf*scale_factor

    # Get V/Q map
    vq = np.where(perf>0,vent/perf,0)
    res = sitk.GetImageFromArray(vq)
    res.CopyInformation(im)
    if outdir: sitk.WriteImage(res,os.path.join(outdir,'vq_map.nii.gz'))
    idx = np.argwhere(vent>0) # get nonzero indices
    vq_1d = vq[tuple(idx.T)] # 1d array
    vent_1d = vent[tuple(idx.T)]
    perf_1d = perf[tuple(idx.T)]
    #endregion

    # region: plot1 data
    # bin maps
    vent_bins = bin_map(vent,n_bins,mode="frac")
    perf_bins = bin_map(perf,n_bins,mode="frac")
    vq_bins = vent_bins/perf_bins
    #endregion

    # region: plot2 data
    # Split units by V/Q bins
    v_total, q_total, vq_total = miget_style(vq_1d,vent_1d,perf_1d,n_bins=50)
    # endregion

    #region: add_model
    if add_model:
        import pandas as pd
        # read terminal ventilation
        vent_dict = read_exnodedata(f'results/{subject}/model/term_average.exnode')
        # read terminal perfusion
        perf_dict = read_exnodedata(f'results/{subject}/model/terminal_perfusion.exnode')
        # bin values
        coords = np.array(vent_dict[('coordinates',3)])
        # % lung height from posterior (max) to anterior (min) (RAS)
        lung_height = np.max(coords[:,1]) - np.min(coords[:,1])
        print(f'Subject: {subject}, Lung Height (mm): {lung_height:.2f}')
        perc_height = (coords[:,1] - np.min(coords[:,1])) / lung_height * 100
        # perc_height = 100 - perc_height # flip: smallest y (posterior): 0%, largest y (anterior): 100%
        # no need flip bc np.linspace from 95% to 5%
        
        # V flow
        vent_model = np.array(vent_dict[('flow',1)])/10**-6 # mm3 to L
        vent_model = vent_model * rr # L/min
        vent_total_model = np.sum(vent_model)
        # norm_vent = (vent_model-np.min(vent_model)) / (np.max(vent_model)-np.min(vent_model))
        norm_vent = vent_model-np.mean(vent_model)

        # Q flow
        perf_model = np.array(perf_dict[('flow',1)])/10**-6 # mm3 to L
        perf_model = perf_model * co # L/min
        perf_total_model = np.sum(perf_model)
        # norm_perf = (perf_model - np.min(perf_model)) / (np.max(perf_model) - np.min(perf_model))
        norm_perf = perf_model/np.mean(perf_model)

        # VQ (normalised?)
        # vent_model[vent_model<0.1] = 0.05 # avoid 0 V/Q ratio. set zero ventilation to v low value
        # perf_model[perf_model<0.1] = 0.05 # avoid inf or NAN values in V/Q ratio. set zero flow to v low value
        vq_model = vent_model/perf_model

        # use pandas to bin data
        df = pd.DataFrame({
            "perc_height": perc_height,
            "norm_vent": norm_vent,
            "norm_perf": norm_perf,
            "vent": vent_model,
            "perf": perf_model
        })
        df["bin"] = pd.cut(df["perc_height"], bins=n_bins)

        # 2. Compute mean and SD per bin
        vent_bins_model = df.groupby("bin",observed=False)["vent"].sum()#.reset_index() # don't reset index to allow operation
        vent_bins_model = vent_bins_model/vent_total_model # fractional perfusion
        perf_bins_model = df.groupby("bin",observed=False)["perf"].sum()#.reset_index()
        perf_bins_model = perf_bins_model/perf_total_model # fractional perfusion
        vq_bins_model = vent_bins_model/perf_bins_model

        v_total_model, q_total_model, vq_total_model = miget_style(vq_model,vent_model,perf_model,n_bins=50)

    #endregion

    # region: Plot
    lung_height = np.linspace(95,5,10) # Anterior->Posterior
    _,axs = plt.subplots(1,2)
    ax1=axs[0]
    v=ax1.plot(lung_height,vent_bins,color='blue',label='Ventilation')
    q=ax1.plot(lung_height,perf_bins,color='red',label='Perfusion')
    ax1.set_ylabel("Total V or Q (L/min)")
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,100)
    ax1.set_xlabel('Lung height % (Posterior -> Anterior)')
    ax1_2 = ax1.twinx()
    vqline=ax1_2.plot(lung_height, vq_bins,color='black',label='V/Q')
    ax1_2.set_ylabel("V/Q")
    ax1_2.set_ylim(0,3)
    # combined legend
    lines = v+q+vqline
    labels = [l.get_label() for l in lines]

    ax2 = axs[1]
    ax2.axvline(x=vq_total[0], ymin=0,ymax=v_total[0], marker='o', color='blue',markerfacecolor='none',linestyle='solid',markevery=[1],linewidth=2) # marker only at the top (1st index)
    ax2.axvline(x=vq_total[0], ymin=0,ymax=q_total[0],marker='o', color='r',linestyle='solid',markevery=[1],linewidth=2)
    ax2.plot(vq_total[1:], v_total[1:],'o', color='blue',markerfacecolor='none',linestyle='solid',linewidth=2)
    ax2.plot(vq_total[1:], q_total[1:], 'o', color='r',linestyle='solid',linewidth=2)
    # ax2.set_ylim(0,1)
    ax2.set_xlim(0.01,100)
    ax2.set_xscale('log')
    ax2.grid(axis='y')
    ax2.set_ylabel("V or Q (L/min) per lung unit")
    ax2.set_xlabel("V/Q")

    if add_model:
        v=ax1.plot(lung_height,vent_bins_model,"--",color='blue',label='Ventilation')
        q=ax1.plot(lung_height,perf_bins_model,"--",color='red',label='Perfusion')
        ax1.plot(lung_height, vq_bins_model,'--',color='black')
        # combined legend
        lines = lines+ v+q+vqline
        labels = labels + [l.get_label() for l in lines]

        ax2.axvline(x=vq_total_model[0], ymin=0,ymax=v_total[0], linestyle='--', marker='s', color='blue',markevery=[1],linewidth=2) # marker only at the top (1st index)
        ax2.axvline(x=vq_total_model[0], ymin=0,ymax=q_total[0],linestyle="--", marker='s',color='r',markevery=[1],linewidth=2)
        ax2.plot(vq_total_model[1:], v_total_model[1:],'s', color='blue',markerfacecolor='none',linestyle='solid',linewidth=2)
        ax2.plot(vq_total_model[1:], q_total_model[1:], 's', color='r',linestyle='solid',linewidth=2)

    ax1.legend(lines,labels)
    plt.tight_layout()
    if outdir: plt.savefig(os.path.join(outdir,'vq-plot.png'))
    plt.show()

if __name__=="__main__":
    main()