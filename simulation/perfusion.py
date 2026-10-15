#!/usr/bin/env python3
import os, time, argparse
import numpy as np
from aether.diagnostics import set_diagnostics_on
from aether.indices import perfusion_indices, get_ne_radius
from aether.pressure_resistance_flow import evaluate_prq
from aether.geometry import define_node_geometry, define_1d_elements, define_init_volume, append_units, add_matching_mesh, define_rad_from_geom
# from aether.geometry import set_rad_upstream_filtered_elem, import_ply_triangles, filter_elems_in_ply # map-with-ply branch
from aether.exports import export_1d_elem_geometry, export_node_geometry, export_terminal_perfusion, export_1d_elem_field, export_node_field
from tools import total_elems_in_ipelem, get_max_order, adjust_strahler_ratio

def read_element_values(filepath):
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith("Values:"):
                # Read next line immediately (contains the numbers)
                next_line = next(f, '').strip()
                if next_line:
                    parts = next_line.split()
                    values.append(float(parts[-1]))
    return values

def read_mpap(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith("Node:"):
                # Read next line immediately (contains the numbers)
                next_line = next(f, '').strip()
                if next_line:
                    parts = next_line.split()
                    return float(parts[-1])

def sample_element_numbers(filepaths, n_samples=100, seed=None):
    if seed is not None:
        np.random.seed(seed)

    samples = []

    for filepath in filepaths:
        with open(filepath, 'r') as f:
            lines = f.readlines()[1:]  # skip header line

        # Extract the 2nd column (Terminal element number)
        terminal_elements = [int(line.split()[1]) for line in lines if line.strip()]

        # Sample up to n_samples per file
        if len(terminal_elements) > n_samples:
            sampled = np.random.choice(terminal_elements, size=n_samples, replace=False)
        else:
            sampled = np.array(terminal_elements, dtype=int)

        samples.append(sampled)

    # Flatten all samples into one array
    return np.concatenate(samples)

def main():
    parser = argparse.ArgumentParser(description="Simulate steady-state perfusion")
    parser.add_argument("--dir-tree","-d",help="Directory containing 1D ipnode/elem/fiel files.")
    parser.add_argument("--init-vol","-vol",default=None,help="Filepath of init_vol.ipfiel (acini FRC volumes)")
    parser.add_argument("--rad-mpa","-ra",type=float,help="Image-derived MPA radius (mm)")
    parser.add_argument("--rad-ven","-rv",type=float,help="Image-derived average vein radius (mm)")
    parser.add_argument("--co","-q",type=float,default=4.0,help="Average cardiac output (L/min)")
    parser.add_argument("--mpap",type=float,default=15.0,help="Mean pulmonary artery pressure (mmHg)")
    parser.add_argument("--lap",type=float,default=8.0,help="Mean left atrial pressure (mmHg)")
    parser.add_argument("--map","-m",default=None,required=False,help="Filepath of PLY files of defect regions")
    parser.add_argument("--outdir","-o",default=None,required=False,help="Output directory of simulation files.")
    args = parser.parse_args()

    input_dir = args.dir_tree
    init_vol = args.init_vol
    mpa_rad_strained = args.rad_mpa
    ave_measured_vein_radius = args.rad_ven
    co = args.co
    mpap = args.mpap
    lap = args.lap
    mapp = args.map
    output_dir = args.outdir
    scale_adjust_art = 1.1
    scale_adjust_ven = 0.5
    
    # Set up directories
    if output_dir:
        os.makedirs(output_dir,exist_ok=True)

    set_diagnostics_on(False)
    
    # Read params_evaluate_flow
    # print(f' Subject {cfg.subject}')

    #########################################################################

    perfusion_indices()  # setting up model indices

    # Read in geometry files
    define_node_geometry(os.path.join(input_dir, 'artery_full'))
    define_1d_elements(os.path.join(input_dir, 'artery_full'))
    last_elem = total_elems_in_ipelem(input_dir,'artery_full')
    print(f" Total number of elements in artery_full: {last_elem}")

    append_units()
    add_matching_mesh()

    # Export 1D geometry
    group_name = 'perf_model'
    filename = output_dir + '/Arterial_tree_full.exelem'
    export_1d_elem_geometry(filename, group_name)
    filename = output_dir + '/Arterial_tree_full.exnode'
    export_node_geometry(filename, group_name)

    # if mapp:
    #     import_ply_triangles(mapp)
    #     terminal_only = True 
    #     filter_elems_in_ply(terminal_only) # get capillary elems found in ply. NB: set False if want all elems in ply instead of just cap elems

    remodeling_grade = 0 # no remodelling

    ############################################
    # define radius by Strahler order
    alpha = 1.49*10**(-4) # from paper
    Ptm = mpap * 133.3 # (mmHg to Pascals). In arteries, Ptm = mPAP. In veins, Ptm = PWCP (left atrial pressure)

    # iteratively adjust s_ratio until suitable (strained) outlet radius found
    target_radius = 0.15 # mm (strained). terminal artery (strahler order 1) diameter 10-20 mm Horsfield 1978
    target_branch_order = 1 # Strahler order of pulmonary arteriole outlets
    max_order = get_max_order(input_dir,'artery_full') # Strahler order of main pulmonary artery inlet based on artery tree.exelem
    s_ratio, term_rad_strained = adjust_strahler_ratio(target_radius, target_branch_order, max_order, mpa_rad_strained, Ptm) # interpolate s_ratio btwn MPA inlet order and arteriole outlet order 1

    scale_factor = (1 + alpha * Ptm)*scale_adjust_art
    mpa_rad0 = mpa_rad_strained / scale_factor
    print(f" Scale factor to unstrain MPA radius: {scale_factor:.4f}")
    print(f" Artery s_ratio: {s_ratio:.2f}; MPA radius (strained): {mpa_rad_strained:.2f}; Terminal radius (strained): {term_rad_strained:.3f}")
    print(f" MPA radius (unstrained): {mpa_rad0:.2f}\n")

    # Define unstrained radii for artery tree
    order_system = 'strahler'
    order_options = 'all'
    name = 'inlet'
    define_rad_from_geom(order_system, s_ratio, name, mpa_rad0, order_options, '')

    # Subject measurements
    Ptm = lap * 133.3 # PWCP of normal children in Pascals (1 mmHg = 133.3 Pa)
    print(f" Measured ave vein radius (strained): {ave_measured_vein_radius:.2f} mm")

    # iteratively adjust s_ratio until suitable (strained) inlet radius found
    target_branch_order = 1 # Strahler order of phantom pulmonary vein root
    max_order = get_max_order(input_dir,'artery_full') # Strahler order of phantom pulmonary vein root based on artery tree
    vein_order = max_order-2 # real pulm veins are 2 orders below phantom pulm vein root
    s_ratio_ven, term_rad_strained = adjust_strahler_ratio(target_radius, target_branch_order, vein_order, ave_measured_vein_radius, Ptm) # return s_ratio and resultant terminal radius (strained)
    vein_rad = (ave_measured_vein_radius * vein_order**2) # extrapolate real pulm vein radius to estimate phantom pulm vein root radius
    scale_factor = (1 + alpha * Ptm)*scale_adjust_ven
    print(f" Scale factor to unstrain vein radius: {scale_factor:.4f}")
    vein_rad = ave_measured_vein_radius / scale_factor # unstrain phantom pulm vein root radius
    print(f" Vein root radius (unstrained): {vein_rad:.2f} mm")
    print(f" Vein s_ratio: {s_ratio_ven:.2f}; Ave vein radius (strained): {ave_measured_vein_radius:.2f}; Terminal radius (strained): {term_rad_strained:.3f}\n")

    order_system = 'strahler'
    order_options = 'list'
    define_rad_from_geom(order_system, s_ratio_ven, str(last_elem+1), vein_rad, order_options,str(last_elem*2))  # matched arteries
    # if mapp:
    #     set_rad_upstream_filtered_elem(0.5) # apply scale factor to radii of filtered elems (prereq: import_ply_triangles & filter_elems_in_ply)        
        
    # Call solve for steady model
    mesh_type = 'full_plus_ladder'
    vessel_type = 'elastic_alpha'
    grav_dirn = 2
    grav_factor = 1.0
    bc_type = 'flow' # 'flow' (flow at inlet (Cardiac Output), pressure at outlet (left atrial pressure)). 'pressure' (at inlet and outlets)
    inlet_bc = co *10**6/60 # convert from l/min to mm3/s. Children CO: 3-5 L/min/m2.
    print(f" Cardiac output (inlet flow): {inlet_bc:.2f} mm3/s")
    outlet_bc = lap * 133.3 # PWCP of normal children in Pascals (1 mmHg = 133.3 Pa)

    if init_vol:
        define_init_volume(init_vol) # initialise volume at FRC to estimate Ppl at units
    start = time.time()
    evaluate_prq(mesh_type, vessel_type, grav_dirn, grav_factor, bc_type, inlet_bc,outlet_bc,remodeling_grade)
    print(f" Solved in {round((time.time()-start)/60)} minutes\n")

    # export element field for radius (strained radius)
    field_name = 'radius_perf'
    ne_radius = get_ne_radius()
    filename = output_dir + '/Artery_radius_perf.exelem'
    export_1d_elem_field(ne_radius, filename, group_name, field_name)

    # export flow element
    filename = output_dir + '/Artery_flow_perf.exelem'
    field_name = 'flow'
    export_1d_elem_field(7, filename, group_name, field_name)

    # export flow element
    filename = output_dir + '/Artery_resistance.exelem'
    field_name = 'resistance'
    export_1d_elem_field(8, filename, group_name, field_name)

    # export node field for pressure
    filename = output_dir + '/Artery_pressure_perf.exnode'
    field_name = 'pressure_perf'
    export_node_field(1, filename, group_name, field_name)

    # Export time-averaged terminal solution
    filename = os.path.join(output_dir, 'terminal_perfusion.exnode')
    export_terminal_perfusion(filename, 'perf_model')

    #######################################
    # Check predicted MPA radius against measured MPA radius
    rad_pred = read_element_values(f'{output_dir}/Artery_radius_perf.exelem')
    mpa_pred = rad_pred[0]
    terminal_radii = rad_pred[last_elem-5:last_elem]
    print(f" Mean terminal radii: {sum(terminal_radii)/len(terminal_radii)}")
    ven_rad = rad_pred[last_elem+7:last_elem+11]
    print(f" Mean vein radii: {sum(ven_rad)/len(ven_rad)}")
    mpap_pred = read_mpap(f'{output_dir}/Artery_pressure_perf.exnode')
    print(f" Predicted MPA radius: {mpa_pred:.2f} mm")
    print(f" Predicted MPAP: {mpap_pred} Pa = {mpap_pred/133.3:.1f} mmHg")
    print(f" Predicted terminal radii: {terminal_radii} mm")
    print(f" Predicted vein radii: {ven_rad} mm")


if __name__ == '__main__':
    main()
