#!/usr/bin/env python3
import polyscope as ps
import numpy as np
import os,re,argparse

def read_terminal_dvdt(filepath):
    node_data = {}
    current_node = None
    current_values = []

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Node:"):
                if current_node is not None:
                    node_data[current_node] = current_values
                current_node = int(line.split()[-1])
                current_values = []
            elif line:
                current_values.extend(map(float, line.split()))
        
        # Don't forget to save the last node's data
        if current_node is not None:
            node_data[current_node] = current_values

    return node_data

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
        float_pattern = r"-?\d+\.\d+" # Regex pattern to match float numbers
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
                    try:
                        value = float(re.findall(float_int_pattern, line)[0])
                        my_list.append(value)
                    except:
                        continue

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

def extract_coordinates(file_path):
    coordinates = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Node number"):
            i += 1
            x = float(lines[i].split(":")[-1].strip())
            i += 1
            y = float(lines[i].split(":")[-1].strip())
            i += 1
            z = float(lines[i].split(":")[-1].strip())
            coordinates.append([x, y, z])
        i += 1
    
    return np.array(coordinates)

def read_ipelem(file_path):
    global_numbers = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Element number"):
            while not lines[i].strip().startswith("Enter the 2 global numbers"):
                i += 1
            numbers = list(map(lambda x: int(x) - 1, lines[i].split(":")[-1].strip().split())) # '-1' bc zero-indexing
            global_numbers.append(numbers)
        i += 1
    
    return np.array(global_numbers)

def read_exelem(filepath):
    """
    Parses element node connections from a text file and returns them as a NumPy array.

    Args:
        filepath (str): Path to the text file.

    Returns:
        np.ndarray: Array of shape (N, 2), where each row is [node1, node2] for an element.
    """
    connections = []

    with open(filepath, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        # Detect an element header
        if re.match(r'\s*Element:\s+\d+', line):
            # Find the next line that contains "Nodes:"
            if 'Nodes:' in lines[i + 1]:
                node_line = lines[i + 2]  # The line after "Nodes:"
                nodes = [int(n) for n in node_line.strip().split()]
                if len(nodes) != 2:
                    raise ValueError(f"Expected 2 nodes, got: {nodes}")
                connections.append(nodes)

    return np.array(connections, dtype=int)

def parse_elements_from_file(filepath):
    """
    Parse element numbers and their values from a text file.

    Args:
        filepath (str): Path to the text file.

    Returns:
        dict: {'Element number': [element numbers],
               'Values': [[values of element 1], [values of element 2], ...]}
    """
    element_numbers = []
    values_list = []

    with open(filepath, 'r') as file:
        lines = file.readlines()

    element_number = None

    for line in lines:
        # Look for element header
        element_match = re.match(r'\s*Element:\s+(\d+)', line)
        if element_match:
            element_number = int(element_match.group(1))
            element_numbers.append(element_number)
            continue

        # Skip the 'Values:' line
        if 'Values:' in line:
            continue

        # Look for the actual values
        if element_number is not None and re.search(r'[0-9]', line):
            values = [float(val) for val in line.strip().split()][0] # keep just one of them
            values_list.append(values)
            element_number = None  # Reset for next element

    return {'Element number': element_numbers, 'Values': values_list}

def compute_joint_radii(nodes, edges, edge_mid_radii):
    """
    Given a radius for each edge (at midpoint), compute per-node radius
    as the average of connected edge radii. If a node has only one incident
    edge, use that edge's radius directly.
    
    Parameters
    ----------
    nodes : (N,3)
    edges : (E,2)
    edge_mid_radii : (E,)
    
    Returns
    -------
    joint_radii : (N,) per-node radii
    edge_radii : (E,2) per-edge start/end radii
    """
    from collections import defaultdict

    N = nodes.shape[0]
    E = edges.shape[0]

    # collect radii per node
    incident = defaultdict(list)
    for e, (i0, i1) in enumerate(edges):
        incident[i0].append(edge_mid_radii[e])
        incident[i1].append(edge_mid_radii[e])

    joint_radii = np.zeros(N, dtype=float)
    for i in range(N):
        if len(incident[i]) == 0:
            joint_radii[i] = 0.0
        elif len(incident[i]) == 1:
            # leaf: just use that edge's radius
            joint_radii[i] = incident[i][0]
        else:
            # average of all connected edge radii
            joint_radii[i] = np.mean(incident[i])

    # now expand to per-edge start/end
    edge_radii = np.zeros((E, 2), dtype=float)
    for e, (i0, i1) in enumerate(edges):
        edge_radii[e, 0] = joint_radii[i0]
        edge_radii[e, 1] = joint_radii[i1]

    return joint_radii, edge_radii

def split_tree_by_edge_removing_last_elements(nodes, edges, edge_radii,
                                              remove_last_n_edges, split_after_e,
                                              prune_unused_nodes=True):

    nodes = np.asarray(nodes)
    edges = np.asarray(edges, dtype=int)
    edge_radii = np.asarray(edge_radii)

    M = len(edges)
    if remove_last_n_edges < 0:
        raise ValueError("remove_last_n_edges must be >= 0")
    if remove_last_n_edges >= M:
        raise ValueError("remove_last_n_edges must be less than number of edges")

    keep_count = M - remove_last_n_edges
    kept_edges = edges[:keep_count]
    kept_radii = edge_radii[:keep_count]

    if split_after_e < 0:
        split_after_e = 0
    if split_after_e >= keep_count - 1:
        split_after_e = keep_count - 1

    # --- Track original edge indices ---
    orig_edge_idx_tree1 = np.arange(keep_count)[:split_after_e + 1]
    orig_edge_idx_tree2 = np.arange(keep_count)[split_after_e + 1:]

    edges_tree1 = kept_edges[:split_after_e + 1]
    radii_tree1 = kept_radii[:split_after_e + 1]

    edges_tree2 = kept_edges[split_after_e + 1:]
    radii_tree2 = kept_radii[split_after_e + 1:]

    def prune_and_remap(edges_sub, nodes_all, radii_sub):
        if edges_sub.size == 0:
            return (np.zeros((0, 3)), np.zeros((0,2), dtype=int),
                    np.zeros((0,)), np.array([]), {})

        used = np.unique(edges_sub)

        if prune_unused_nodes:
            nodes_new = nodes_all[used]
            new_index = {old: i for i, old in enumerate(used)}
            edges_new = np.array([[new_index[a], new_index[b]] for a, b in edges_sub], dtype=int)
            return nodes_new, edges_new, radii_sub, used, new_index
        else:
            return nodes_all.copy(), edges_sub.copy(), radii_sub, np.arange(len(nodes_all)), {}

    # --- Tree 1 ---
    nodes_tree1, edges_tree1, radii_tree1, orig_nodes_tree1, node_map_tree1 = \
        prune_and_remap(edges_tree1, nodes, radii_tree1)

    # --- Tree 2 ---
    nodes_tree2, edges_tree2, radii_tree2, orig_nodes_tree2, node_map_tree2 = \
        prune_and_remap(edges_tree2, nodes, radii_tree2)

    return {
        # Tree 1
        "nodes_tree1": nodes_tree1,
        "edges_tree1": edges_tree1,
        "radii_tree1": radii_tree1,
        "orig_edge_indices_tree1": orig_edge_idx_tree1,
        "orig_node_indices_tree1": orig_nodes_tree1,
        "node_index_map_tree1": node_map_tree1,

        # Tree 2
        "nodes_tree2": nodes_tree2,
        "edges_tree2": edges_tree2,
        "radii_tree2": radii_tree2,
        "orig_edge_indices_tree2": orig_edge_idx_tree2,
        "orig_node_indices_tree2": orig_nodes_tree2,
        "node_index_map_tree2": node_map_tree2,
    }

def remove_first_edges(nodes, edges, edge_radii, remove_first_n_edges, prune_unused_nodes=True):
    """
    Remove the first N edges (and corresponding radii) from a tree,
    while also returning original indices for nodes, edges, and radii.
    """

    nodes = np.asarray(nodes)
    edges = np.asarray(edges, dtype=int)
    edge_radii = np.asarray(edge_radii)

    M = len(edges)
    if remove_first_n_edges < 0:
        raise ValueError("remove_first_n_edges must be >= 0")
    if remove_first_n_edges >= M:
        raise ValueError("remove_first_n_edges must be less than number of edges")

    # --- Track original edge & radii indices ---
    orig_edge_indices_trimmed = np.arange(M)[remove_first_n_edges:]
    orig_radii_indices_trimmed = orig_edge_indices_trimmed.copy()

    # Keep the remaining edges
    kept_edges = edges[remove_first_n_edges:]
    kept_radii = edge_radii[remove_first_n_edges:]

    if prune_unused_nodes:
        # find only the nodes used by the remaining edges
        used_nodes = np.unique(kept_edges)
        nodes_trimmed = nodes[used_nodes]

        # remap old indices to new ones
        new_idx = {old: new for new, old in enumerate(used_nodes)}
        edges_trimmed = np.array([[new_idx[a], new_idx[b]] for a, b in kept_edges], dtype=int)

        orig_node_indices_trimmed = used_nodes
        node_index_map_trimmed = new_idx

    else:
        nodes_trimmed = nodes.copy()
        edges_trimmed = kept_edges.copy()

        orig_node_indices_trimmed = np.arange(len(nodes))
        node_index_map_trimmed = {}

    return {
        "nodes_trimmed": nodes_trimmed,
        "edges_trimmed": edges_trimmed,
        "radii_trimmed": kept_radii,

        # Original index tracking
        "orig_edge_indices_trimmed": orig_edge_indices_trimmed,
        "orig_radii_indices_trimmed": orig_radii_indices_trimmed,
        "orig_node_indices_trimmed": orig_node_indices_trimmed,
        "node_index_map_trimmed": node_index_map_trimmed,
    }

def main():
    parser = argparse.ArgumentParser(description="Visualise lung model simulation results.")
    parser.add_argument("-v",type=str,help="Directory containing ventilation results.")
    parser.add_argument("-q",type=str,help="Directory containing perfusion results")
    parser.add_argument("-s", type=str,help="Subject")
    args=parser.parse_args()
    dir_vent = args.v
    dir_perf = args.q
    subject = args.s

    if subject: print(f"Subject {subject}")

    # region: Read terminal vent & perf
    terminal_q = read_exnodedata(os.path.join(dir_perf,"terminal_perfusion.exnode"))
    vessel_dict = read_exnodedata(os.path.join(dir_perf,'Artery_pressure_perf.exnode'))
    n_units = len(terminal_q[('Node number',None)])

    terminal_v = read_exnodedata(os.path.join(dir_vent,"term_average.exnode"))
    # endregion

    # region: Read vessels
    nodes_dict = read_exnodedata(os.path.join(dir_perf,'Arterial_tree_full.exnode'))
    nodes = np.array(nodes_dict[('coordinates',3)])
    edges = np.array(read_exelem(os.path.join(dir_perf,'Arterial_tree_full.exelem')))
    edges = edges-1 # set Node numbers to zero-indexing
    last_art = int((len(edges)-n_units)/2) # get last element in artery tree
    elem_dict = parse_elements_from_file(os.path.join(dir_perf,'Artery_radius_perf.exelem'))
    radius = np.array(elem_dict['Values'])

    # region: split tree
    tree_dict = split_tree_by_edge_removing_last_elements(nodes,edges,radius,n_units,last_art-1) # split Arterial_tree into Artery and Vein trees
    assert len(tree_dict['nodes_tree1'])==len(tree_dict['nodes_tree2']);'len nodes not the same'
    assert len(tree_dict['edges_tree1'])==len(tree_dict['edges_tree2']);'len edges not the same'
    assert len(tree_dict['radii_tree1'])==len(tree_dict['radii_tree2']);'len radii not the same'
    radii_art, _ = compute_joint_radii(tree_dict['nodes_tree1'],tree_dict['edges_tree1'],tree_dict['radii_tree1'])
    radii_ven, _ = compute_joint_radii(tree_dict['nodes_tree2'],tree_dict['edges_tree2'],tree_dict['radii_tree2'])

    flow = np.array(parse_elements_from_file(os.path.join(dir_perf,'Artery_flow_perf.exelem'))['Values'])
    flows_art = np.array(flow[tree_dict['orig_edge_indices_tree1']]) # remove last 7 elements to match removed edges
    flows_ven = np.array(flow[tree_dict['orig_edge_indices_tree2']])
    term_v = np.array(terminal_v[('flow',1)])
    term_q = np.array(terminal_q[('flow',1)])
    # split tree values
    pressures_art = np.array(vessel_dict[('pressure_perf',1)])[tree_dict['orig_node_indices_tree1']]
    pressures_ven = np.array(vessel_dict[('pressure_perf',1)])[tree_dict['orig_node_indices_tree2']]
    # endregion

    # region: Read airways
    nodes_dict = read_exnodedata(os.path.join(dir_vent,'grown.exnode'))
    nodes = np.array(nodes_dict[('coordinates',3)])
    edges = np.array(read_exelem(os.path.join(dir_vent,'grown.exelem')))
    edges = edges-1
    elem_dict = parse_elements_from_file(os.path.join(dir_vent,'grown_radius.exelem'))
    radius = np.array(elem_dict['Values'])
    joint_radii_air, _ = compute_joint_radii(nodes,edges,radius)
    
    # region: polyscope
    ps.init()

    pts = ps.register_point_cloud("units", np.array(terminal_v[('coordinates',3)]),radius=0.01)
    pts.add_scalar_quantity("vent", term_v,cmap='jet',enabled=True)
    pts.add_scalar_quantity("perf", term_q,cmap='jet',enabled=True)

    tree = ps.register_curve_network(f"airways",nodes,edges)
    tree.add_scalar_quantity("radius",joint_radii_air, defined_on='nodes')
    tree.set_node_radius_quantity('radius',False)

    tree = ps.register_curve_network(f"arteries",tree_dict['nodes_tree1'],tree_dict['edges_tree1'],color=[1,0,0],enabled=True)
    tree.add_scalar_quantity("pressure", pressures_art, defined_on='nodes',cmap='jet',vminmax=(1000,3500),enabled=True)
    tree.add_scalar_quantity("flow", flows_art, defined_on='edges',cmap='jet',enabled=False)
    tree.add_scalar_quantity("radius", radii_art, defined_on='nodes',enabled=False)
    tree.set_node_radius_quantity("radius",False)

    tree = ps.register_curve_network(f"veins",tree_dict['nodes_tree2'],tree_dict['edges_tree2'],color=[0,0,1],radius=0.012,enabled=True)
    tree.add_scalar_quantity("pressure", pressures_ven, defined_on='nodes',cmap='jet',vminmax=(1000,3500),enabled=True)
    tree.add_scalar_quantity("flow", flows_ven, defined_on='edges',cmap='jet',enabled=False)
    tree.add_scalar_quantity("radius", radii_ven, defined_on='nodes',enabled=False)
    tree.set_node_radius_quantity("radius",False)

    ## set slice planes
    cor_plane_pos = ps.add_scene_slice_plane()
    cor_plane_pos.set_pose([0,0,0],[0,1,0])
    cor_plane_pos.set_active(False)
    cor_plane_neg = ps.add_scene_slice_plane()
    cor_plane_neg.set_pose([0,0,0],[0,-1,0])
    cor_plane_neg.set_active(False)

    ax_plane_pos = ps.add_scene_slice_plane()
    ax_plane_pos.set_pose([0,0,0],[0,0,1])
    ax_plane_pos.set_active(False)
    ax_plane_neg = ps.add_scene_slice_plane()
    ax_plane_neg.set_pose([0,0,0],[0,0,-1])
    ax_plane_neg.set_active(False)

    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_background_color([0,0,0,0])
    ps.show() # give control to the main loop, blocks until window is exited

if __name__=="__main__":
    main()