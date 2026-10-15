import os, re

def total_elems_in_ipelem(path, filename):
    search_string = 'elements'
    with open(os.path.join(path, filename + '.ipelem'), 'r') as file:
        for line in file:
            if search_string in line:
                # Extract float values using a regular expression
                float_values = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                float_values = [float(value) for value in float_values]  # Convert to float
                return int(float_values[-1])
        else:
            print(" Couldn't find total number of elements.")

def build_tree(pairs):
    """
    Build a tree dictionary from parent-daughter node pairs.

    Args:
        pairs (list of tuples): List of (parent, daughter) node pairs.
    
    Returns:
        dict: A dictionary representing the tree structure.
    """
    tree = {}
    for parent, daughter in pairs:
        if parent not in tree:
            tree[parent] = []  # Initialize the list for the parent node
        tree[parent].append(daughter)  # Add the daughter node
        
        # Ensure daughter is also in the dictionary as a key (for completeness)
        if daughter not in tree:
            tree[daughter] = []  # Daughter has no children initially

    return tree

def get_tree_dict_from_exelem(path, filename):
    # read exelem file
    with open(os.path.join(path, filename+'.exelem'), 'r') as file:
        lines = file.readlines()
    
    # get indices with 'Element'
    indices = [i for i, s in enumerate(lines) if 'Element' in s]

    pairs = []
    for i in range(len(indices)):
        line = lines[indices[i]+2]
        # get nodes connected by that element
        pairs.append(list(map(int, re.findall(r'\b\d+\b', line)))) # store parent-daughter pairs of nodes
    
    tree = build_tree(pairs)
    return tree

def compute_strahler_order(tree, node):
    """
    Compute the Strahler order of a tree.

    Args:
        tree (dict): A dictionary representing the tree structure. Keys are nodes, and values are lists of child nodes.
        node: The current node for which the Strahler order is calculated.

    Returns:
        int: Strahler order of the node.
    """
    if not tree[node]:  # Leaf node
        return 1
    
    child_orders = [compute_strahler_order(tree, child) for child in tree[node]]
    max_order = max(child_orders)
    if child_orders.count(max_order) > 1:
        return max_order + 1
    else:
        return max_order

def calculate_radius_given_other_branch(order_k, order_m, radius_m, scaling_factor):
    """
    Calculate the radius of a target branch given its Strahler order and the radius of a known branch.
    Equation: R_target = R_known * alpha**(order_target - order_known) where alpha is the scaling factor aka s_ratio
    
    Args:
        order_k (int): Strahler order of the target branch.
        order_m (int): Strahler order of the known branch.
        radius_m (float): Radius of the known branch.
        scaling_factor (float): Scaling factor for the branch radius.
    
    Returns:
        float: Radius of the target branch.
    """
    return radius_m * (scaling_factor ** (order_k - order_m))

def get_max_order(exelem_path, exelem_filename):
    tree = get_tree_dict_from_exelem(exelem_path,exelem_filename) # from artery_full.exelem, get dict with parent node as key and child node(s) as val
    root_node = 1 # Node 1 in artery tree is the root (the ultimate parent node)
    max_order = compute_strahler_order(tree, root_node)
    return max_order

def get_inlet_rad_ven(exelem_path, exelem_filename, ave_measured_vein_radius, s_ratio, Ptm):
    tree = get_tree_dict_from_exelem(exelem_path,exelem_filename) # artery_full.exelem. Get dict listing parent node as key and child node(s) as val
    pulm_outlet_node = 1 # pulmonary vein tree is a copy of pulmonary artery tree, so Node 1 of artery tree may be used as the vein outlet
    pulm_outlet_order = compute_strahler_order(tree, pulm_outlet_node) # Strahler order of the pulmonary vein outlet
    max_order = pulm_outlet_order
    inlet_order = 1 # Strahler order of terminal veins (attaching to capillary sheet)
    strained_radius = calculate_radius_given_other_branch(inlet_order, max_order, ave_measured_vein_radius, s_ratio)
    
    alpha = 0.0001503 # pressure_resistance_flow.f90, line 90
    unstrained_radius = strained_radius / (1 + alpha * Ptm) # in arteries, Ptm=mPAP. In veins, Ptm=PWCP

    return unstrained_radius

def adjust_strahler_ratio(target_radius, target_branch_order, known_order, known_radius, Ptm):
    """
    Adjust the Strahler ratio to get a branch radius close to the target value.

    Parameters:
        root_radius (float): Radius of the root branch.
        target_radius (float): Desired radius of the branch.
        order (int): The order of the branch getting its radius fitted.
        error_threshold (float): The acceptable error threshold for the branch radius.
        max_iterations (int): Maximum number of iterations for convergence.

    Returns:
        float: The optimal Strahler ratio.
    """

    alpha = 0.0001503 # Fixed constant from pressure_resistance_flow.f90, line 90

    # Initial bounds for Strahler ratio
    low = 1.0  # Strahler ratio cannot be less than 1
    high = 10.0  # Arbitrary upper bound for the search

    # Set up simple iterative fit settings
    error_threshold=1e-3
    max_iterations=1000

    for iteration in range(max_iterations):
        # Midpoint of the current range
        mid = (low + high) / 2

        # Compute the radius with the current Strahler ratio
        radius = calculate_radius_given_other_branch(target_branch_order, known_order, known_radius, mid)
        # unstrained_radius = strained_radius / (1 + alpha * Ptm) # in arteries, Ptm=mPAP. In veins, Ptm=PWCP

        # Check the error
        error = abs(radius - target_radius)

        if error <= error_threshold:
            return mid, radius  # Found a suitable Strahler ratio

        # Adjust search bounds
        if radius < target_radius:
            high = mid  # Decrease the ratio
        else:
            low = mid  # Increase the ratio

        if iteration>max_iterations:
            break

    raise ValueError("Unable to find a suitable Strahler ratio within the given constraints.")
