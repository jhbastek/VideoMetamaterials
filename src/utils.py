import numpy as np
import networkx as nx

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_graph(image):
    #CONSTRUCTION OF HORIZONTAL EDGES
    hx, hy = np.where(image[1:] & image[:-1]) #horizontal edge start positions
    h_units = np.array([hx, hy]).T
    h_starts = [tuple(n) for n in h_units]
    h_ends = [tuple(n) for n in h_units + (1, 0)] #end positions = start positions shifted by vector (1,0)
    horizontal_edges = zip(h_starts, h_ends)
    #CONSTRUCTION OF VERTICAL EDGES
    vx, vy = np.where(image[:,1:] & image[:,:-1]) #vertical edge start positions
    v_units = np.array([vx, vy]).T
    v_starts = [tuple(n) for n in v_units]
    v_ends = [tuple(n) for n in v_units + (0, 1)] #end positions = start positions shifted by vector (0,1)
    vertical_edges = zip(v_starts, v_ends)

    G = nx.Graph()
    G.add_edges_from(horizontal_edges)
    G.add_edges_from(vertical_edges)
        
    return G

# post-processing
def clean_pred(geom_pred_raw, pixels):
    # floats to ints
    geom_pred_raw[geom_pred_raw < 0.5] = 0
    geom_pred_raw[geom_pred_raw > 0.5] = 1

    geom_pred = geom_pred_raw.astype('int')
    geom_red_full = np.zeros(geom_pred.shape).reshape(-1,pixels,pixels).astype('int')

    # post-processing
    for i in range(geom_pred_raw.shape[0]):

        geom_pred_cur = geom_pred[i].reshape(pixels,pixels)

        # remove individual pixels
        for j in range(pixels):
            for k in range(pixels):
                neighbours = np.full(4, True)
                if j != 0:
                    if not geom_pred_cur[j-1,k]:
                        neighbours[0] = False
                if j != pixels-1:
                    if not geom_pred_cur[j+1,k]:
                        neighbours[1] = False
                if k != 0:
                    if not geom_pred_cur[j,k-1]:
                        neighbours[2] = False
                if  k != pixels-1:
                    if not geom_pred_cur[j,k+1]:
                        neighbours[3] = False
                if (~neighbours).all():
                    geom_pred_cur[j,k] = 0

        # remove subgraphs
        G = create_graph(geom_pred_cur)
        max_edges = 0
        largest_graph_idx = 0
        for iter, c in enumerate(nx.connected_components(G)):
            no_edges = len(c)
            if max_edges < no_edges:
                largest_graph_idx = iter
                max_edges = no_edges

        G_red_edgelist = list(nx.connected_components(G))[largest_graph_idx]
        geom_red = np.zeros((pixels, pixels)).astype('int')

        for edge in G_red_edgelist:
            geom_red[edge[0], edge[1]] = 1

        geom_red_full[i,:,:] = geom_red

    return geom_red_full.reshape(-1, pixels**2)

def reduce_csv_to_first_n_rows(csv_file, n):
    """
    Reduce a csv file to the first n rows.
    """
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        lines = lines[:n]
        with open(csv_file, 'w') as f:
            f.writelines(lines)

def compute_NRMSE(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_pred - y_true))/np.sum(np.square(y_true)))

def compute_full_error(data, samples, closest_match = False, full_data = None, skip_first=False):
    tot_data = data.shape[0]
    err = []
    assert tot_data % samples == 0, 'Number of samples must be a divisor of the total number of data points'
    assert not closest_match or full_data is not None, 'If closest_match is True, full_data must be provided'
    if closest_match:
        err_match = []
    data_per_sample = tot_data//samples
    for i in range(samples):
        for j in range(1, data_per_sample):
            # check if pred has not converged, only take error before
            valid_entries = 0
            for k in range(data.shape[1]):
                if np.abs(data[i*data_per_sample+j,k]) > 50:
                    break
                valid_entries += 1
            if valid_entries == 0:
                err_cur = np.nan
            else:
                # # do not consider first values which is zero --> this is wrong, commented out but left here for reference
                # err_cur = compute_NRMSE(data[i*data_per_sample,1:valid_entries], data[i*data_per_sample+j,1:valid_entries])
                err_cur = compute_NRMSE(data[i*data_per_sample,:valid_entries], data[i*data_per_sample+j,:valid_entries])
            err.append(err_cur)

        if closest_match:
            assert len(data[i*data_per_sample,:]) == full_data.shape[1], 'Data dimensions do not match.'
            err_cur_closest_match, _ = find_closest_match(data[i*data_per_sample,:], full_data, skip_first)
            err_match.append(err_cur_closest_match)

    errors_np = np.array(err).reshape(samples, data_per_sample - 1)
    min_errors_np = np.nanmin(errors_np, axis=1)

    # retrieve the index of the minimum error, set to -1 if no valid error was found
    try:
        min_errors_np_idx = np.nanargmin(errors_np, axis=1)
    except:
        min_errors_np_idx = -1

    if closest_match:
        err_match = np.array(err_match).reshape(samples)
        relative_err = (min_errors_np - err_match) / err_match
        best_relative_err = np.nanmin(relative_err)
        # Return mean of errors, mean of (minimum errors per sample), indices of minimum errors (per sample),
        # closest match errors, and best performance of inverse model compared to closest match.
        return np.mean(errors_np), np.mean(min_errors_np), min_errors_np_idx, err_match, best_relative_err
    else:
        # Return mean of errors, mean of (minimum errors per sample), indices of minimum errors (per sample).
        return np.mean(errors_np), np.mean(min_errors_np), min_errors_np_idx

def compute_NRMSE_arrays(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_pred - y_true), axis=1)/np.sum(np.square(y_true)))

def find_closest_match(y_true, y_pred, skip_first=False):
    NRMSE = compute_NRMSE_arrays(y_true, y_pred)
    # select the index of the minimum NRMSE (skip the first one, which is the same as the target)
    if skip_first:
        idx = np.argsort(NRMSE)[1]
    else:
        idx = np.argsort(NRMSE)[0]
    return NRMSE[idx], idx