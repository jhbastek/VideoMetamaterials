import os
import numpy as np
import networkx as nx
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

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

def normalize(arr, min_val, max_val):
    return (arr - min_val) / (max_val - min_val)

def unnorm(arr, min_val, max_val):
    return arr * (max_val - min_val) + min_val

def convert_isolated_pixels_gif(gif_path):
    img = Image.open(gif_path)
    frames = []
    for frame in ImageSequence.Iterator(img):
        frame = frame.convert("RGBA")
        data = frame.load()
        width, height = frame.size
        # Define offsets for neighbouring pixels
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Iterate over the image pixels
        for y in range(height):
            for x in range(width):
                # Get the current pixel
                r, g, b, a = data[x, y]
                # If the pixel is not transparent
                if a != 0:
                    is_isolated = True
                    # Check if all neighbouring pixels are transparent
                    for dx, dy in offsets:
                        nx, ny = x + dx, y + dy
                        # Make sure we're not out of the image boundaries
                        if 0 <= nx < width and 0 <= ny < height:
                            nr, ng, nb, na = data[nx, ny]
                            # If the neighbouring pixel is not transparent, 
                            # then the current pixel is not isolated
                            if na != 0:
                                is_isolated = False
                                break
                    # If the pixel is isolated, make it transparent
                    if is_isolated:
                        data[x, y] = (r, g, b, 0)        
        frames.append(frame)
    imageio.v3.imwrite(gif_path, frames, plugin='pillow', mode='RGBA', duration=200, loop=0, disposal=2, quantize=256)

def remove_artifacts(image, threshold=10, check_isolated_pixels=False):
    pixels = len(image)
    for i in range(pixels):
        for j in range(pixels):
            neighbors = []
            if i > 0:
                neighbors.append(image[i-1,j])
            if i < pixels - 1:
                neighbors.append(image[i+1,j])
            if j > 0:
                neighbors.append(image[i,j-1])
            if j < pixels - 1:
                neighbors.append(image[i,j+1])
            neighbors = [val for val in neighbors]
            if check_isolated_pixels:
                neighbors = [val for val in neighbors]
                averaged_abs_value = np.mean(np.abs(neighbors))
                if np.abs(averaged_abs_value - image[i,j]) > threshold and averaged_abs_value < 1.e-6:
                    image[i,j] = averaged_abs_value
            else:
                if len(neighbors) == 4:
                    averaged_value = np.mean(neighbors)
                    if np.abs(averaged_value - image[i,j]) > threshold:
                        image[i,j] = averaged_value
    return image

def smooth_frame(frame, neighbors_required=3, recursive=True):
    pixels = len(frame)
    smooth_frame = frame.copy()
    update = True    
    while update:
        update = False
        for i in range(pixels):
            for j in range(pixels):
                if smooth_frame[i,j] == 0:
                    neighbors = []
                    if i > 0:
                        neighbors.append(smooth_frame[i-1,j])
                    if i < pixels - 1:
                        neighbors.append(smooth_frame[i+1,j])
                    if j > 0:
                        neighbors.append(smooth_frame[i,j-1])
                    if j < pixels - 1:
                        neighbors.append(smooth_frame[i,j+1])
                    nonzero_neighbors = [val for val in neighbors if val != 0]
                    if len(nonzero_neighbors) >= neighbors_required:
                        averaged_value = np.mean(nonzero_neighbors)
                        if averaged_value != smooth_frame[i,j]:
                            update = True
                            smooth_frame[i,j] = averaged_value
        if not recursive:
            break

    smooth_frame = remove_artifacts(smooth_frame)

    return smooth_frame

def update_pixel_positions(shift_x, shift_y, pixel_values, bin_data):
    frames, pixels, _ = shift_x.shape
    result = np.zeros((frames, pixels, pixels))
    for f in range(frames):
        # NOTE last axis is x, second last is y, also negative y shift is must be added to pixels
        for x_lagr in range(pixels):
            for y_lagr in range(pixels):
                x_euler = x_lagr + shift_x[f, y_lagr, x_lagr]
                # horizontal wrap around
                x_euler = x_euler % pixels
                y_euler = y_lagr - shift_y[f, y_lagr, x_lagr]
                if 0 <= y_euler < pixels:
                    if bin_data[f, y_lagr, x_lagr] != 0:
                        result[f, y_euler, x_euler] = pixel_values[f, y_lagr, x_lagr]
        result[f] = smooth_frame(result[f])
    return result

def gif_to_array(path):
    reader = imageio.get_reader(path)
    tot_rows, tot_cols = reader.get_data(0).shape[0] // 100, reader.get_data(0).shape[1] // 100
    if tot_rows == 0 and tot_cols == 0:
        tot_rows, tot_cols = 1, 1
    num_frames = reader.get_length()
    if num_frames != 11:
        print('Warning: Number of frames of provided gif is not 11. This is likely due to convergence problems in the simulation.')
    if tot_rows == 1 and tot_cols == 1:
        frames = np.zeros((num_frames, 96, 96), dtype=np.uint8)
    else:
        frames = np.zeros((num_frames, 100*tot_rows, 100*tot_cols), dtype=np.uint8)
    for i in range(num_frames):
        frame = reader.get_data(i)
        if len(frame.shape) == 3:
            frame = frame[:,:,0]
        frames[i] = frame
    reader.close()
    return frames, tot_rows, tot_cols

def crop_gif(path, row, col, save=True):
    frames, tot_rows, tot_cols = gif_to_array(path)
    if not (tot_rows == 1 and tot_cols == 1):
        row_start = row * 100
        row_end = (row + 1) * 100
        col_start = col * 100
        col_end = (col + 1) * 100
        frames = frames[:, row_start:row_end, col_start:col_end]
        frames = frames.astype(np.uint8)
        pad = 2 # unpad
        frames = frames[:, pad:-pad, pad:-pad]
    if save:
        path_res = path[:-4] + '-' + str(row) + '-' + str(col) + '.gif'
        imageio.mimsave(path_res, frames, duration=0.2)
    if not save:
        return frames

def create_visualization(path, frame_ranges, row, col, ref_frame, atol=0.02, disp_compression = True):

    if ref_frame == 'eulerian':
        s_22_field_idx = 2
    elif ref_frame == 'lagrangian':
        s_22_field_idx = 3
    
    frame_range_full = np.genfromtxt(frame_ranges, delimiter=',')
    frame_range_full = frame_range_full[~np.isnan(frame_range_full).any(axis=1)] # remove nan rows (potentially due to header)

    # add batch dim if necessary
    if len(frame_range_full.shape) == 1:
        frame_range_full = frame_range_full[np.newaxis, :]
    
    if ref_frame == 'lagrangian':
        min_u_1 = frame_range_full[:, 0].min()
        max_u_1 = frame_range_full[:, 1].max()
        min_u_2 = frame_range_full[:, 2].min()
        max_u_2 = frame_range_full[:, 3].max()
        frame_range_full = frame_range_full[:, -4:]
    if ref_frame == 'eulerian':
        path_pred_bin = path + 'prediction_channel_0.gif'
        data_pred_bin = crop_gif(path_pred_bin, row, col, save=False)

    elif ref_frame == 'lagrangian':
        zero_u_2 = normalize(np.zeros(1), min_u_2, max_u_2)
        path_pred_u_1 = path + 'prediction_channel_0.gif'
        data_pred_u_1 = crop_gif(path_pred_u_1, row, col, save=False)
        path_pred_u_2 = path + 'prediction_channel_1.gif'
        data_pred_u_2 = crop_gif(path_pred_u_2, row, col, save=False)

        # extract upper left quarter since we evaluate u_2, which will be nonzero there
        pixels = data_pred_u_2.shape[-1]
        pred_u_2_red = data_pred_u_2[:,:pixels//2,:pixels//2].copy()
        pred_u_2_red = normalize(pred_u_2_red, 0, 255)
        topology = np.zeros((pixels//2, pixels//2))
        close_to_value = np.isclose(pred_u_2_red[:, :, :], zero_u_2, atol=atol)
        all_frames_match = np.all(close_to_value, axis=0)
        topology[:,:] = np.logical_not(all_frames_match)
        topology = np.concatenate((topology, np.flip(topology, axis=0)), axis=0)
        data_pred_bin = np.concatenate((topology, np.flip(topology, axis=1)), axis=1)
        # repeat along first axis to match number of frames
        data_pred_bin = np.repeat(data_pred_bin[np.newaxis, :, :], data_pred_u_2.shape[0], axis=0)
        
        data_pred_u_1 = normalize(data_pred_u_1, 0, 255)
        data_pred_u_2 = normalize(data_pred_u_2, 0, 255)
        # bring to real range
        data_pred_u_1 = unnorm(data_pred_u_1, min_u_1, max_u_1)
        data_pred_u_2 = unnorm(data_pred_u_2, min_u_2, max_u_2)
        # convert to 96 pixels and round
        data_pred_u_1 = np.round(data_pred_u_1 * 96).astype(int)
        data_pred_u_2 = np.round(data_pred_u_2 * 96).astype(int)

    # global min max range for prediction
    data_min = np.min(frame_range_full[:, 1])
    data_max = np.max(frame_range_full[:, 2])

    path_pred = path + 'prediction_channel' + '_' + str(s_22_field_idx) + '.gif'
    data_pred = crop_gif(path_pred, row, col, save=False)
    
    save_path = path + 'visualization_' + str(row) + '-' + str(col) + '/'
    # create directory if it does not exist
    os.makedirs(save_path, exist_ok=True)

    # list of strain steps
    strain = 0.2
    strain_list = np.linspace(0., strain, num = len(data_pred))
    strain_list[0] = 0.01*strain

    # normalize from sth. in between [0,255] to [0,1]
    data_pred = normalize(data_pred, 0., 255.)
    # unnormalize from [0,1] to [data_min, data_max] ---> true values
    data_pred = unnorm(data_pred, data_min, data_max)

    # manually set stresses to 0 for pixels with no topology
    if ref_frame == 'eulerian':
        data_pred[data_pred_bin < 255/2] = 0
    elif ref_frame == 'lagrangian':
        data_pred[data_pred_bin == 0] = 0
        # convert to Eulerian Frame
        data_pred_bin_euler = update_pixel_positions(data_pred_u_1, data_pred_u_2, data_pred_bin, data_pred_bin)
        # convert to Eulerian Frame
        data_pred_euler = update_pixel_positions(data_pred_u_1, data_pred_u_2, data_pred, data_pred_bin)
        # manually set stresses to 0 for pixels with no topology
        data_pred_euler[data_pred_bin_euler == 0] = 0

    # extract summed values over time
    curve_pred = np.zeros((data_pred.shape[0]))
    for i in range(data_pred.shape[0]):
        if ref_frame == 'eulerian':
            top_row = int(np.floor((1.-strain_list[i])*data_pred.shape[-1]))
            curve_pred[i] = -np.mean(data_pred[i,-top_row:,:])
        elif ref_frame == 'lagrangian':
            curve_pred[i] = -np.mean(data_pred[i,:,:])
            curve_pred[i] *= 1./(1.-strain_list[i])
            top_row = int(np.floor((1.-strain_list[i])*data_pred.shape[-1]))

    # shift curve_pred so that 0 strain is at 0 based on linearization of the first two values taken at 0.2% and 2% strain
    shift = curve_pred[0] - ((curve_pred[1] - curve_pred[0]) / (strain_list[1] - strain_list[0])) * strain_list[0]
    curve_pred_shifted = curve_pred.copy()
    curve_pred_shifted -= shift
    np.savetxt(save_path + 'stress_strain_estimate.csv', np.concatenate((strain_list[:,None], curve_pred_shifted[:,None]), axis=1), delimiter=',', header='strain,pred_pixel_shifted', comments='')

    # we want to be symmetric around 0 for the colormap so we need to find the max absolute value
    data_max = np.max([np.abs(data_min), np.abs(data_max)])
    data_min = -data_max

    # normalize from true values to range [0, 1]
    data_pred = normalize(data_pred, data_min, data_max)    
    # unnormalize from [0, 1] to range [0, 255]
    data_pred = unnorm(data_pred, 0., 255.)
    # round to int
    data_pred = np.round(data_pred).astype(np.uint8)

    if ref_frame == 'lagrangian':
        data_pred_euler = normalize(data_pred_euler, data_min, data_max)
        # unnormalize from [0, 1] to range [0, 255]
        data_pred_euler = unnorm(data_pred_euler, 0., 255.)
        # round to int
        data_pred_euler = np.round(data_pred_euler).astype(np.uint8)

    # crop data that has zero entry in data_pred_bin
    if ref_frame == 'eulerian':
        data_pred[data_pred_bin < 255/2] = 0
    elif ref_frame == 'lagrangian':
        data_pred[data_pred_bin == 0] = 0

    cmap = plt.get_cmap('jet')
    data_color_pred = cmap(data_pred)
    data_color_pred = (data_color_pred*255).astype(np.uint8)

    # overwrite alpha channel of data_color with 0 where data_fem_bin_euler is 0
    data_color_pred[data_pred_bin == 0, 3] = 0

    if ref_frame == 'eulerian':
        if disp_compression:
            gray_color = np.array([227, 227, 227, 255]).astype(np.uint8) # RGB values for light gray
            for frame in range(len(data_color_pred)):
                box_end = round(strain_list[frame]*data_color_pred[frame].shape[0])
                data_color_pred[frame, :box_end, :, :] = gray_color

    path_res_pred = save_path + 'visualization.gif'
    imageio.v3.imwrite(path_res_pred, data_color_pred, plugin='pillow', mode='RGBA', duration=200, loop=0, disposal=2, quantize=256)

    if ref_frame == 'lagrangian':
        path_res_pred_euler = save_path + 'visualization_conv_euler.gif'
        # crop data that has zero entry in data_fem_bin
        data_pred_euler[data_pred_bin_euler == 0] = 0
        for f in range(len(data_pred_euler)):
            data_pred_euler[f,:,:] = remove_artifacts(data_pred_euler[f,:,:], check_isolated_pixels=True)
        data_color = cmap(data_pred_euler)
        data_color = (data_color*255).astype(np.uint8)

        # overwrite alpha channel of data_color with 0 where data_fem_bin_euler is 0
        data_color[data_pred_bin_euler == 0, 3] = 0

        if disp_compression:
            gray_color = np.array([227, 227, 227, 255]).astype(np.uint8) # RGB values for light gray
            for frame in range(len(data_color)):
                box_end = round(strain_list[frame]*data_color[frame].shape[0])
                data_color[frame, :box_end, :, :] = gray_color

        imageio.v3.imwrite(path_res_pred_euler, data_color, plugin='pillow', mode='RGBA', duration=200, loop=0, disposal=2, quantize=256)

        # removes artefacts
        convert_isolated_pixels_gif(path_res_pred_euler)

    # save the colormap as a PNG
    fig, ax = plt.subplots(figsize=(1, 30))
    fig.subplots_adjust(bottom=0.5)
    norm = plt.Normalize(data_min, data_max)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=ax, orientation='vertical')
    cb1.set_label(r'$\sigma_{22}$', rotation=0, labelpad=15, fontsize=30)
    cb1.ax.tick_params(labelsize=20)
    colormap_path = save_path + 'visualization_legend.png'
    fig.savefig(colormap_path, bbox_inches='tight')
    plt.close(fig)