import numpy as np
import numpy.random as rand
import scipy.fftpack
import networkx as nx

def gaussian_random_field(alpha = 1.0,
                          size = 128,
                          seed = None,
                          flag_normalize = True):
    if seed:
        np.random.seed(seed=seed)   
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    k_idx = scipy.fftpack.fftshift(k_ind)
    amplitude = np.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
    amplitude[0,0] = 0
    noise = np.random.normal(size = (size, size)) \
        + 1j * np.random.normal(size = (size, size))
    gfield = np.fft.ifft2(noise * amplitude).real
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)
    return gfield

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

def generate_geometry(grf_alpha, pixels, pixel_threshold_rel, grf_threshold_rel):

    pixel_threshold = int(pixels*pixel_threshold_rel)
    grf_threshold = rand.uniform() * grf_threshold_rel

    flag = False
    while flag == False:
        filling_bin = gaussian_random_field(alpha = grf_alpha, size = pixels).reshape(-1)
        filling_bin[filling_bin < grf_threshold] = 0
        filling_bin[filling_bin > grf_threshold] = 1
        geom_flattened = filling_bin.copy().astype('int')
        geom_l = geom_flattened[0:-1:pixels]
        geom_r = geom_flattened[pixels-1::pixels]
        geom_d = geom_flattened[-pixels:]
        geom_u = geom_flattened[0:pixels]

        geom = geom_flattened.reshape(pixels,pixels)

        if (np.sum(geom_l) >= pixel_threshold and np.sum(geom_r) >= pixel_threshold
            and np.sum(geom_d) >= pixel_threshold and np.sum(geom_u) >= pixel_threshold):
            G = create_graph(geom)
            if nx.is_connected(G):
                flag = True

    return geom_flattened