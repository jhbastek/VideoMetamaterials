from src.utils import create_visualization

'''
This script generates a visualization of the gif files provided in the <given path>.
The visualization is stored in '<given path>/visualization_<row>_<col>/'.
Row and col are the indices of the conditioning and prediction to visualize, respectively, 
and coincide with the row and column indices of the gif containing the predictions.
For FEM evaluation, the row and col indices are both 0 (since we do not collect multiple predictions).
Ensure that the reference frame is consistent with the provided gifs for correct visualization.
The script also creates the estimated stress-strain response based on the pixel representation in '<given path>/visualization_<row>_<col>/stress_strain_estimate.csv'.
'''

def main():

    # diffusion model prediction evaluation
    path = './runs/pretrained/eval_target_w_5.0_0/step_200000/gifs/' # path to unprocessed gifs
    ref_frame = 'lagrangian' # 'eulerian', 'lagrangian' # reference frame of the gif
    path_frames_full = './data/' + ref_frame + '/training/frame_range_data.csv' # path to csv file with ranges of the data, necessary for normalization
    row = 0 # index of conditioning to visualize (0-indexed)
    col = 0 # index of prediction to visualize (0-indexed)

    # # FEM evaluation
    # ref_frame = 'eulerian'
    # path = './runs/pretrained/eval_target_w_5.0_0/step_200000/abaqus_eval_sample_0/gif/'
    # path_frames_full = path + 'frame_range.csv'
    # row = 0 # keep this at 0 for FEM evaluation
    # col = 0 # keep this at 0 for FEM evaluation

    create_visualization(path, path_frames_full, row, col, ref_frame=ref_frame, atol = 0.02)
    
if __name__ == "__main__":
    main()