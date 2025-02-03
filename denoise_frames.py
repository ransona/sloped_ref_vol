
import os
import glob
import shutil
import numpy as np
import imageio.v3 as imageio
import tempfile
import subprocess
import tifffile as tiff
import numpy as np
                          
def find_deepest_folder(start_path):
    current_path = start_path
    while True:
        subfolders = [f for f in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, f))]
        if len(subfolders) != 1:  # Stop if there isn't exactly one subfolder
            break
        current_path = os.path.join(current_path, subfolders[0])
    return current_path

def denoise_np(nps_list, denoise_config):
    """
    Processes a list of NumPy arrays (each with shape X * Y * T), saving them as multipage TIFFs,
    running a denoising pipeline, and then reconstructing the denoised arrays.
    
    Parameters:
        nps_list (list of np.array): List of NumPy arrays with shape (X, Y, T).
        temp_input_folder (str): Path to temporary input folder.
        temp_output_folder (str): Path to temporary output folder.
        pipeline_command (list): Command to run the denoising pipeline as a subprocess.

    Returns:
        list of np.array: Denoised NumPy arrays with shape (X, Y, T).
    """
    denoise_config['temp_output_folder'] = os.path.join(denoise_config['temp_folder'], 'denoised')
    # Ensure temporary folders exist and are empty
    if os.path.exists(denoise_config['temp_folder']):
        shutil.rmtree(denoise_config['temp_folder'])  # Remove old folder
    os.makedirs(denoise_config['temp_folder'])  # Create empty folder
    os.makedirs(denoise_config['temp_output_folder']) # Create empty folder for output

    # Save each NumPy array as a multipage TIFF
    pix_value_offset = []
    for i, np_array in enumerate(nps_list):
        file_path = os.path.join(denoise_config['temp_folder'], f"image_{i:03d}.tif")
        pix_value_offset.append(-np.min(np_array)+1000)
        all_frames = np_array + pix_value_offset[-1]
        if np_array.shape[0] < denoise_config["patch_t"]:
            times_to_reproduce = np.ceil(denoise_config["patch_t"]/np_array.shape[0]).astype(int)
            all_frames = np.tile(all_frames, (times_to_reproduce, 1, 1))
        tiff.imwrite(file_path, all_frames, dtype=np.int16, photometric='minisblack')

    # Run the denoising pipeline
    denoise_model = denoise_config["denoise_model"]
    gpus = denoise_config["gpus"]
    datasets_path = os.path.dirname(denoise_config["temp_folder"])
    datasets_folder = os.path.basename(denoise_config["temp_folder"])
    srdtrans_launcher = denoise_config["srdtrans_launcher"]
    output_path = denoise_config['temp_output_folder']
    patch_x = denoise_config["patch_x"]
    patch_t = denoise_config["patch_t"]
    # model path
    pth_path = '/home/adamranson/data/srt_models'
    denoise_env = denoise_config["denoise_env"]
    # Command to execute the script with arguments inside the conda environment
    denoise_command = f'conda run -n {denoise_env} python {srdtrans_launcher} --datasets_folder {datasets_folder} --output_path {output_path} --pth_path {pth_path} --denoise_model {denoise_model} --GPU {gpus} --patch_x {patch_x} --patch_t {patch_t} --datasets_path {datasets_path}'
    # Run the command
    result = subprocess.run(denoise_command, shell=True, capture_output=True, text=True)
    # Print or return output
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)

    # deal with output data being in subfolders and tidy up
    # move the tif files to root of denoised_tif folder
    data_subdir = find_deepest_folder(output_path)
    [shutil.move(os.path.join(data_subdir, item), output_path) for item in os.listdir(data_subdir)]
    # Iterate through all items in the output_path
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path):  # Check if it's a directory
            shutil.rmtree(item_path)  # Delete the directory
            print(f"Deleted folder: {item_path}")  # Print the path of the deleted folder  

    # Read the denoised TIFFs back into NumPy arrays
    denoised_nps = []
    for i in range(len(nps_list)):  # Same number of files expected as input
        matching_file = glob.glob(f"{denoise_config['temp_output_folder']}/image_{i:03d}_*")
        all_frames = tiff.imread(matching_file)
        # truncate to original number of frames
        all_frames = all_frames[:nps_list[i].shape[0]]
        # reverse pix value offset
        all_frames = all_frames - pix_value_offset[i]
        denoised_nps.append(all_frames)
        

    return denoised_nps

############################################################
# Example Usage
############################################################

if __name__ == "__main__":

    denoise_config = {
        "denoise_model": "multiplane_9_202412201343", # should be a model matched to data being denoised
        "gpus": "0,1", # machine specific (will just be one gpu on most computers)
        "temp_folder": "/home/adamranson/data/Temp/denoise_temp", # a temporary folder used by denoiser - should be accessible by the user
        "patch_x": 160,
        "patch_t": 160, # can potentially speed up by reducing this but must match model in denoise_model
        "pth_path": "/home/adamranson/data/srt_models", # where denoise model is stored
        "srdtrans_launcher": "/home/adamranson/code/SRDTrans/test.py", # where SRDTrans repo is stored
        "denoise_env": "srdtrans" # environment should be established on the user's machine (install srdtrans)
    }  
    
    # load tif file as example
    tif_path = "/home/adamranson/data/Temp/z_ref_29-Jan-2025 10_02_30_000001__00001.tif"
    tif_step = 10
    tif_start = 0
    with tiff.TiffFile(tif_path) as tif:
        num_frames = len(tif.pages)  # Total number of frames
        selected_frames = [tif.pages[i].asarray() for i in range(tif_start+tif_step, num_frames, tif_step)]

    nps_list = []
    nps_list.append(np.array(selected_frames))

    denoise_np(
        nps_list, 
        denoise_config
    )
