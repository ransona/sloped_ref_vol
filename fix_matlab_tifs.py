import os
import numpy as np
import tifffile as tiff

############################################################
# Utility: More Robust Reading of TIFF Pages
############################################################
def read_tif_pages(input_path):
    """
    Read all pages from a multi-page TIFF file into a NumPy array
    of shape (rows, cols, n_frames).
    """
    frames = []
    with tiff.TiffFile(input_path) as tif_in:
        for page in tif_in.pages:
            data = page.asarray()
            frames.append(data)

    # Convert list to a NumPy array
    stack_3d = np.stack(frames, axis=0)         # shape: (n_pages, rows, cols)
    stack_3d = stack_3d.transpose((1, 2, 0))    # shape: (rows, cols, n_frames)
    return stack_3d

############################################################
# Save a 3D NumPy array back as a multi-page TIFF
############################################################
def save_tif_pages(output_path, stack_3d):
    """
    Save a 3D NumPy array as a multi-page TIFF file.
    Ensures correct ordering: (frames, rows, cols)
    """
    # Transpose to (frames, rows, cols) for correct TIFF writing
    stack_3d = stack_3d.transpose(2, 0, 1)  # Now (frames, rows, cols)

    # Ensure 16-bit unsigned integer format
    stack_3d = np.clip(stack_3d, 0, 65535).astype(np.uint16)

    # Save as a multi-page TIFF
    tiff.imwrite(output_path, stack_3d[0], photometric='minisblack')
    for i in range(1, stack_3d.shape[0]):
        tiff.imwrite(output_path, stack_3d[i], photometric='minisblack', append=True)

############################################################
# Main Function: Process all TIFF files in a folder
############################################################
def process_tiff_folder(input_folder):
    """
    Reads all TIFF files in the given folder and writes them out verbatim
    to a subfolder named 'fixed'.
    """
    output_folder = os.path.join(input_folder, "fixed")
    os.makedirs(output_folder, exist_ok=True)

    # Get list of all TIFF files in the folder
    tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]

    print(f"Found {len(tiff_files)} TIFF files in '{input_folder}'.")

    for file_name in tiff_files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        print(f"Processing: {file_name}")

        # Read TIFF
        stack_3d = read_tif_pages(input_path)

        # Save TIFF
        save_tif_pages(output_path, stack_3d)

        print(f"Saved to: {output_path}")

    print("\nAll files processed. Check the 'fixed' folder.")


############################################################
# Run the script
############################################################
if __name__ == "__main__":
    input_folder = "C:\\Temp\\data\\29-Jan-2025 10_02_30_subset"  # Change this to the directory containing TIFF files
    process_tiff_folder(input_folder)
