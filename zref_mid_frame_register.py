import os
import random
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy.signal import correlate
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

############################################################
# 1) Read TIFF Pages
############################################################
def read_tif_pages(input_path):
    """Read all pages from a multi-page TIFF file into a NumPy array (rows, cols, n_frames)."""
    frames = []
    with tiff.TiffFile(input_path) as tif_in:
        for page in tif_in.pages:
            data = page.asarray()
            frames.append(data)

    stack_3d = np.stack(frames, axis=0)        # (n_pages, rows, cols)
    stack_3d = stack_3d.transpose((1, 2, 0))   # (rows, cols, n_frames)
    return stack_3d

############################################################
# 2) Frame-to-Frame Registration
############################################################
def rapid_reg_non_par(frames, ref_frame):
    """Register frames to a reference frame using phase cross-correlation."""
    registered_frames = np.zeros_like(frames)
    for i in range(frames.shape[2]):
        shift_values, _, _ = phase_cross_correlation(ref_frame, frames[:, :, i])
        registered_frames[:, :, i] = shift(frames[:, :, i], shift_values)
    return registered_frames

############################################################
# 3) Calculate Odd/Even Line Offset (Using Median)
############################################################
def calc_odd_even_line_offset_median(frame2d, num_lines=200):
    """Compute line-to-line offset using cross-correlation on up to 200 odd/even pairs."""
    rows, cols = frame2d.shape
    valid_odds = [r for r in range(1, rows, 2) if (r + 1) < rows]
    if not valid_odds:
        return 0.0

    chosen = valid_odds if len(valid_odds) <= num_lines else random.sample(valid_odds, num_lines)
    offsets = []
    for r in chosen:
        line_odd = frame2d[r, :]
        line_even = frame2d[r + 1, :]
        l1_c = line_odd - np.mean(line_odd)
        l2_c = line_even - np.mean(line_even)

        cross_corr = correlate(l1_c, l2_c, mode='full')
        n = len(line_odd)
        lags = np.arange(-n + 1, n)
        best_lag = lags[np.argmax(cross_corr)]
        offsets.append(best_lag)

    if len(offsets) > 0:
        median_offset = -float(np.median(offsets))  # Negate before applying
        print(f"  -> Median line-to-line offset (odd to even) = {median_offset:.2f} px")
        return median_offset
    else:
        print("  -> No valid odd/even pairs found for line offset correction.")
        return 0.0

############################################################
# 4) Apply Line Offset to Odd Rows Only
############################################################
def apply_line_offset_odd_only_2d(img2d, offset):
    """Shift only the odd rows of a 2D image by the given offset."""
    rows, cols = img2d.shape
    output = np.copy(img2d)
    for r in range(1, rows, 2):
        output[r, :] = shift(output[r, :], offset)
    print(f"  -> Applied line-to-line correction: Shifted odd rows by {offset:.2f} px")
    return output

############################################################
# 5) Save Multi-Page TIFF
############################################################
def save_channel_stack_as_tiff(filename, slice_list):
    """Save a list of 2D slices as a multi-page TIFF file."""
    stack = [np.clip(slc, 0, 65535).astype(np.uint16) for slc in slice_list]
    stack = np.stack(stack, axis=0)

    tiff.imwrite(filename, stack[0], photometric='minisblack')
    for i in range(1, stack.shape[0]):
        tiff.imwrite(filename, stack[i], photometric='minisblack', append=True)

############################################################
# 6) Main Registration Function
############################################################
def zref_mid_frame_register(fileID, animalID, ref_name, plane_spacing, ch, ch_active, fast_z_slices,
                            fast_z_step, tiff_dir, fast_z_slice):
    """Process and register frames from multiple microscopy TIFF files, ensuring depth-to-depth alignment."""
    
    print("\n=== Starting Registration ===\n")
    stack_filename_stem = fileID
    exp_dir_temp = tiff_dir

    image_files = sorted([f for f in os.listdir(exp_dir_temp) if stack_filename_stem in f])
    final_slices = {c: [] for c in range(ch_active)}

    ##################################################
    # STEP 1: Intra-file Registration & Averaging
    ##################################################
    for iFile, file_name in enumerate(image_files):
        file_path = os.path.join(exp_dir_temp, file_name)
        print(f"\nProcessing file {iFile+1}/{len(image_files)}: {file_name}")

        all_pages = read_tif_pages(file_path)

        start_ref = fast_z_slice * ch_active + ch
        step = fast_z_slices * ch_active
        ref_stack = all_pages[:, :, start_ref::step]
        if ref_stack.shape[2] > 1:
            ref_stack = ref_stack[:, :, 1:]  # Skip first frame

        sub_ref = ref_stack[:, :, :max(1, int(0.2 * ref_stack.shape[2]))]
        sub_avg = np.mean(rapid_reg_non_par(sub_ref, sub_ref[:, :, 0]), axis=2)
        ref_stack_reg = rapid_reg_non_par(ref_stack, sub_avg)
        avg_ref_slice = np.mean(ref_stack_reg, axis=2)

        offset_odd_even = calc_odd_even_line_offset_median(avg_ref_slice, num_lines=200)
        corrected_ref = apply_line_offset_odd_only_2d(avg_ref_slice, offset_odd_even)
        final_slices[ch].append(corrected_ref)

        for c_idx in range(ch_active):
            if c_idx == ch:
                continue
            stack_c = all_pages[:, :, fast_z_slice * ch_active + c_idx::step]
            if stack_c.shape[2] > 1:
                stack_c = stack_c[:, :, 1:]
            stack_c_reg = np.mean(rapid_reg_non_par(stack_c, sub_avg), axis=2)
            corrected_c = apply_line_offset_odd_only_2d(stack_c_reg, offset_odd_even)
            final_slices[c_idx].append(corrected_c)

    ##################################################
    # STEP 2: Inter-depth Registration (Fixed)
    ##################################################
    print("\n--- Performing Depth-to-Depth Registration (Mid-Out) ---\n")

    mid_idx = len(image_files) // 2
    shifts = [np.array([0.0, 0.0])] * len(image_files)

    for i in range(mid_idx + 1, len(image_files)):
        shifts[i], _, _ = phase_cross_correlation(final_slices[ch][i - 1], final_slices[ch][i])
        for c_idx in range(ch_active):
            final_slices[c_idx][i] = shift(final_slices[c_idx][i], shifts[i])

    for i in range(mid_idx - 1, -1, -1):
        shifts[i], _, _ = phase_cross_correlation(final_slices[ch][i + 1], final_slices[ch][i])
        for c_idx in range(ch_active):
            final_slices[c_idx][i] = shift(final_slices[c_idx][i], shifts[i])

    ##################################################
    # STEP 3: Saving TIFFs (RESTORED)
    ##################################################
    print("\n--- Saving Output TIFFs ---\n")
    for c_idx in range(ch_active):
        save_channel_stack_as_tiff(os.path.join(tiff_dir, f"{ref_name}{stack_filename_stem}_ch{c_idx}.tif"),
                                   final_slices[c_idx])

    print("\n=== Processing Complete ===\n")


############################################################
# Example Usage
############################################################

if __name__ == "__main__":
    # Example usage with the new param 'fast_z_slice'
    zref_mid_frame_register(
        fileID="z_ref_29-Jan-2025 10_02_30",        # File identifier
        animalID="animalID",    # Animal identifier (not used in path anymore)
        ref_name="ref_name",    # Reference name
        plane_spacing=10,        # Spacing between planes
        ch=0,                    # Reference channel (0-based)
        ch_active=2,            # Total channels
        fast_z_slices=5,        # Number of z-slices in raw data
        fast_z_step=50,         # Step size between z-slices (metadata only)
        tiff_dir="C:\\Temp\\data\\29-Jan-2025 10_02_30_subset",     # Path to TIFF folder
        fast_z_slice=2          # Depth index to extract (0-based)
    )
