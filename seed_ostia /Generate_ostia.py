import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops

# Load the image data
image_file = "/Users/Imagecas/data/test_data_mhd/3.img.mhd"
image = sitk.ReadImage(image_file)
image_array = sitk.GetArrayFromImage(image)

# Intensity threshold (adjust based on your image intensity range)
threshold = 0.25

# Initialize list to store 3D ostia points
ostia_points_3D = []

# Process each slice
for z_index in range(image_array.shape[0]):
    slice_image = image_array[z_index, :, :]
    binary_image = slice_image > threshold

    # Label connected components
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)

    # Collect centroids of connected regions
    for region in regions:
        if region.area > 10:  # Ignore very small regions
            centroid = region.centroid  # (row, col)
            ostia_points_3D.append([centroid[1], centroid[0], z_index])  # Append x, y, z

# Convert to NumPy array
ostia_points_3D = np.array(ostia_points_3D)

# Save ostia points
np.savetxt("generated_ostia_points_full_volume_3.txt", ostia_points_3D, fmt="%.2f")
print(f"Generated {len(ostia_points_3D)} 3D ostia points across the entire volume.")
