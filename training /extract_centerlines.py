import os
from skimage.morphology import skeletonize
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Directories
image_dir = r"/Users/Practical/imagecas_project/data/test_data_mhd"
label_dir = r"/Users/Practical/imagecas_project/data/test_data_mhd"
output_centerline_dir = r"/Users/Practical/imagecas_project/data/centerlines"

os.makedirs(output_centerline_dir, exist_ok=True)

def process_single_label(label_path, output_path, max_points=1000):
    """Process a single label file to extract centerline points."""
    label_image = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label_image)

    # Binary mask
    binary_mask = (label_array > 0).astype(np.uint8)

    # Initialize skeleton 3D array
    skeleton_3d = np.zeros_like(binary_mask, dtype=np.uint8)

    # Apply skeletonization slice by slice
    for z in range(binary_mask.shape[0]):
        skeleton_3d[z] = skeletonize(binary_mask[z])

    # Extract skeleton points (coordinates of non-zero points in the skeleton)
    skeleton_points = np.argwhere(skeleton_3d > 0)

    # Compute distance transform (Euclidean Distance Transform)
    distance_map = distance_transform_edt(binary_mask)

    # Get radii for each skeleton point
    radii = distance_map[skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2]]

    # Combine (x, y, z) and radii
    centerline_data = np.hstack((skeleton_points, radii[:, np.newaxis]))  # Shape: (N, 4)

    # Downsample points if necessary
    downsampled_data = random.sample(centerline_data.tolist(), k=min(len(centerline_data), max_points))
    downsampled_data = np.array(downsampled_data)

    # Save centerline points to a .txt file
    np.savetxt(output_path, downsampled_data, fmt="%.6f", header="x y z radius", comments="")
    print(f"Saved centerline to {output_path}")

def plot_centerline(binary_mask, centerline_data):
    """Plot the centerline on the 3D structure."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the original binary mask for context
    structure_points = np.argwhere(binary_mask > 0)
    ax.scatter(structure_points[:, 0], structure_points[:, 1], structure_points[:, 2], alpha=0.1, label="Structure")

    # Overlay the centerline points
    ax.scatter(
        centerline_data[:, 0],
        centerline_data[:, 1],
        centerline_data[:, 2],
        c=centerline_data[:, 3],
        cmap="viridis",
        label="Centerline",
    )
    ax.legend()
    plt.show()

# Process all labels
for i in range(1, 201):  # Assuming 200 images/labels
    label_path = os.path.join(label_dir, f"{i}.label.mhd")
    centerline_txt_path = os.path.join(output_centerline_dir, f"{i}_centerline.txt")

    if os.path.exists(label_path):
        try:
            process_single_label(label_path, centerline_txt_path)
        except Exception as e:
            print(f"Error processing label {i}: {e}")
    else:
        print(f"Label file missing for {i}")