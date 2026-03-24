import os
import SimpleITK as sitk
import numpy as np

# Define the folder where your label files are located
label_folder = "/Users/Practical/imagecas_project/data/test_data_mhd"
output_folder = "/Users/Practical/imagecas_project/data/seed_points"  # Folder to save seed points

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over all .label.mhd files in the directory
for filename in os.listdir(label_folder):
    if filename.endswith("label.mhd"):
        label_path = os.path.join(label_folder, filename)
        
        # Load the label image
        label_image = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(label_image)

        # Check the dimensions of the label array
        print(f"Processing {filename} - Label array shape: {label_array.shape}")

        # List to store valid seed points
        seed_points = []

        # Iterate through the label array to find the seed points (i.e., where the label is 1)
        for z in range(label_array.shape[0]):
            for y in range(label_array.shape[1]):
                for x in range(label_array.shape[2]):
                    if label_array[z, y, x] == 1:  # Find the centerline (label is 1)
                        seed_points.append([x, y, z])  # Save the point as (x, y, z)

        # Print out seed points before filtering nan values
        print(f"Seed points before filtering 'nan' values: {len(seed_points)} points")

        # Filter out points with 'nan' values
        filtered_seed_points = [point for point in seed_points if not any(np.isnan(point))]

        # Print out the filtered seed points
        print(f"Filtered seed points (without 'nan' values): {len(filtered_seed_points)} points")

        # Write the filtered seed points to a file
        seed_filename = filename.replace("label.mhd", "seed_points.txt")
        seed_file_path = os.path.join(output_folder, seed_filename)
        with open(seed_file_path, "w") as f:
            for point in filtered_seed_points:
                f.write(f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}\n")

        print(f"Seed points saved to {seed_file_path}")
