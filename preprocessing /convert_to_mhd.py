import os
import SimpleITK as sitk

# Path to normalized .nii files
nii_path = '../data/test_data_normalized/'
mhd_output_path = '../data/test_data_mhd/'

# Create output directory if it doesn't exist
os.makedirs(mhd_output_path, exist_ok=True)

for file_name in os.listdir(nii_path):
    if file_name.endswith('.nii'):
        input_file = os.path.join(nii_path, file_name)
        output_file = os.path.join(mhd_output_path, file_name.replace('.nii', '.mhd'))
        
        # Read .nii and write .mhd
        image = sitk.ReadImage(input_file)
        sitk.WriteImage(image, output_file)
        print(f"Converted: {input_file} to {output_file}")
