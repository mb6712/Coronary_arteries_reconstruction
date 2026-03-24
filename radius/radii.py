import numpy as np
import SimpleITK as sitk

def calculate_radius(volumetric_image, centerline_points, intensity_threshold, max_radius):
    """
    Calculate the radius along centerline points based on intensity and bounds.
    """
    radii = []
    print("Calculating radii along centerline points...")
    
    for i, point in enumerate(centerline_points):
        x, y, z = map(int, point)  # Convert to integers for indexing
        
        # Validate bounds of the point
        if not (0 <= x < volumetric_image.shape[1] and 
                0 <= y < volumetric_image.shape[2] and 
                0 <= z < volumetric_image.shape[0]):
            print(f"Error: Point {i + 1} ({x}, {y}, {z}) is out of bounds for the image dimensions {volumetric_image.shape}. Skipping.")
            radii.append(0)  # Append a default radius of 0 for invalid points
            continue

        # Intensity at the centerline point
        intensity_at_point = volumetric_image[z, y, x]  # Note: (z, y, x) indexing for SimpleITK
        print(f"Point {i + 1}: Intensity = {intensity_at_point}")

        # Skip point if intensity is below threshold
        if intensity_at_point < intensity_threshold:
            print(f"  Warning: Intensity below threshold at Point {i + 1}. Radius may be inaccurate.")
            radii.append(0)
            continue

        # Calculate radius
        radius = 0
        for r in range(1, max_radius + 1):
            if not is_within_bounds(x, y, z, r, volumetric_image.shape):
                break
            if not check_intensity(volumetric_image, x, y, z, r, intensity_threshold):
                break
            radius = r
        radii.append(radius)
        print(f"Point {i + 1}: Calculated Radius = {radius}")

    return radii

def is_within_bounds(x, y, z, radius, shape):
    """Check if the radius around the point is within the image bounds."""
    return (x - radius >= 0 and x + radius < shape[2] and  # X-axis
            y - radius >= 0 and y + radius < shape[1] and  # Y-axis
            z - radius >= 0 and z + radius < shape[0])     # Z-axis

def check_intensity(volumetric_image, x, y, z, radius, intensity_threshold):
    """Check the intensity values in a spherical region."""
    coords = [
        (x - radius, y, z),
        (x + radius, y, z),
        (x, y - radius, z),
        (x, y + radius, z),
        (x, y, z - radius),
        (x, y, z + radius),
    ]
    for cx, cy, cz in coords:
        if volumetric_image[cz, cy, cx] < intensity_threshold:
            return False
    return True

# Main script
if __name__ == "__main__":
    volumetric_image_path = "/Users/Imagecas/data/test_data_mhd/3.img.mhd"
    centerline_path = "/Users/Imagecas/output/tracked_points.txt"
    output_radii_path = "/Users/Imagecas/output/radii.txt"

    # Parameters
    intensity_threshold = 0.28  # Adjust as needed
    max_radius = 10  # Maximum radius to calculate

    print("Loading volumetric image...")
    image = sitk.ReadImage(volumetric_image_path)
    volumetric_image = sitk.GetArrayFromImage(image)  # Shape: (Depth, Height, Width)
    print(f"Image loaded with shape: {volumetric_image.shape}")

    print("Loading centerline points...")
    with open(centerline_path, "r") as f:
        centerline_points = [list(map(float, line.strip().split(","))) for line in f]

    print("Calculating radii along centerline points...")
    radii = calculate_radius(volumetric_image, centerline_points, intensity_threshold, max_radius)

    print(f"Saving radii to {output_radii_path}...")
    with open(output_radii_path, "w") as f:
        for radius in radii:
            f.write(f"{radius:.2f}\n")
    print("Radii calculation completed and saved.")
