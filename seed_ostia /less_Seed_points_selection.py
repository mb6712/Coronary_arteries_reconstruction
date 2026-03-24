import numpy as np
from sklearn.cluster import KMeans

def select_valid_seed_points_for_ostia(seed_points_file, ostia_coords, distance_threshold, num_seed_points):
    """
    Automatically selects valid seed points for each ostia.

    Parameters:
    - seed_points_file (str): Path to the seed points file (Nx3 format: [x, y, z]).
    - ostia_coords (list): List of tuples for ostia coordinates [(x1, y1, z1), (x2, y2, z2), ...].
    - distance_threshold (float): Maximum allowed distance from the ostia to filter seed points.
    - num_seed_points (int): Number of seed points to select after clustering.

    Returns:
    - valid_seed_points_dict (dict): Dictionary with ostia as keys and selected seed points as values.
    """

    # Load seed points (space-separated format)
    seed_points = np.loadtxt(seed_points_file, delimiter=" ")  # Corrected to use space delimiter
    valid_seed_points_dict = {}

    for idx, ostia in enumerate(ostia_coords):
        # Convert ostia coordinates to numpy array
        ostia_point = np.array(ostia)

        # Step 1: Filter seed points by distance from the current ostia
        distances = np.linalg.norm(seed_points - ostia_point, axis=1)
        filtered_seed_points = seed_points[distances <= distance_threshold]

        if len(filtered_seed_points) == 0:
            print(f"No seed points found within distance threshold for Ostia {idx + 1}: {ostia}")
            continue

        print(f"Ostia {idx + 1}: Filtered {len(filtered_seed_points)} seed points.")

        # Step 2: Subsample seed points using KMeans clustering
        kmeans = KMeans(n_clusters=num_seed_points, random_state=0).fit(filtered_seed_points)
        subsampled_seed_points = kmeans.cluster_centers_

        print(f"Ostia {idx + 1}: Selected {num_seed_points} seed points.")
        valid_seed_points_dict[ostia] = subsampled_seed_points

        # Save valid seed points for this ostia
        output_filename =  f"/Users/Imagecas/data/seed_points/valid_seed_points_ostia_3_{idx + 1}.txt"

        np.savetxt(output_filename, subsampled_seed_points, delimiter=" ", fmt="%.2f")
        print(f"Valid seed points for Ostia {idx + 1} saved to '{output_filename}'.")

    return valid_seed_points_dict


# Input parameters
seed_points_file = "/Users/Imagecas/data/seed_points/3.seed_points_corrected.txt"  # Path to your space-separated seed points file
ostia_coords = [
    (45.35, 6.47, 0.00),  # Ostia 1
    (16.75, 2.52, 274.00)  # Ostia 2
]
distance_threshold = 400.0  # Adjust the distance threshold to a larger value if needed
num_seed_points = 250  # Number of seed points to select for each ostia

# Run the function
valid_seed_points_dict = select_valid_seed_points_for_ostia(
    seed_points_file, ostia_coords, distance_threshold, num_seed_points
)

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all seed points
    seed_points = np.loadtxt(seed_points_file, delimiter=" ")  # Reload seed points for visualization
    ax.scatter(seed_points[:, 0], seed_points[:, 1], seed_points[:, 2], c='gray', label="All Seed Points", alpha=0.5)
    
    # Plot seed points for each ostia
    for idx, ostia in enumerate(ostia_coords):
        if ostia in valid_seed_points_dict:
            valid_points = valid_seed_points_dict[ostia]
            ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], label=f"Ostia {idx + 1} Valid Points")
        
        # Plot the ostia point
        ax.scatter(*ostia, c='blue', label=f"Ostia {idx + 1}", s=100)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.title("Seed Points Selection for Multiple Ostia")
    plt.show()

except ImportError:
    print("Matplotlib is not installed. Skipping visualization.")
