import numpy as np

# Load the generated ostia points
ostia_points_3D = np.loadtxt("/Users/Imagecas/src/ostia/generated_ostia_points_full_volume_3.txt")

# Select the entry and exit points
entry_point = ostia_points_3D[np.argmin(ostia_points_3D[:, 2])]
exit_point = ostia_points_3D[np.argmax(ostia_points_3D[:, 2])]

# Save the selected points to a file
selected_ostia_points = np.array([entry_point, exit_point])
np.savetxt("selected_ostia_points.txt", selected_ostia_points, fmt="%.6f")

print("Entry Point:", entry_point)
print("Exit Point:", exit_point)
