input_file = "/users/practical/imagecas_project/data/seed_points/3.seed_points.txt"
output_file = "/users/practical/imagecas_project/data/seed_points/3.seed_points_corrected.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Remove trailing commas and extra spaces
        fixed_line = line.replace(",", "").strip()
        outfile.write(fixed_line + "\n")

print(f"File corrected and saved as {output_file}")
