import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from tqdm import tqdm

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(32)
        self.conve = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bne = nn.BatchNorm3d(32)
        self.convf = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bnf = nn.BatchNorm3d(32)
        self.conv5 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(32)
        self.conv6 = nn.Conv3d(32, 64, kernel_size=1)
        self.bn6 = nn.BatchNorm3d(64)
        self.conv6_c = nn.Conv3d(64, 500, kernel_size=1)
        self.conv6_r = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bne(self.conve(x)))
        x = torch.relu(self.bnf(self.convf(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        centerline = self.conv6_c(x)
        radius = self.conv6_r(x)
        return centerline, radius

def load_image(impath):
    print(f"Loading image from {impath}")
    image = sitk.ReadImage(impath)
    return sitk.GetArrayFromImage(image)

def load_points(filepath):
    print(f"Loading points from {filepath}")
    # Automatically detect whether the file is comma or space-delimited
    with open(filepath, 'r') as f:
        sample_line = f.readline()
        delimiter = ',' if ',' in sample_line else ' '
    points = np.loadtxt(filepath, delimiter=delimiter)
    return points


def extract_patch(image, center, patch_size):
    z, y, x = map(int, center)
    half = patch_size // 2
    z_min, z_max = max(0, z - half), min(image.shape[0], z + half)
    y_min, y_max = max(0, y - half), min(image.shape[1], y + half)
    x_min, x_max = max(0, x - half), min(image.shape[2], x + half)
    if z_max - z_min <= 0 or y_max - y_min <= 0 or x_max - x_min <= 0:
        print(f"Invalid patch region for center {center}: z [{z_min}:{z_max}], y [{y_min}:{y_max}], x [{x_min}:{x_max}]")
        return None
    patch = np.zeros((patch_size, patch_size, patch_size), dtype=image.dtype)
    patch[
        : z_max - z_min,
        : y_max - y_min,
        : x_max - x_min,
    ] = image[z_min:z_max, y_min:y_max, x_min:x_max]
    return patch[np.newaxis, np.newaxis, :, :, :]  # Add batch and channel dims

def track(args, image, seeds, ostia):
    cnn_model = CNNModel()
    print(f"Loading CNN model from {args.tracknet}")
    cnn_model.load_state_dict(torch.load(args.tracknet, map_location=torch.device('cpu')))
    cnn_model.eval()

    tracked_points = []
    patch_size = 32

    for seed in tqdm(seeds, desc="Processing Seeds"):
        patch = extract_patch(image, seed, patch_size)
        if patch is None:
            print(f"Skipping seed {seed} due to invalid patch.")
            continue
        patch = torch.tensor(patch, dtype=torch.float32)

        with torch.no_grad():
            centerline, radius = cnn_model(patch)
            centerline_prob = torch.softmax(centerline, dim=1)
            prob = centerline_prob[0, 1].max().item()
            radius_value = radius.max().item()

            print(f"Seed {seed}, Centerline Probability: {prob}, Radius: {radius_value}")

            if prob > args.entropythreshold:
                tracked_points.append(seed)

    print(f"Tracking complete. Found {len(tracked_points)} tracked points.")
    save_results(tracked_points, args.outdir)

def save_results(points, outdir):
    os.makedirs(outdir, exist_ok=True)
    output_file = os.path.join(outdir, "tracked_points.txt")
    np.savetxt(output_file, points, fmt='%0.3f', delimiter=',')
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track vessels using CNN-based tracker.")
    parser.add_argument("--impath", required=True, help="Path to the input image (MHD format).")
    parser.add_argument("--outdir", required=True, help="Directory to save the output.")
    parser.add_argument("--ostia", required=True, help="Path to the ostia points file.")
    parser.add_argument("--seeds", nargs="+", required=True, help="Paths to the seed points files (space-separated).")
    parser.add_argument("--tracknet", required=True, help="Path to the CNN model file (tracker.pt).")
    parser.add_argument("--entropythreshold", type=float, default=0.2, help="Entropy threshold value.")

    args = parser.parse_args()
    image = load_image(args.impath)
    seeds = []
    for seed_file in args.seeds:
        seeds.extend(load_points(seed_file))
    ostia = load_points(args.ostia)
    track(args, image, seeds, ostia)
