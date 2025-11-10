
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio

# Add the project root to sys.path to allow imports
sys.path.append(os.path.abspath('..'))

from runner.experiment import Experiment, ssim
from data_loader.data import PatchSet, get_pair_path_with_masks
from data_loader.utils import save_array_as_tif, load_checkpoint
from model.WGAST import CombinFeatureGenerator, NUM_BANDS

def get_metrics(real_img, fake_img):
    """Calculate image quality metrics (PSNR, SSIM, RMSE)."""
    # Ensure tensors are on CPU and are float type
    real_img = real_img.astype(np.float32)
    fake_img = fake_img.astype(np.float32)

    # RMSE
    rmse = np.sqrt(np.mean((real_img - fake_img) ** 2))

    # PSNR
    mse = np.mean((real_img - fake_img) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        pixel_max = np.max(real_img)
        psnr = 20 * np.log10(pixel_max / np.sqrt(mse))

    # SSIM
    # Convert to torch tensors with shape (B, C, H, W)
    real_img_t = torch.from_numpy(real_img).unsqueeze(0).unsqueeze(0)
    fake_img_t = torch.from_numpy(fake_img).unsqueeze(0).unsqueeze(0)
    ssim_val = ssim(real_img_t, fake_img_t, val_range=np.max(real_img) - np.min(real_img))

    return psnr, ssim_val.item(), rmse

def predict_and_evaluate(model_path, input_dir, output_dir):
    """
    Runs prediction and evaluation on a test dataset using a trained WGAST model.
    """
    # --- Configuration ---
    class Options:
        def __init__(self):
            self.image_size = [400, 400]
            self.patch_size = [32, 32]
            self.patch_stride = 8
            self.ifAdaIN = True
            self.ifAttention = True
            self.ifTwoInput = False
            self.ngpu = 1
            self.cuda = torch.cuda.is_available()

    opt = Options()
    device = torch.device('cuda' if opt.cuda else 'cpu')

    # --- Model Initialization and Loading ---
    print("Initializing and loading the model...")
    generator = CombinFeatureGenerator(ifAdaIN=opt.ifAdaIN, ifAttention=opt.ifAttention, ifTwoInput=opt.ifTwoInput).to(device)
    load_checkpoint(Path(model_path), generator)
    generator.eval()

    # --- Prepare Output ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / 'evaluation_results.csv'
    results_data = []

    # --- Experiment setup for utility functions ---
    exp = Experiment(opt)

    # --- Data Loading ---
    print(f"Scanning for test data in {input_dir}")
    test_dir = Path(input_dir)
    image_dirs = [p for p in test_dir.glob('pair_*') if p.is_dir()]
    if not image_dirs:
        print("Error: No 'pair_*' directories found in the input directory.")
        return

    # --- Main Evaluation Loop ---
    with torch.no_grad():
        for pair_dir in image_dirs:
            print(f"\nProcessing {pair_dir.name}...")
            
            # --- Create Patches for the current pair ---
            patch_set = PatchSet(pair_dir.parent, opt.image_size, opt.patch_size, opt.patch_stride)
            patch_set.image_dirs = [pair_dir] # Process only the current pair
            patch_set.num_im_pairs = 1
            patch_set.num_patches = patch_set.num_patches_x * patch_set.num_patches_y
            
            data_loader = torch.utils.data.DataLoader(patch_set, batch_size=1, shuffle=False, num_workers=1)

            patches = []
            for data in tqdm(data_loader, desc=f"Generating patches for {pair_dir.name}"):
                images, _ = data
                images = [im.to(device) for im in images]
                inputs = images[:-1]
                prediction_patch = generator(inputs)
                prediction_patch = exp.apply_gaussian_blur(prediction_patch, sigma=1.0)
                patches.append(prediction_patch.cpu().numpy())

            # --- Stitch Patches Together ---
            print("Stitching patches into full image...")
            # This logic is adapted from the original experiment.test method
            rows = int((opt.image_size[1] - opt.patch_size[1]) / opt.patch_stride) + 1
            cols = int((opt.image_size[0] - opt.patch_size[0]) / opt.patch_stride) + 1
            scaled_image_size = tuple(i * 3 for i in opt.image_size)
            sum_buffer = np.zeros((NUM_BANDS, *scaled_image_size), dtype=np.float32)
            weight_buffer = np.zeros(sum_buffer.shape, dtype=np.float32)
            
            patch_idx = 0
            for r in range(rows):
                for c in range(cols):
                    row_start = r * (opt.patch_stride * 3)
                    col_start = c * (opt.patch_stride * 3)
                    patch = patches[patch_idx][0]
                    sum_buffer[:, row_start:row_start+patch.shape[1], col_start:col_start+patch.shape[2]] += patch
                    weight_buffer[:, row_start:row_start+patch.shape[1], col_start:col_start+patch.shape[2]] += 1
                    patch_idx += 1

            # Avoid division by zero
            weight_buffer[weight_buffer == 0] = 1
            stitched_prediction = sum_buffer / weight_buffer

            # --- Load Ground Truth Image ---
            pair_paths = get_pair_path_with_masks(pair_dir)
            landsat_t2_path = pair_paths[-1][0] # The last image in the pair is the target
            with rasterio.open(landsat_t2_path) as src:
                ground_truth = src.read().astype(np.float32)
                # Upsample ground truth to match prediction size
                ground_truth_t = torch.from_numpy(ground_truth).unsqueeze(0)
                ground_truth_t = F.interpolate(ground_truth_t, size=scaled_image_size, mode='bicubic', align_corners=False)
                ground_truth = ground_truth_t.squeeze(0).numpy()

            # --- Calculate Metrics ---
            psnr, ssim_val, rmse = get_metrics(ground_truth, stitched_prediction)
            print(f"Metrics for {pair_dir.name}:")
            print(f"  PSNR: {psnr:.4f}")
            print(f"  SSIM: {ssim_val:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            results_data.append([pair_dir.name, psnr, ssim_val, rmse])

            # --- Save Prediction Image ---
            sentinel_prototype_path = pair_paths[2][0] # Use Sentinel as prototype for geo-data
            output_file_name = f"prediction_{pair_dir.name}.tif"
            output_file_path = output_path / output_file_name
            save_array_as_tif(stitched_prediction, str(output_file_path), prototype=str(sentinel_prototype_path))
            print(f"Saved prediction to {output_file_path}")

    # --- Save Metrics to CSV ---
    df = pd.DataFrame(results_data, columns=['ImagePair', 'PSNR', 'SSIM', 'RMSE'])
    df.to_csv(results_file, index=False)
    print(f"\nEvaluation complete. All metrics saved to {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict and evaluate using a trained WGAST model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained generator model (.pth file).')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the test directory containing pair_* subfolders.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory where predictions and results will be saved.')
    
    args = parser.parse_args()

    predict_and_evaluate(args.model_path, args.input_dir, args.output_dir)
