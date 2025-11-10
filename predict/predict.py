
import torch
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm
import numpy as np

# Add the project root to sys.path to allow imports from other folders
sys.path.append(os.path.abspath('..'))

from runner.experiment import Experiment 
from data_loader.data import PatchSet
from data_loader.utils import save_array_as_tif, load_checkpoint
from model.WGAST import CombinFeatureGenerator

def predict(model_path, input_dir, output_dir):
    """
    Runs the prediction on custom data using a trained WGAST model.

    Args:
        model_path (str): Path to the trained generator model (.pth file).
        input_dir (str): Path to the directory containing the input image pairs.
        output_dir (str): Path to the directory where the predictions will be saved.
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
    
    # --- Device Setup ---
    device = torch.device('cuda' if opt.cuda else 'cpu')

    # --- Model Initialization ---
    print("Initializing the model...")
    generator = CombinFeatureGenerator(ifAdaIN=opt.ifAdaIN, ifAttention=opt.ifAttention, ifTwoInput=opt.ifTwoInput).to(device)
    if opt.cuda and opt.ngpu > 1:
        generator = torch.nn.DataParallel(generator, device_ids=[i for i in range(opt.ngpu)])

    # --- Load Trained Model ---
    print(f"Loading trained model from {model_path}")
    load_checkpoint(Path(model_path), generator)
    generator.eval()

    # --- Prepare Output Directory ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Data Loading and Prediction ---
    print(f"Loading data from {input_dir}")
    test_set = PatchSet(Path(input_dir), opt.image_size, opt.patch_size, opt.patch_stride)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    # --- Experiment setup for utility functions ---
    # We instantiate it to use the apply_gaussian_blur method
    exp = Experiment(opt)

    print("Starting prediction...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Predicting")):
            images = [im.to(device) for im in images]
            
            inputs, _ = images[:-1], images[-1:]
            
            # --- Generate Prediction ---
            prediction = generator(inputs)
            prediction = exp.apply_gaussian_blur(prediction, sigma=1.0) # Apply blur as in training

            # --- Save Prediction ---
            # Assuming one image pair per run for simplicity now
            # Constructing a unique name for the output
            # This part might need adjustment based on your input file naming
            input_file_name = Path(test_set.image_dirs[i]).name
            output_file_name = f"prediction_{input_file_name}.tif"
            output_file_path = output_path / output_file_name

            # To save as tif, we need a prototype file for the metadata.
            # Here we'll have to make an assumption or create a default profile.
            # For now, let's assume the input Sentinel file can be a prototype.
            # This needs to be adapted based on what files are available.
            
            # Let's find a .tif file in the input directory to use as a prototype
            prototype_path = next(Path(input_dir).glob('**/*.tif'), None)

            if prototype_path:
                save_array_as_tif(prediction.cpu().numpy()[0], str(output_file_path), prototype=str(prototype_path))
                print(f"Saved prediction to {output_file_path}")
            else:
                print("Could not find a .tif file in the input directory to use as a prototype for saving.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using a trained WGAST model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained generator model (.pth file).')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing the input image pairs.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory where the predictions will be saved.')
    
    args = parser.parse_args()

    predict(args.model_path, args.input_dir, args.output_dir)
