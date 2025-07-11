import torch
from dataset import MAEDataModule
import argparse
import json
from torch.utils.data import DataLoader
from multiprocessing import Pool
from functools import partial

import numpy as np
from mae_pretrain_main import MAEtrainer
import json
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch.nn.functional as F
from pytorch_msssim import ssim

def reconstruct_imge(patches):
    reconstructed_patches = patches.clone()  

    # Step 3: Reshape the patches back into the original 3D image shape
    patch_size = 16 # PATCH_SIZE 16 in our case Patch dimensions (assumes cubic patches)

    grid_size = 8 

    # Reshape patches into a 3D grid
    reconstructed_patches = reconstructed_patches.view(1, grid_size, grid_size, grid_size, 
                                                    patch_size, patch_size, patch_size)

    # Permute to move patch dimensions into the correct positions
    reconstructed_image = reconstructed_patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()

    # Reshape to final 3D volume
    reconstructed_image = reconstructed_image.view(128, 128, 128)
    return reconstructed_image

def calculate_metrics_3d(volume1, volume2):
    """
    Compute MAE, MSE, and SSIM between two 3D medical volumes.

    Args:
        volume1 (torch.Tensor): First input volume, shape (D, H, W).
        volume2 (torch.Tensor): Second input volume, shape (D, H, W).

    Returns:
        dict: Dictionary containing MAE, MSE, and SSIM scores.
    """
    assert volume1.shape == volume2.shape, "Input volumes must have the same shape"

    # Convert to float for precision
    volume1, volume2 = volume1.float(), volume2.float()

    # MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(volume1 - volume2))

    # MSE (Mean Squared Error)
    mse = torch.mean((volume1 - volume2) ** 2)

    # Compute SSIM slice-by-slice (assuming depth D = 128)
    ssim_scores = []
    for i in range(volume1.shape[0]):  # Iterate over slices
        slice1 = volume1[i].unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)
        slice2 = volume2[i].unsqueeze(0).unsqueeze(0)
        ssim_scores.append(ssim(slice1, slice2, data_range=1.0).item())

    avg_ssim = sum(ssim_scores) / len(ssim_scores)  # Average over slices

    return {"MAE": mae.item(), "MSE": mse.item(), "SSIM": avg_ssim}

def eval(model_path, dataset, device, args):
    """Evaluate a single model."""
    # Load model and move to device
    model = MAEtrainer.load_from_checkpoint(
        checkpoint_path=model_path,
        model_name='test',
        model_dict=args.model['model_dict'],
    ).model.to(device)

    model.eval()

    # Initialize metric accumulators
    whole_image_metrics = {'MAE': 0.0, 'MSE': 0.0, 'SSIM': 0.0}
    lung_metrics = {'MAE': 0.0, 'MSE': 0.0, 'SSIM': 0.0}

    val_loader = dataset.val_dataloader()
    len_dataset = len(val_loader)

    for batch in val_loader:
        image, mask = batch["image"], batch["lung_mask"]
        with torch.no_grad():
            image = image.to(device)
            mask = mask.to(device)
            mask = mask.squeeze(dim=1)
            print(mask.shape)
            pred_pixel_values, patches = model.predict(image)
            gt = reconstruct_imge(patches)
            image = reconstruct_imge(pred_pixel_values)
            print(gt.shape)
            print(image.shape)
            lung_gt = gt * mask
            lung_pred = image * mask
            print(lung_gt.shape)
            print(lung_pred.shape)

            # Calculate metrics
            whole_image_batch_metrics = calculate_metrics_3d(gt, image)
            lung_batch_metrics = calculate_metrics_3d(lung_gt, lung_pred)

            # Accumulate metrics
            for key in whole_image_metrics:
                whole_image_metrics[key] += whole_image_batch_metrics[key]
            for key in lung_metrics:
                lung_metrics[key] += lung_batch_metrics[key]

    # Compute average metrics
    avg_whole_image_metrics = {key: value / len_dataset for key, value in whole_image_metrics.items()}
    avg_lung_metrics = {key: value / len_dataset for key, value in lung_metrics.items()}

    return model_path, avg_whole_image_metrics, avg_lung_metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-c",
        "--config-file",
        default="/config/mae_config_new.json",
        help="config json file that stores hyper-parameters",
    )
    args = parser.parse_args()

    # Load config
    config_dict = json.load(open(args.config_file, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    dataset = MAEDataModule(
        json_path="/config/mae_config_new.json",
        downsample_ratio=[1.5, 1.5, 2.0],
        batch_size=64,
        val_batch_size=32,
        num_workers=0,
        data_path="/data/mae/numpy_128",
        test_size=0.2,
        random_state=42
    )
    dataset.setup()

    # List of models to evaluate
    models = [
        '/logs/mae/vitmae_128_7_w_0.1/checkpoints/best_0.1.ckpt',
        '/ild_prognosis/logs/mae/vitmae_128_7_w_0.1/checkpoints/last_0.1.ckpt',
        '/ild_prognosis/logs/mae/vitmae_128_8_w_0.01/checkpoints/best_0.01.ckpt',
        '/logs/mae/vitmae_128_8_w_0.01/checkpoints/last_0.01.ckpt',
        '/logs/mae/vitmae_128_9_w_1.0/checkpoints/best_1.0.ckpt',
        '/logs/mae/vitmae_128_9_w_1.0/checkpoints/last_1.0.ckpt',
        '/logs/mae/vitmae_128_10_w_0.0/checkpoints/best_0.0.ckpt',
        '/logs/mae/vitmae_128_10_w_0.0/checkpoints/last_0.0.ckpt'
    ]

    # Evaluate models sequentially
    results = []
    for model_path in models:
        print(f"Evaluating model: {model_path}")
        result = eval(model_path, dataset=dataset, device=device, args=args)
        results.append(result)

        # Print results for the current model
        print(f"Model: {result[0]}")
        print("Whole Image Metrics:", result[1])
        print("Lung Metrics:", result[2])
        print()

if __name__ == "__main__":
    main()