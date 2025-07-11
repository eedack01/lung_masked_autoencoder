import torch
import numpy as np
from mae_pretrain_main import MAEtrainer
import json
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch.nn.functional as F
from pytorch_msssim import ssim

from monai.transforms import (
    Compose,
    CropForeground,
    Orientation,
    Resize,
    ToTensor,
)       

from monai.utils import set_determinism   

# set_determinism(42)

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "-c",
    "--config-file",
    default="ild_finetune.json",
    help="config json file that stores hyper-parameters",
)
args = parser.parse_args()

val_transforms = Compose(
    [
        ToTensor(),
    ]
)

# print_config()
torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)
torch.autograd.set_detect_anomaly(True)
resize = Resize((256,256,256))
config_dict = json.load(open(args.config_file, "r"))

for k, v in config_dict.items():
    setattr(args, k, v)

# Set device
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

# Load image (assuming it's a NumPy array)
image = np.load('ild_138.npz')['image']
print(image.shape)
exit()
mask = np.load('ild_138.npz')['lung_mask'].astype(np.float16)
mask = np.squeeze(mask, axis=0)
print(image.shape)
image = np.expand_dims(image, axis=0)  # Add channel dim if needed

# Convert to PyTorch tensor and move to device
image = torch.tensor(image, dtype=torch.float32, device=device)  # Ensure correct dtype

# Load model and move to device
model = MAEtrainer.load_from_checkpoint(
    checkpoint_path='best_0.1.ckpt',
    model_name='test',
    model_dict=args.model['model_dict'],
).model.to(device)

# Apply transformations
# Apply transformations and move to device
image = val_transforms(image)  # Ensure val_transforms works with PyTorch tensors
# image2 = image2.unsqueeze(dim=0)
gt = sitk.GetImageFromArray(image.squeeze(0))
sitk.WriteImage(gt, 'test.nii.gz')
# image = image.unsqueeze(0).to(device)  # Add batch dimension if needed

# Ensure correct dtype
image = image.to(torch.float32)  # Change to float32 (or float16 if using mixed precision)

# Debugging output
print(f"Final image shape: {image.shape}, dtype: {image.dtype}, device: {image.device}")

# Run the model
pred_pixel_values, patches = model.predict(image)

print(pred_pixel_values.shape, patches.shape)

# Print shapes
print(f"pred_pixel_values shape: {pred_pixel_values.shape}")
print(f"patches shape: {patches.shape}")

PATCH_SIZE = 16
num_patches = 512
reshaped_pred_patches = pred_pixel_values.view(1, num_patches, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE).squeeze(0)
ground_truth_reshaped = patches.view(1, num_patches, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE).squeeze(0)


print(reshaped_pred_patches.shape, ground_truth_reshaped.shape)

reshaped_pred_patches = reshaped_pred_patches.cpu().detach().numpy()
ground_truth_reshaped = ground_truth_reshaped.cpu().detach().numpy()

slice_patches = np.arange(100, 250)
random_indices = np.random.choice(slice_patches, 8, replace=False)

selected_pred_patches = reshaped_pred_patches[random_indices]
pred_slices = selected_pred_patches[:, 0, :, :]  # First slice of each


selected_gt_patches = ground_truth_reshaped[random_indices]
gt_slices = selected_gt_patches[:, 0, :, :]  # First slice of each


combined_pred = np.hstack(pred_slices)  # Predicted patches in one row
combined_gt = np.hstack(gt_slices)      # Ground truth patches in one row

final_combined = np.vstack([combined_pred, combined_gt])

plt.figure(figsize=(8, 32))
plt.imshow(final_combined, cmap="gray")
plt.axis("off")
plt.title("Predicted (Top) vs Ground Truth (Bottom)")
plt.show()

print(reshaped_pred_patches[100].cpu().detach().numpy())

sitk.WriteImage(sitk.GetImageFromArray(reshaped_pred_patches[100].cpu().detach().numpy()), 'test.nii.gz')
sitk.WriteImage(sitk.GetImageFromArray(ground_truth_reshaped[100].cpu().detach().numpy()), 'test1.nii.gz')