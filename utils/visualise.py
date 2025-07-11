import torch
import numpy as np
from mae_pretrain_main import MAEtrainer
import json
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk

from monai.transforms import (
    Compose,
    CropForeground,
    Orientation,
    Resize,
    ToTensor,
)       

from monai.utils import set_determinism   

set_determinism(42)

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
        Resize((128,128,128)),
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
image = np.load('')['ct']
image = np.flip(image, axis=0)
image = np.expand_dims(image, axis=0)  # Add channel dim if needed

# Convert to PyTorch tensor and move to device
image = torch.tensor(image, dtype=torch.float32, device=device)  # Ensure correct dtype

# Load model and move to device
model = MAEtrainer.load_from_checkpoint(
    checkpoint_path='path_here',
    model_name='test',
    model_dict=args.model['model_dict'],
).model.to(device)

# Apply transformations
# Apply transformations and move to device
image = val_transforms(image)  # Ensure val_transforms works with PyTorch tensors
gt = sitk.GetImageFromArray(image.squeeze(0))
sitk.WriteImage(gt, 'test.nii.gz')
image = image.unsqueeze(0).to(device)  # Add batch dimension if needed

# Ensure correct dtype
image = image.to(torch.float32)  # Change to float32 (or float16 if using mixed precision)

# Debugging output
print(f"Final image shape: {image.shape}, dtype: {image.dtype}, device: {image.device}")

# Run the model
pred_pixel_values, patches, batch_range, masked_indices = model(image)

# Print shapes
print(f"pred_pixel_values shape: {pred_pixel_values.shape}")
print(f"patches shape: {patches.shape}")
print(f"batch_range shape: {batch_range.shape}")
print(f"masked_indices shape: {masked_indices.shape}")

PATCH_SIZE = 16
reshaped_pred_patches = pred_pixel_values.view(1, 384, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE).squeeze(0)
ground_truth = patches[batch_range, masked_indices]
ground_truth_reshaped = ground_truth.view(1, 384, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE).squeeze(0)

print(reshaped_pred_patches.shape, ground_truth_reshaped.shape)

print(reshaped_pred_patches[100].cpu().detach().numpy())

reconstructed_patches = patches.clone()  

# Ensure masked_indices is long type and on the same device
masked_indices = masked_indices.to(torch.long)

# Step 2: Replace masked patches with predicted pixel values
reconstructed_patches[:, masked_indices[0]] = pred_pixel_values[:, :]

# Step 3: Reshape the patches back into the original 3D image shape
patch_size = 16  # Patch dimensions (assumes cubic patches)
num_channels = 1  # Modify if using multiple channels (e.g., RGB)

grid_size = 8 

# Reshape patches into a 3D grid
reconstructed_patches = reconstructed_patches.view(1, grid_size, grid_size, grid_size, 
                                                   patch_size, patch_size, patch_size)

# Permute to move patch dimensions into the correct positions
reconstructed_image = reconstructed_patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()

# Reshape to final 3D volume
reconstructed_image = reconstructed_image.view(128, 128, 128)  

# Step 4: Verify the output shape
print("Final reconstructed image shape:", reconstructed_image.shape)  

# Step 5: Save as NIfTI file
sitk.WriteImage(sitk.GetImageFromArray(reconstructed_image.cpu().detach().numpy()), 'test.0.nii.gz')