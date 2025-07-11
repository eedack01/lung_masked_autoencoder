import glob
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd
import torch
import torch.nn.functional as F
from lungmask import LMInferer
import os
from tqdm import tqdm
inferer = LMInferer(modelname="R231CovidWeb")

import glob

def resize_image(image, target_shape=(256, 256, 256)):
    """
    Resizes a 3D medical image to the target shape using trilinear interpolation.

    Args:
    image (np.ndarray): Input 3D medical image (height, width, depth).
    target_shape (tuple): Desired shape (height, width, depth). Default is (240, 480, 480).

    Returns:
    np.ndarray: Resized 3D image.
    """
    # Convert to tensor
    tensor = torch.tensor(image, dtype=torch.float32)
    
    # Add batch and channel dimensions to match (batch, channel, depth, height, width)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Calculate the scaling factors based on target shape
    current_shape = tensor.shape[2:]  # (depth, height, width)
    scaling_factors = [
        target_shape[i] / current_shape[i] for i in range(len(target_shape))
    ]
    
    # Resize the image using trilinear interpolation
    resized_tensor = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)
    
    # Remove batch and channel dimensions
    resized_image = resized_tensor.squeeze(0).squeeze(0).numpy()
    
    return resized_image

def crop_image(image, lung_mask):
    # cb = [(np.min(a), np.max(a)) for a, m in zip(np.where(lung_mask), list(lung_mask.shape))]
    # lung_mask = lung_mask > 0
    margin = 32
    cb = [(max(0,np.min(a)-margin), min(np.max(a)+margin,m)) for a,m in zip(np.where(lung_mask), list(lung_mask.shape))]
#     cb = [
#     (0 if a.size == 0 else max(0, np.min(a) - margin),
#      m if a.size == 0 else min(np.max(a) + margin, m))
#     for a, m in zip(np.where(lung_mask), lung_mask.shape)
# ]
    msg = 'crop ratio: '+str(np.prod(image[cb[0][0]:cb[0][1], cb[1][0]:cb[1][1], cb[2][0]:cb[2][1]].shape)/np.prod(image.shape)*100)+'%'
    print(msg)
    image = image[cb[0][0]:cb[0][1], cb[1][0]:cb[1][1], cb[2][0]:cb[2][1]] 
    return image

def normalise(image):
    image[image < -1000] = -1000
    image[image > 200] = 200
    image = (image + 1000) / 1200
    image = image.astype(np.float32)
    return image

output_dir = 'data/mae/numpy'
files = glob.glob('data/mae/unprocessed/*.nii.gz')

count = 0
for f in files:
    file_name = os.path.basename(f)
    new_file_name = file_name.replace('.nii.gz', '.npz')
    file_path = os.path.join(output_dir, new_file_name)
    image = sitk.ReadImage(f)
    spacing = image.GetSpacing()
    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    mean_spacing = np.mean([spacing_x, spacing_y, spacing_z])
    ratio_x = spacing_x / mean_spacing
    ratio_y = spacing_y / mean_spacing
    ratio_z = spacing_z / mean_spacing

    image = sitk.GetArrayFromImage(image)
    print(f'image shape before resampling:{image.shape}')
    image = nd.zoom(image, (ratio_z, ratio_y, ratio_x), order=3)
    print(image.dtype)
    print(f'image shape after resampling:{image.shape}')

    lung_mask = inferer.apply(image)
    lung_mask[lung_mask != 0] = 1
    image = crop_image(image,lung_mask)

    image = resize_image(image)
    lung_mask = inferer.apply(image)
    lung_mask[lung_mask != 0] = 1
    image = normalise(image)
    print(image.shape, lung_mask.shape)
    np.savez_compressed(file_path, image=image, lung_mask=lung_mask)