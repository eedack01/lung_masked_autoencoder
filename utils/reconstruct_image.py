
def reconstruct_imge(patches):
    reconstructed_patches = patches.clone()  

    # Step 3: Reshape the patches back into the original 3D image shape
    batch_size = patches.shape[0]
    patch_size = 16  # Patch dimensions (assumes cubic patches)

    grid_size = 8 

    # Reshape patches into a 3D grid
    reconstructed_patches = reconstructed_patches.view(batch_size, grid_size, grid_size, grid_size, 
                                                    patch_size, patch_size, patch_size)

    # Permute to move patch dimensions into the correct positions
    reconstructed_image = reconstructed_patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()

    # Reshape to final 3D volume
    reconstructed_image = reconstructed_image.view(batch_size, 128, 128, 128)
    return reconstructed_image


