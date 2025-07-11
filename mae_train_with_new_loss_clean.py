import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from models import MAE
from torch.nn import L1Loss
import torch.nn as nn
import torch
import os
from optimizers import CosineAnnealingWarmupRestarts
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def lung_area_weighted_loss_function(pred_pixel_values, patches, lung_patches_masked_indices, nonlung_patches_masked_indices, weight_nonlung_patches=0.1):
    
    recon_loss_fn = torch.nn.L1Loss(reduction='none')

    # Compute element-wise loss
    per_pixel_loss = recon_loss_fn(pred_pixel_values, patches)

    # Convert index lists into a single tensor
    batch_size = len(lung_patches_masked_indices)

    # Create batch offsets for indexing
    batch_indices_lung = torch.arange(batch_size).repeat_interleave(
        torch.tensor([len(indices) for indices in lung_patches_masked_indices])
    )
    
    batch_indices_nonlung = torch.arange(batch_size).repeat_interleave(
        torch.tensor([len(indices) for indices in nonlung_patches_masked_indices])
    )
    
    lung_indices = torch.cat(lung_patches_masked_indices)
    nonlung_indices = torch.cat(nonlung_patches_masked_indices)

    # Gather the losses 
    loss_lung_area = per_pixel_loss[batch_indices_lung, lung_indices]
    loss_nonlung_area = per_pixel_loss[batch_indices_nonlung, nonlung_indices]

    # Compute final weighted loss
    loss = loss_lung_area.mean() + weight_nonlung_patches * loss_nonlung_area.mean()
    
    return loss

def get_lung_mask_idx(seq, threshold_percentage=0.25):
    all_idx = torch.arange(seq.shape[1], device=seq.device)
    
    # Compute lung area percentage per patch
    perc_lung_area = seq.mean(dim=-1)  # Equivalent to sum()/seq.shape[-1]

    # Identify patches meeting the threshold
    lung_patches_bool = perc_lung_area > threshold_percentage

    # Get indices of lung and non-lung patches
    lung_patches_indices = [lpb.nonzero(as_tuple=True)[0] for lpb in lung_patches_bool]
    non_lung_patches_indices = [all_idx[~lpb] for lpb in lung_patches_bool]

    return lung_patches_indices, non_lung_patches_indices


class PatchReshape3DNoEmbed(nn.Module):
    
    def __init__(self, img_size=(128, 128, 128), patch_size=(16, 16, 16)):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_shape = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = self.patch_shape[0] * self.patch_shape[1] * self.patch_shape[2]

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert (
            D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2]
        ), f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        
        # Reshape into patches
        x = x.view(B, C, 
                   self.patch_shape[0], self.patch_size[0],
                   self.patch_shape[1], self.patch_size[1],
                   self.patch_shape[2], self.patch_size[2])
        
        # Rearrange to (B, num_patches, C * patch_size^3)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(B, self.num_patches, -1)
        
        return x
    

class MAEtrainer(pl.LightningModule):
    """Pretraining on 3D Imaging with Masked Auto Encoder"""

    def __init__(
        self, model_name: str, model_dict: dict,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict
        self.patchify = PatchReshape3DNoEmbed()
        self.model = MAE(**model_dict)
        self.weight_nonlung_patches = 0.01

        self.recon_loss = L1Loss()
        self.recon_patches = []
        # self.save_hyperparameters()
        self.val_outputs = []

    def training_step(self, batch, batch_idx):
        # --------------------------
        image, mask = batch["image"], batch["lung_mask"]
        pred_pixel_values, patches, batch_range, masked_indices = self.model(image)
        with torch.no_grad():
            mask_seq = self.patchify(mask)
            mask_seq = mask_seq[batch_range, masked_indices]
            lung_patches_masked_indices, nonlung_patches_masked_indices = get_lung_mask_idx(mask_seq)
            batch_size = pred_pixel_values.shape[0]

        loss = lung_area_weighted_loss_function(pred_pixel_values, patches[batch_range, masked_indices], lung_patches_masked_indices, nonlung_patches_masked_indices, self.weight_nonlung_patches)
        self.log("train/l1_loss", loss, batch_size=batch_size, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # --------------------------
        image = batch["image"]
        pred_pixel_values, patches, batch_range, masked_indices = self.model(image)
        batch_size = pred_pixel_values.shape[0]
        loss = self.recon_loss(pred_pixel_values, patches[batch_range, masked_indices])

        self.log("val/l1_loss", loss, batch_size=batch_size, sync_dist=True)

        self.val_outputs.append({
            "val_loss": loss,
            "val_number": batch_size
        })

        return {"val_loss": loss, "val_number": batch_size}
    
    def on_validation_epoch_end(self):
        # Compute the mean validation loss after each epoch
        val_loss = 0
        num_items = 0

        # Loop through the saved outputs and aggregate the results
        for output in self.val_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_loss = val_loss / num_items  # Average validation loss per item
        self.log("val/l1_loss_avg", mean_val_loss, sync_dist=True)

        # Log the hyperparameters and metrics
        self.logger.log_hyperparams(
            params={
                "model": self.model_name,
                **self.model_dict,
                "data": self.trainer.datamodule.json_path,
                "ds_ratio": self.trainer.datamodule.downsample_ratio,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
            },
            metrics={"l1_loss": mean_val_loss},
        )

        # Clear the stored outputs to free memory
        self.val_outputs = []
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=0.05)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=198000,
                cycle_mult=1.0,
                max_lr=0.0003,
                min_lr=0.0,
                warmup_steps=19800,
                gamma=1.0)
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Ensures it updates per batch, not per epoch
                "frequency": 1
            }
        }

if __name__ == "__main__":
    # Here we manually load the configuration into the LightningCLI
    cli = LightningCLI(
        model_class=MAEtrainer,    # Your model class
        save_config_kwargs={"overwrite": True}
    )

