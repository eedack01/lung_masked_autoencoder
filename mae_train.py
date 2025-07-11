import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from models import MAE
from torch.nn import L1Loss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MAEtrainer(pl.LightningModule):
    """Pretraining on 3D Imaging with Masked Auto Encoder"""

    def __init__(
        self, model_name: str, model_dict: dict,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict

        self.model = MAE(**model_dict)

        self.recon_loss = L1Loss()
        self.recon_patches = []
        # self.save_hyperparameters()
        self.val_outputs = []

    def training_step(self, batch, batch_idx):
        # --------------------------
        image = batch["image"]
        pred_pixel_values, patches, batch_range, masked_indices = self.model(image)
        batch_size = pred_pixel_values.shape[0]
        loss = self.recon_loss(pred_pixel_values, patches[batch_range, masked_indices])
        
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

if __name__ == "__main__":
    # Here we manually load the configuration into the LightningCLI
    cli = LightningCLI(
        model_class=MAEtrainer,    # Your model class
        save_config_kwargs={"overwrite": True}
    )

