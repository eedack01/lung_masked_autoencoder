import os
import argparse
import json
import logging
import sys
from pathlib import Path
import itertools

import torch
from monai.config import print_config
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from mae_pretrain_main import MAEtrainer
import numpy as np
from time import time
from datetime import datetime, timedelta
from models import DownstreamModel, VisionTransformer3D
import pandas as pd
from tqdm import tqdm
import copy
from dataset import ILDDiagnosisDataset, IntactDatasetNew
from utils.class_weights import get_class_weights
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from timm.data.mixup import Mixup
from utils.custom_scheduler import CosineAnnealingWarmupRestarts
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from monai.transforms import (
    Compose,
    CropForeground,
    Orientation,
    RandShiftIntensity,
    RandRotate90,
    ToTensor,
    RandFlip,
    RandGaussianNoise,
)       

import warnings
warnings.filterwarnings("ignore")

def main(lr, wd, bs, seed):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-c",
        "--config-file",
        default="/storage/homefs/ed22q093/ild_prognosis/config/ild/ild_finetune.json",
        help="config json file that stores hyper-parameters",
    )
    args = parser.parse_args()

    # print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    config_dict = json.load(open(args.config_file, "r"))

    for k, v in config_dict.items():
        setattr(args, k, v)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device}")
    
    # Log hyperparameters
    logging.info(f"Hyperparameters set | lr: {lr}, weight decay: {wd}, batch size: {bs}")

    set_determinism(42)

    # Encoder setup
    ct_encoder = MAEtrainer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model['model_name'],
        model_dict=args.model['model_dict'],
    ).model.to(device)
    model = DownstreamModel(model=ct_encoder, classes=args.classes).to(device)
    if args.linear_probe:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.linear.parameters():
            param.requires_grad = True

        for param in model.layer_norm.parameters():
            param.requires_grad = True

        for param in model.cls_token:
            param.requires_grad = True

    # Step 1: set data loader
    train_transforms = Compose(
        [
            # Orientation(axcodes="RAS"),  # Ensure consistent orientation
            # CropForeground( source_key="image"),  # Crop foreground
            # RandFlip(spatial_axis=[0], prob=0.10),
            # RandFlip(spatial_axis=[1], prob=0.10),
            # RandFlip(spatial_axis=[2], prob=0.10),
            # RandRotate90( prob=0.10, max_k=3),
            ToTensor(),  # Convert to tensor
        ]
    )
    val_transforms = Compose(
        [
            # Orientation( axcodes="RAS"),
            # CropForeground( source_key="image"),
            ToTensor(),
        ]
    )
    
    train_dataset = IntactDatasetNew(
                 data_path=args.train['root_dir'], 
                 split_size = args.split_size, 
                 transform=train_transforms, 
                 split=args.train['mode'],
                 binary=args.binary,
                 random_state=seed
    )
    val_dataset = IntactDatasetNew(
                 data_path=args.val['root_dir'], 
                 split_size = args.split_size, 
                 transform=val_transforms, 
                 split=args.val['mode'],
                 binary=args.binary,
                 random_state=seed
    )
    
    dataloaders = {
        'train': DataLoader(dataset=train_dataset,
                              batch_size=bs, 
                              shuffle=args.train['shuffle'], 
                              num_workers=args.train['num_workers'],
                            #   drop_last=True
                              ),
        'val': DataLoader(dataset=val_dataset,
                              batch_size=bs, 
                              shuffle=args.val['shuffle'], 
                              num_workers=args.val['num_workers'],
                            #   drop_last=True
                              )
    }
    
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    # optimizer setup
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    
    total_iters = len(dataloaders['train']) * args.train['num_epochs']
    print(total_iters, total_iters * 0.1) 
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_iters)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=total_iters,
                                                  cycle_mult=1.0,
                                                  max_lr=lr,
                                                  min_lr=0.0,
                                                  warmup_steps=int(total_iters * 0.1),
                                                  gamma=1.0)
    
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.train['num_epochs'])
    class_weights = torch.tensor([3.8889, 5.0000, 5.8333, 2.6923]).to(device)
    if args.mixup:
        mixup_fn = Mixup(
        mixup_alpha= 1.0,
        cutmix_alpha= 0.0,
        cutmix_minmax= None,
        prob= 1.0,
        switch_prob= 0.0,
        mode= 'batch',
        label_smoothing= 0.1,
        num_classes= args.classes)
    else:
        mixup_fn = None

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        print('using soft target cross entropy')
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        print('using label smoothing cross entropy')
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    elif args.binary:
        print('using cross entropy with zero class weights')
        criterion = torch.nn.CrossEntropyLoss()
    else:
        print('using cross entropy with class weights')
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    

    if args.binary:
        print('using cross entropy with zero class weights')
        val_criterion = torch.nn.CrossEntropyLoss()
    else:
        val_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    best_f1 = -np.inf
    best_model_wts = None
    df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "balanced_acc", "f1"])
    t0 = time()
    for epoch in range(args.train['num_epochs']):
        epoch_data = {"epoch": epoch + 1}
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                if (epoch) % args.check_val_every_n_epoch != 0:
                    continue
                model.eval()
                all_preds = []
                all_targets = []
                
            running_loss = 0.0

            for batch in tqdm(dataloaders[phase], desc=f"Epoch {epoch + 1}/{args.train['num_epochs']} ({phase})"):
                images = batch[0].float().to(device)
                targets = batch[1].long().to(device)

                if phase == 'train' and mixup_fn != None:
                    images, targets = mixup_fn(images, targets)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    preds = model(images, None)
                    if phase == 'train':
                        loss = criterion(preds, targets)
                    else:
                        loss = val_criterion(preds, targets)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
 
                running_loss += loss.item() * images.size(0)
                if phase == 'val':
                    all_preds.extend(preds.argmax(dim=-1).cpu().numpy())  
                    all_targets.extend(targets.cpu().numpy())
                
                    print(all_preds)

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_data[f"{phase}_loss"] = epoch_loss
            logging.info(f"Epoch {epoch + 1} - {phase} Loss: {epoch_loss:.4f}")
            
            if phase == "val":
                f1 = f1_score(all_targets, all_preds, average='weighted')
                balanced_acc = balanced_accuracy_score(all_targets, all_preds)
                acc = accuracy_score(all_targets, all_preds)
                
                logging.info(f"Epoch {epoch + 1} - F1 Score: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, acc: {acc}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_model_wts = copy.deepcopy(model.state_dict())

                epoch_data["f1"] = f1
                epoch_data["balanced_acc"] = balanced_acc
                epoch_data["best_f1"] = best_f1  # Track best F1 score
                
        # Save the last model for the current epoch
        # if (epoch) % 1 == 0:
        #     # Save the last model for the current epoch
        #     last_model_file = Path(args.model_dir) / f"last_model_epoch_{epoch + 1}_{str(lr)}_{str(wd)}_{str(bs)}.pth"
        #     torch.save(model.state_dict(), last_model_file)
        #     logging.info(f"Saved last model for epoch {epoch + 1} at {last_model_file}")
        #     # Save the last model for the current epoch
        last_model_file = Path(args.model_dir) / f"last_model_epoch_{str(lr)}_{str(wd)}_{str(bs)}_{str(seed)}.pth"
        torch.save(model.state_dict(), last_model_file)
        logging.info(f"Saved last model for epoch {epoch + 1} at {last_model_file}")

        # Add metrics to DataFrame
        print(epoch_data)
        df = pd.concat([df, pd.DataFrame([epoch_data])], ignore_index=True)

    # Save the best model after training
    model.load_state_dict(best_model_wts)
    best_model_file = Path(args.model_dir) / f"best_model_{datetime.now().strftime('%Y%m%d')}_{str(lr)}_{str(wd)}_{str(bs)}_{str(seed)}.pth"
    torch.save(best_model_wts, best_model_file)
    logging.info(f"Saved best model at {best_model_file}")

    # Save the training log
    training_log_file = Path(args.model_dir) / f"training_log_{datetime.now().strftime('%Y%m%d')}_{str(lr)}_{str(wd)}_{str(bs)}_{str(seed)}.csv"
    df.to_csv(training_log_file, index=False)
    logging.info(f"Training log saved at {training_log_file}")

    logging.info(f"Training complete! Time elapsed: {timedelta(seconds=time() - t0)}")

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # learning_rates = [1e-4, 1e-3, 1e-5]
    # weight_decays = [1e-4, 1e-3, 1e-5]
    # batch_sizes = [8, 16, 32, 48]

    # learning_rates = [1e-4, 1e-5, 5e-5]
    # weight_decays = [1e-4, 1e-3, 1e-5, 1e-2]
    # batch_sizes = [4, 8, 12, 16, 20, 24, 32]
    # seeds = [1, 2, 3, 4, 5]

    learning_rates = [1e-5]
    weight_decays = [1e-3]
    batch_sizes = [4]
    seeds = [1, 2, 3, 4, 5]

    # Generate all hyperparameter combinations
    param_grid = list(itertools.product(learning_rates, weight_decays, batch_sizes, seeds))
    for lr, wd, bs, seed in param_grid:
        main(lr, wd, bs, seed)
