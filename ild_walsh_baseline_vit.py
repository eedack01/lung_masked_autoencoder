import os
import argparse
import json
import logging
import sys
from pathlib import Path
import itertools

import torch
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from time import time
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import copy
from dataset import WalshBaselineDataset
from utils.custom_scheduler import CosineAnnealingWarmupRestarts
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from torchvision.transforms import Compose, ToTensor, Resize
import clip
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

class Classifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.drop_out = nn.Dropout()
        self.linear = nn.Linear(512, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear(x)
        return x

def convert_models_to_fp32(model): 
        for p in model.parameters():
            if p is not None:
                p.data = p.data.float() 
        return model

def main(lr, bs):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-c",
        "--config-file",
        default="config/ild/vit_baseline.json",
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
    logging.info(f"Hyperparameters set | lr: {lr}, batch size: {bs}")

    set_determinism(42)

    # Encoder setup
    model, _ = clip.load('ViT-B/32', device=device, jit=False)
    checkpoint = torch.load('pretrained_weights/best_64_5e-05_original_22000_0.864.pt')
    model.load_state_dict(checkpoint, strict=True)
    model = model.visual
    backbone = convert_models_to_fp32(model)
    backbone = backbone.to(device)
    classifier = Classifier(num_class=args.classes).to(device)
    model = nn.Sequential(backbone, classifier)

    # Step 1: set data loader
    train_transforms = Compose(
        [
            Resize((224, 224)),
            ToTensor(),  # Convert to tensor
        ]
    )
    
    train_dataset = WalshBaselineDataset(
                 data_path=args.train['root_dir'], 
                 split_size = 0.7, 
                 transform=train_transforms, 
                 split='train',
                 split_csv=args.train['csv_file'],
                 binary=args.binary,
                 random_state=args.random_seed
    )
    val_dataset = WalshBaselineDataset(
                 data_path=args.train['root_dir'], 
                 split_size = 0.7, 
                 transform=train_transforms, 
                 split='val',
                 split_csv=args.train['csv_file'],
                 binary=args.binary,
                 random_state=args.random_seed
    )
    
    
    class_weights = torch.tensor([3.8889, 5.0000, 5.8333, 2.6923]).to(device)
    dataloaders = {
        'train': DataLoader(dataset=train_dataset,
                              batch_size=bs, 
                              shuffle=args.train['shuffle'], 
                              num_workers=args.train['num_workers']
                              ),
        'val': DataLoader(dataset=val_dataset,
                              batch_size=bs, 
                              shuffle=args.val['shuffle'], 
                              num_workers=args.val['num_workers']
                              )
    }

    if args.binary:
        print('using cross entropy with zero class weights')
        criterion = torch.nn.CrossEntropyLoss()
    else:
        print('using cross entropy with class weights')
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    # optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
    
    best_f1 = 0
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
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    preds = model(images)
                    
                    loss = criterion(preds, targets)
                    
                    # print(loss)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
 
                running_loss += loss.item() * images.size(0)
                if phase == 'val':
                    all_preds.extend(preds.argmax(dim=-1).cpu().numpy())  
                    all_targets.extend(targets.cpu().numpy())
                
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
                

        last_model_file = Path(args.model_dir) / f"last_model_epoch_{str(lr)}_{str(bs)}.pth"
        torch.save(model.state_dict(), last_model_file)
        logging.info(f"Saved last model for epoch {epoch + 1} at {last_model_file}")

        # Add metrics to DataFrame
        print(epoch_data)
        df = pd.concat([df, pd.DataFrame([epoch_data])], ignore_index=True)

    # Save the best model after training
    model.load_state_dict(best_model_wts)
    best_model_file = Path(args.model_dir) / f"best_model_{datetime.now().strftime('%Y%m%d')}_{str(lr)}_{str(bs)}.pth"
    torch.save(best_model_wts, best_model_file)
    logging.info(f"Saved best model at {best_model_file}")

    # Save the training log
    training_log_file = Path(args.model_dir) / f"training_log_{datetime.now().strftime('%Y%m%d')}_{str(lr)}_{str(bs)}.csv"
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

    learning_rates = [1e-4]
    batch_sizes = [32]

    # Generate all hyperparameter combinations
    param_grid = list(itertools.product(learning_rates, batch_sizes))
    for lr, bs in param_grid:
        main(lr, bs)
