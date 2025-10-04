import os
import time
import argparse
from datetime import datetime
import cv2

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle
from monai.losses import GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

from Datasets.create_dataset import *
from Models.DeepLabV3Plus.modeling import *
from Utils.utils import DotDict, fix_all_seed


def main(config):
    """
    Main training function for GAPMatch.
    """
    dataset = get_dataset_without_full_label(
        config,
        img_size=config.data.img_size,
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment4
    )

    l_train_loader = DataLoader(
        dataset['lb_dataset'],
        batch_size=config.train.l_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )

    u_train_loader = DataLoader(
        dataset['ulb_dataset'],
        batch_size=config.train.u_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset['val_dataset'],  # same as val in this case
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )

    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    # Initialize single model for GAPMatch
    model = deeplabv3plus_resnet101(num_classes=3, output_stride=8, pretrained_backbone=True).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params / 1e6:.2f}M total parameters")
    print(f"{total_trainable_params / 1e6:.2f}M trainable parameters")

    criterion = [
        GeneralizedDiceFocalLoss(
            softmax=True,
            to_onehot_y=False,
            include_background=True
        ).cuda()
    ]

    # Pack dataloaders and train
    train_loader = {'l_loader': l_train_loader, 'u_loader': u_train_loader}
    train_val_gapmatch(config=config, model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion)

    # Optionally test
    test(config=config, model=model, model_dir=best_model_dir, test_loader=test_loader, criterion=criterion)
#i-fgsm
def generate_adversarial_ifgsm(model, img_s, pseudo_label, criterion, epsilon=4/255, step_size=1/255, num_iter=5):
    """
    Generate adversarial example using I-FGSM.
    
    Args:
        model: segmentation model
        img_s: strong augmented input image (B, C, H, W)
        pseudo_label: one-hot pseudo label (B, C, H, W)
        criterion: loss function
        epsilon: max perturbation
        step_size: per-step update
        num_iter: number of iteration
        
    Returns:
        Adversarially perturbed image (img_adv)
    """
    img_adv = img_s.clone().detach()
    img_adv.requires_grad = True

    for _ in range(num_iter):
        model.zero_grad()
        pred = model(img_adv)
        loss = criterion(pred, pseudo_label)
        loss.backward()

        with torch.no_grad():
            grad_sign = img_adv.grad.data.sign()
            img_adv = img_adv + step_size * grad_sign
            eta = torch.clamp(img_adv - img_s, min=-epsilon, max=epsilon)
            img_adv = torch.clamp(img_s + eta, 0, 1).detach()
            img_adv.requires_grad = True

    return img_adv
#GAP
def compute_gap_perturbation(model, img_s, pseudo_label, criterion, epsilon=1e-4, alpha=0.5):
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Step 1: Forward and backward to get g1
    pred = model(img_s)
    loss_unsup = criterion(pred, pseudo_label)
    loss_unsup.backward()

    g1 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
    backup = [p.clone() for p in model.parameters()]

    # Step 2: Normalize whole gradient vector and apply perturbation
    flat_g1 = torch.cat([g.view(-1) for g in g1])
    norm = flat_g1.norm() + 1e-8
    with torch.no_grad():
        for p, g in zip(model.parameters(), g1):
            if g is not None:
                p += epsilon * g / norm

    # Step 3: Forward and backward again to get g2
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    pred_star = model(img_s)
    loss_star = criterion(pred_star, pseudo_label)
    loss_star.backward()

    g2 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]

    # Step 4: Restore parameters
    with torch.no_grad():
        for p, b in zip(model.parameters(), backup):
            p.copy_(b)

    # Step 5: Combine g1 and g2
    gu = [g1_i + alpha * g2_i for g1_i, g2_i in zip(g1, g2)]

    return gu, loss_unsup.item(), loss_star.item()


def sigmoid_rampup(current, rampup_length):
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242
    """
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
def get_current_alpha(epoch, rampup_length=30, max_alpha=1.0):
    return max_alpha * sigmoid_rampup(epoch, rampup_length)

def get_current_consistency_weight(epoch):
    """Calculate the consistency weight for the current epoch."""
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def flatten_features(features):
    """Flatten feature maps for cosine similarity calculation."""
    return [f.view(f.size(0), -1) for f in features]

def calculate_cosine_similarity(features_1, features_2):
    """
    Calculate mean cosine similarity between two sets of feature maps.
    
    Args:
        features_1: First set of feature maps
        features_2: Second set of feature maps
        
    Returns:
        Mean cosine similarity across all feature map pairs
    """
    flattened_1 = flatten_features(features_1)
    flattened_2 = flatten_features(features_2)
    
    cosine_similarities = []
    for f1, f2 in zip(flattened_1, flattened_2):
        cos_sim = F.cosine_similarity(f1, f2, dim=1, eps=1e-6)
        cosine_similarities.append(cos_sim)
    
    return torch.stack(cosine_similarities).mean()
def train_val_gapmatch(config, model, train_loader, val_loader, criterion):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.num_epochs, eta_min=1e-6
    )

    best_dice = -1
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    torch.save(model.state_dict(), best_model_dir)

    for epoch in range(config.train.num_epochs):
        train_one_epoch_gapmatch(config, model, train_loader, optimizer, criterion, epoch, dice_metric)
        
        val_metrics = validate_model(model, val_loader, criterion)
        dice = val_metrics['dice']

        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), best_model_dir)
            print(f"✔ Saved best model at epoch {epoch}, Dice = {dice:.4f}")

        scheduler.step()

def train_one_epoch_gapmatch(config, model, train_loader, optimizer, criterion, epoch, dice_metric):
    model.train()
    alpha = get_current_alpha(epoch, rampup_length=30, max_alpha=1.0)

    epsilon = 1e-4
    loop = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
    pbar = tqdm(loop, total=len(train_loader['u_loader']), desc=f"Epoch {epoch}")

    for batch_l, batch_u in pbar:
        img_l = batch_l['image'].cuda().float()
        lbl_l = batch_l['label'].cuda().long()
        img_w = batch_u['img_w'].cuda().float()
        img_s = batch_u['img_s'].cuda().float()

        with torch.no_grad():
            pred_w = torch.softmax(model(img_w), dim=1)
            conf, pseudo_lbl = torch.max(pred_w, dim=1)
            mask = (conf >= config.semi.conf_thresh).float()

        pseudo_lbl = pseudo_lbl.detach()
        pseudo_lbl_onehot = F.one_hot(pseudo_lbl, num_classes=pred_w.shape[1]).permute(0, 3, 1, 2).float()

        # Step 1: Generate adversarial image using I-FGSM on img_s
        img_s_adv = generate_adversarial_ifgsm(
            model=model,
            img_s=img_s,
            pseudo_label=pseudo_lbl_onehot,
            criterion=criterion[0],
            epsilon=4/255,        # tùy chỉnh
            step_size=1/255,      # tùy chỉnh
            num_iter=3      # bạn chọn số K bước
        )
            # epsilon=4/255,        # tùy chỉnh
            # step_size=1/255,      # tùy chỉnh
            # num_iter=3   
        # Step 2: Use adversarial image in GAP update
        gu, loss_unsup, loss_star = compute_gap_perturbation(
            model, img_s_adv, pseudo_lbl_onehot, criterion[0], epsilon, alpha
        )

        pred_l = model(img_l)
        loss_sup = criterion[0](pred_l, lbl_l)

        optimizer.zero_grad()
        loss_sup.backward(retain_graph=True)
        gl = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]

        with torch.no_grad():
            for p, g_l, g_u in zip(model.parameters(), gl, gu):
                if p.grad is not None:
                    p.grad.copy_(g_l + g_u)
        optimizer.step()

        output_onehot = torch.zeros_like(pred_l)
        output_onehot.scatter_(1, pred_l.argmax(dim=1, keepdim=True), 1)
        dice_metric(y_pred=output_onehot, y=lbl_l)

        pbar.set_postfix({
            'Lsup': f"{loss_sup.item():.4f}",
            'Lunsup': f"{loss_unsup:.4f}",
            'Lunsup*': f"{loss_star:.4f}",
            'Dice': f"{dice_metric.aggregate().item():.4f}"
        })

    dice_metric.reset()


def validate_model(model, val_loader, criterion):
    """
    Validate a single model using MONAI metrics.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function(s)
    
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0
    
    # Initialize MONAI metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    
    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]
        
        with torch.no_grad():
            output = torch.softmax(model(img), dim=1)
            loss = criterion[0](output, label)
            
            # Convert predictions to one-hot format
            preds = torch.argmax(output, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(output)
            preds_onehot.scatter_(1, preds, 1)
            
            # Convert labels to one-hot format if needed
            if len(label.shape) == 4:  # If already one-hot
                labels_onehot = label
            else:  # If not one-hot
                labels_onehot = torch.zeros_like(output)
                labels_onehot.scatter_(1, label.unsqueeze(1), 1)
            
            # Compute metrics
            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            iou_metric(y_pred=preds_onehot, y=labels_onehot)
            hd_metric(y_pred=preds_onehot, y=labels_onehot)
            
            # Update loss
            metrics['loss'] = (metrics['loss'] * num_val + loss.item() * batch_len) / (num_val + batch_len)
            num_val += batch_len
            
            val_loop.set_postfix({
                'Loss': f"{loss.item():.4f}"
            })
    
    # Aggregate metrics
    metrics['dice'] = dice_metric.aggregate().item()
    metrics['iou'] = iou_metric.aggregate().item()
    metrics['hd'] = hd_metric.aggregate().item()
    
    # Reset metrics for next validation
    dice_metric.reset()
    iou_metric.reset()
    hd_metric.reset()
    
    return metrics

def test(config, model, model_dir, test_loader, criterion):
    """
    Test the model on the test set.
    
    Args:
        config: Test configuration
        model: Model to test
        model_dir: Path to saved model weights
        test_loader: Test data loader
        criterion: Loss function(s)
    """
    model.load_state_dict(torch.load(model_dir))
    metrics = validate_model(model, test_loader, criterion)
    
    # Save and print results
    results_str = (f"Test Results:\n"
                  f"Loss: {metrics['loss']:.4f}\n"
                  f"Dice: {metrics['dice']:.4f}\n"
                  f"IoU: {metrics['iou']:.4f}\n"
                  f"HD: {metrics['hd']:.4f}")
    
    with open(test_results_dir, 'w') as f:
        f.write(results_str)
    
    print('='*80)
    print(results_str)
    print('='*80)
    
    file_log.write('\n' + '='*80 + '\n')
    file_log.write(results_str + '\n')
    file_log.write('='*80 + '\n')
    file_log.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--exp', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    
    args = parser.parse_args()
    
    # Load and update config
    # config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    with open(args.config_yml, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['data']['name'] = args.dataset
    config['model_adapt']['adapt_method'] = args.adapt_method
    config['model_adapt']['num_domains'] = args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    # Setup CUDA and seeds
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])
    
    # Print configuration
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}")
    
    store_config = config
    config = DotDict(config)
    
    # Train each fold
    for fold in [1,2,3,4,5]:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        # Setup directories
        exp_dir = f"{config.data.save_folder}/{args.exp}/fold{fold}"
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = f'{exp_dir}/best.pth'
        test_results_dir = f'{exp_dir}/test_results.txt'
        
        # Save config
        if not config.debug:
            yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))
        
        # Train fold
        with open(f'{exp_dir}/log.txt', 'w') as file_log:
            main(config)
        
        torch.cuda.empty_cache()
