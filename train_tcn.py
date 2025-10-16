#!/usr/bin/env python3
"""
TCN Training Script for SPMA ML

This script trains the Temporal Convolutional Network (TCN) to predict
non-stationary Channel Occupancy Status (COS) with capacity constraints.

Usage:
    python train_tcn.py --config config.yaml --data-dir data/
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
from typing import Dict, Any, Tuple
import time
from tqdm import tqdm

from models.tcn import TCN, create_tcn_from_config
from utils.dataset import create_data_loaders
from utils.metrics import evaluate_tcn_model, MetricsTracker
from utils.logger import setup_logger, TrainingLogger, log_model_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TCN for SPMA COS prediction')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (auto/cpu/cuda)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation, skip training')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_str: str) -> torch.device:
    """Setup training device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_str)
        print(f"Using device: {device}")
    
    return device


def create_model(config: Dict[str, Any]) -> TCN:
    """Create TCN model from configuration."""
    tcn_config = config['tcn']
    model = create_tcn_from_config(tcn_config)
    return model


def create_optimizer(model: TCN, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer for model training."""
    training_config = config['training']
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 1e-5)
    )
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    training_config = config['training']
    
    # Use ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=training_config.get('patience', 10),
        verbose=True
    )
    
    return scheduler


def compute_loss(
    cos_pred: torch.Tensor,
    capacity_pred: torch.Tensor,
    targets: torch.Tensor,
    capacity_targets: torch.Tensor,
    config: Dict[str, Any]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute training loss with capacity constraints."""
    # COS prediction loss (MSE)
    cos_loss = nn.MSELoss()(cos_pred, targets)
    
    # Capacity prediction loss (MSE)
    capacity_loss = nn.MSELoss()(capacity_pred, capacity_targets)
    
    # Capacity constraint penalty
    data_config = config.get('data', {})
    max_capacity = data_config.get('max_capacity', 1.0)
    penalty_weight = data_config.get('capacity_penalty_weight', 10.0)
    
    # Compute capacity penalty
    total_capacity = torch.sum(cos_pred, dim=-1)
    excess_capacity = torch.clamp(total_capacity - max_capacity, min=0.0)
    capacity_penalty = torch.mean(excess_capacity) * penalty_weight
    
    # Combined loss
    total_loss = cos_loss + 0.1 * capacity_loss + capacity_penalty
    
    loss_dict = {
        'total_loss': total_loss,
        'cos_loss': cos_loss,
        'capacity_loss': capacity_loss,
        'capacity_penalty': capacity_penalty
    }
    
    return total_loss, loss_dict


def train_epoch(
    model: TCN,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    logger: TrainingLogger,
    epoch: int
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_cos_loss = 0.0
    total_capacity_loss = 0.0
    total_capacity_penalty = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        features = batch['features'].to(device)
        
        # Prepare targets
        cos_targets = features[:, :, :config['tcn']['input_dim'] - config['tcn']['n_prio']]
        capacity_targets = batch['capacity_used'].unsqueeze(-1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        cos_pred, capacity_pred = model(features)
        
        # Compute loss
        loss, loss_dict = compute_loss(
            cos_pred, capacity_pred, cos_targets, capacity_targets, config
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss_dict['total_loss'].item()
        total_cos_loss += loss_dict['cos_loss'].item()
        total_capacity_loss += loss_dict['capacity_loss'].item()
        total_capacity_penalty += loss_dict['capacity_penalty'].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'COS': f"{loss_dict['cos_loss'].item():.4f}",
            'Penalty': f"{loss_dict['capacity_penalty'].item():.4f}"
        })
        
        # Log training progress
        if batch_idx % 100 == 0:
            log_training_progress(
                logger.logger, epoch, config['training']['num_epochs'],
                batch_idx, len(train_loader), loss.item(), {
                    'COS_Loss': loss_dict['cos_loss'].item(),
                    'Capacity_Penalty': loss_dict['capacity_penalty'].item()
                }
            )
    
    # Average metrics
    avg_metrics = {
        'loss': total_loss / num_batches,
        'cos_loss': total_cos_loss / num_batches,
        'capacity_loss': total_capacity_loss / num_batches,
        'capacity_penalty': total_capacity_penalty / num_batches
    }
    
    return avg_metrics


def validate_epoch(
    model: TCN,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Validate model for one epoch."""
    model.eval()
    
    val_metrics = evaluate_tcn_model(
        model, val_loader, device,
        max_capacity=config.get('data', {}).get('max_capacity', 1.0),
        penalty_weight=config.get('data', {}).get('capacity_penalty_weight', 10.0)
    )
    
    return val_metrics


def save_checkpoint(
    model: TCN,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: TCN,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    filepath: str
) -> Tuple[int, Dict[str, float]]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(args.device)
    
    # Setup logging
    logger = TrainingLogger(
        log_dir=config['logging']['log_dir'],
        model_name='tcn',
        level=config['logging']['level']
    )
    
    logger.logger.info("Starting TCN training")
    logger.logger.info(f"Configuration: {args.config}")
    logger.logger.info(f"Device: {device}")
    
    # Log configuration
    log_configuration(logger.logger, config, "Training Configuration")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Log model info
    log_model_info(logger.logger, model, "TCN")
    
    # Create data loaders
    train_path = os.path.join(args.data_dir, 'train.csv')
    val_path = os.path.join(args.data_dir, 'val.csv')
    
    train_loader, val_loader = create_data_loaders(
        train_path, val_path, config,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    logger.logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        start_epoch, metrics = load_checkpoint(model, optimizer, scheduler, args.resume)
        best_val_loss = metrics.get('loss', float('inf'))
        logger.logger.info(f"Resumed from epoch {start_epoch}")
    
    # Validate only mode
    if args.validate_only:
        logger.logger.info("Running validation only")
        val_metrics = validate_epoch(model, val_loader, device, config)
        logger.log_epoch_end(0, {}, val_metrics)
        return
    
    # Training loop
    logger.start_training()
    
    for epoch in range(start_epoch + 1, config['training']['num_epochs'] + 1):
        logger.log_epoch_start(epoch, config['training']['num_epochs'])
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, logger, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, config)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log epoch results
        logger.log_epoch_end(epoch, train_metrics, val_metrics)
        
        # Save checkpoint
        if config['logging']['save_checkpoints']:
            if epoch % config['logging']['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(args.output_dir, f'tcn_checkpoint_epoch_{epoch}.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_path)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(args.output_dir, 'tcn_best_model.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_model_path)
            logger.logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Early stopping
        if epoch - start_epoch > config['training']['patience']:
            logger.logger.info("Early stopping triggered")
            break
    
    # Training completed
    logger.log_training_end()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'tcn_final_model.pt')
    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, final_model_path)
    
    # Log best metrics
    best_metrics = logger.get_best_metrics()
    logger.logger.info(f"Best validation metrics: {best_metrics}")
    
    logger.logger.info("TCN training completed successfully")


if __name__ == '__main__':
    main()
