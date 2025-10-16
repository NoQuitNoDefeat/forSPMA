"""
Logging utilities for SPMA ML training and inference.

This module provides structured logging for training metrics, model performance,
and edge deployment statistics with support for file and console output.
"""

import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import numpy as np


def setup_logger(
    name: str = "spma_ml",
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Setup structured logger for SPMA ML project.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    
    Returns:
        logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_training_metrics(
    logger: logging.Logger,
    epoch: int,
    metrics: Dict[str, Any],
    prefix: str = ""
):
    """
    Log training metrics in a structured format.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        metrics: Dictionary of metrics to log
        prefix: Optional prefix for metric names
    """
    # Format metrics for logging
    metric_strs = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            metric_strs.append(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, torch.Tensor):
            metric_strs.append(f"{prefix}{key}: {value.item():.4f}")
        else:
            metric_strs.append(f"{prefix}{key}: {value}")
    
    # Log metrics
    logger.info(f"Epoch {epoch} - " + " | ".join(metric_strs))


def log_model_info(
    logger: logging.Logger,
    model: torch.nn.Module,
    model_name: str = "Model"
):
    """
    Log model architecture and parameter information.
    
    Args:
        logger: Logger instance
        model: PyTorch model
        model_name: Name of the model for logging
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"{model_name} Architecture:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    # Log model structure
    logger.info(f"{model_name} Structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                logger.info(f"  {name}: {type(module).__name__} ({param_count:,} params)")


def log_inference_performance(
    logger: logging.Logger,
    inference_times: list,
    model_name: str = "Model",
    device: str = "CPU"
):
    """
    Log inference performance metrics.
    
    Args:
        logger: Logger instance
        inference_times: List of inference times in seconds
        model_name: Name of the model
        device: Device used for inference
    """
    if not inference_times:
        logger.warning("No inference times provided")
        return
    
    times_ms = [t * 1000 for t in inference_times]  # Convert to milliseconds
    
    logger.info(f"{model_name} Inference Performance on {device}:")
    logger.info(f"  Mean: {np.mean(times_ms):.2f} ms")
    logger.info(f"  Std: {np.std(times_ms):.2f} ms")
    logger.info(f"  Min: {np.min(times_ms):.2f} ms")
    logger.info(f"  Max: {np.max(times_ms):.2f} ms")
    logger.info(f"  P50: {np.percentile(times_ms, 50):.2f} ms")
    logger.info(f"  P95: {np.percentile(times_ms, 95):.2f} ms")
    logger.info(f"  P99: {np.percentile(times_ms, 99):.2f} ms")


def log_configuration(
    logger: logging.Logger,
    config: Dict[str, Any],
    config_name: str = "Configuration"
):
    """
    Log configuration parameters.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        config_name: Name of the configuration
    """
    logger.info(f"{config_name}:")
    
    def log_dict(d: Dict[str, Any], indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)


def save_metrics_to_json(
    metrics: Dict[str, Any],
    filepath: str,
    include_timestamp: bool = True
):
    """
    Save metrics to JSON file for later analysis.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save JSON file
        include_timestamp: Whether to include timestamp in metrics
    """
    # Convert numpy arrays and tensors to lists
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    # Add timestamp if requested
    if include_timestamp:
        metrics['timestamp'] = datetime.now().isoformat()
    
    # Convert metrics
    json_metrics = convert_for_json(metrics)
    
    # Save to file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(json_metrics, f, indent=2)


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    batch_idx: int,
    total_batches: int,
    loss: float,
    additional_metrics: Optional[Dict[str, float]] = None
):
    """
    Log training progress with progress bar-like output.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        total_epochs: Total number of epochs
        batch_idx: Current batch index
        total_batches: Total number of batches per epoch
        loss: Current loss value
        additional_metrics: Additional metrics to log
    """
    # Calculate progress percentage
    epoch_progress = (epoch - 1) / total_epochs * 100
    batch_progress = batch_idx / total_batches * 100
    
    # Create progress bar
    bar_length = 20
    filled_length = int(bar_length * batch_idx // total_batches)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # Format additional metrics
    metrics_str = ""
    if additional_metrics:
        metric_parts = []
        for key, value in additional_metrics.items():
            if isinstance(value, (int, float)):
                metric_parts.append(f"{key}: {value:.4f}")
        if metric_parts:
            metrics_str = " | " + " | ".join(metric_parts)
    
    # Log progress
    logger.info(
        f"Epoch [{epoch}/{total_epochs}] "
        f"[{bar}] {batch_progress:.1f}% "
        f"Loss: {loss:.4f}{metrics_str}"
    )


class TrainingLogger:
    """
    Advanced training logger with metric tracking and visualization support.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        model_name: str = "model",
        level: str = "INFO"
    ):
        self.log_dir = log_dir
        self.model_name = model_name
        self.logger = setup_logger(
            name=f"{model_name}_training",
            log_dir=log_dir,
            level=level
        )
        
        self.metrics_history = []
        self.start_time = None
        
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        self.logger.info(f"Starting {self.model_name} training")
        
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log the start of an epoch."""
        self.logger.info(f"Starting epoch {epoch}/{total_epochs}")
        
    def log_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log the end of an epoch with metrics."""
        # Store metrics
        epoch_data = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if val_metrics:
            epoch_data['val_metrics'] = val_metrics
        
        self.metrics_history.append(epoch_data)
        
        # Log metrics
        log_training_metrics(self.logger, epoch, train_metrics, "Train ")
        if val_metrics:
            log_training_metrics(self.logger, epoch, val_metrics, "Val ")
        
        # Save metrics to file
        metrics_file = os.path.join(
            self.log_dir, f"{self.model_name}_metrics.json"
        )
        save_metrics_to_json(self.metrics_history, metrics_file)
        
    def log_training_end(self):
        """Log the end of training."""
        if self.start_time:
            training_time = time.time() - self.start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best metrics across all epochs."""
        if not self.metrics_history:
            return {}
        
        best_metrics = {}
        
        for epoch_data in self.metrics_history:
            val_metrics = epoch_data.get('val_metrics', {})
            for key, value in val_metrics.items():
                if key not in best_metrics or value > best_metrics[key]:
                    best_metrics[key] = value
        
        return best_metrics
