"""
Evaluation metrics and tracking utilities for SPMA ML models.

This module provides comprehensive metrics for evaluating TCN and DQN performance,
including capacity constraint penalties and edge deployment metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import time


def compute_capacity_penalty(
    cos_predictions: torch.Tensor,
    max_capacity: float = 1.0,
    penalty_weight: float = 10.0
) -> torch.Tensor:
    """
    Compute capacity constraint penalty for COS predictions.
    
    Args:
        cos_predictions: Predicted COS values (batch_size, seq_len, cos_dim)
        max_capacity: Maximum allowed total capacity
        penalty_weight: Weight for penalty computation
    
    Returns:
        penalty: Capacity penalty tensor
    """
    # Sum COS across channels for each time step
    total_capacity = torch.sum(cos_predictions, dim=-1)  # (batch_size, seq_len)
    
    # Compute excess capacity (positive values when exceeding limit)
    excess_capacity = torch.clamp(total_capacity - max_capacity, min=0.0)
    
    # Average penalty across time steps and batch
    penalty = torch.mean(excess_capacity) * penalty_weight
    
    return penalty


def compute_cos_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Mean Squared Error for COS predictions.
    
    Args:
        predictions: Predicted COS values
        targets: Target COS values
        mask: Optional mask for valid predictions
    
    Returns:
        mse: Mean squared error
    """
    if mask is not None:
        predictions = predictions * mask
        targets = targets * mask
        mse = torch.sum((predictions - targets) ** 2) / torch.sum(mask)
    else:
        mse = torch.mean((predictions - targets) ** 2)
    
    return mse


def compute_cos_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Mean Absolute Error for COS predictions.
    
    Args:
        predictions: Predicted COS values
        targets: Target COS values
        mask: Optional mask for valid predictions
    
    Returns:
        mae: Mean absolute error
    """
    if mask is not None:
        predictions = predictions * mask
        targets = targets * mask
        mae = torch.sum(torch.abs(predictions - targets)) / torch.sum(mask)
    else:
        mae = torch.mean(torch.abs(predictions - targets))
    
    return mae


def compute_q_learning_loss(
    q_values: torch.Tensor,
    target_q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute Q-learning loss for DQN training.
    
    Args:
        q_values: Current Q-values from network
        target_q_values: Target Q-values
        actions: Selected actions
        rewards: Observed rewards
        dones: Episode termination flags
        gamma: Discount factor
    
    Returns:
        loss: Q-learning loss
    """
    # Gather Q-values for selected actions
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute target Q-values
    target_q = rewards + (gamma * target_q_values.max(1)[0] * ~dones)
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(current_q, target_q.detach())
    
    return loss


class MetricsTracker:
    """
    Comprehensive metrics tracker for SPMA ML training and evaluation.
    
    Features:
    - Training metrics (loss, accuracy, capacity penalties)
    - Edge deployment metrics (latency, memory usage)
    - Moving averages and statistics
    - Export capabilities for monitoring
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.epoch_metrics = defaultdict(list)
        self.best_metrics = {}
        self.training_start_time = None
        
    def start_training(self):
        """Start timing training process."""
        self.training_start_time = time.time()
    
    def update_metrics(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
    
    def update_epoch_metrics(self, epoch: int, metrics_dict: Dict[str, float]):
        """Update epoch-level metrics."""
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.epoch_metrics[key].append((epoch, value))
            
            # Track best metrics
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values (latest)."""
        current = {}
        for key, values in self.metrics.items():
            if values:
                current[key] = values[-1]
        return current
    
    def get_averaged_metrics(self) -> Dict[str, float]:
        """Get averaged metrics over the window."""
        averaged = {}
        for key, values in self.metrics.items():
            if values:
                averaged[key] = np.mean(list(values))
        return averaged
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        values = list(self.metrics[metric_name])
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    def log_metrics(self, logger, prefix: str = ""):
        """Log current metrics to logger."""
        current = self.get_current_metrics()
        averaged = self.get_averaged_metrics()
        
        for key, value in averaged.items():
            logger.info(f"{prefix}{key}: {value:.4f}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if self.training_start_time is None:
            training_time = 0
        else:
            training_time = time.time() - self.training_start_time
        
        summary = {
            'training_time': training_time,
            'total_epochs': len(self.epoch_metrics.get('epoch', [])),
            'best_metrics': self.best_metrics,
            'current_metrics': self.get_current_metrics(),
            'averaged_metrics': self.get_averaged_metrics()
        }
        
        return summary


class EdgeDeploymentMetrics:
    """Metrics specific to edge deployment performance."""
    
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.model_size = 0
        
    def measure_inference_time(self, model, input_tensor, num_runs: int = 100):
        """Measure inference time for a model."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append(end_time - start_time)
        
        self.inference_times.extend(times)
        return np.mean(times), np.std(times)
    
    def measure_model_size(self, model) -> int:
        """Measure model size in parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.model_size = total_params
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def check_performance_targets(
        self,
        tcn_target: float = 1.0,
        dqn_target: float = 0.5,
        memory_target: float = 10.0
    ) -> Dict[str, bool]:
        """Check if performance targets are met."""
        if not self.inference_times:
            return {'tcn_target': False, 'dqn_target': False}
        
        avg_time = np.mean(self.inference_times)
        
        return {
            'tcn_target': avg_time * 1000 < tcn_target,  # Convert to ms
            'dqn_target': avg_time * 1000 < dqn_target,  # Convert to ms
            'memory_target': self.model_size * 4 / (1024 * 1024) < memory_target
        }


def evaluate_tcn_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_capacity: float = 1.0,
    penalty_weight: float = 10.0
) -> Dict[str, float]:
    """
    Comprehensive evaluation of TCN model.
    
    Args:
        model: TCN model to evaluate
        data_loader: Validation data loader
        device: Device to run evaluation on
        max_capacity: Maximum capacity constraint
        penalty_weight: Capacity penalty weight
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_capacity_penalty = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            cos_pred, capacity_pred = model(features)
            
            # Compute metrics
            mse = compute_cos_mse(cos_pred, targets)
            mae = compute_cos_mae(cos_pred, targets)
            capacity_penalty = compute_capacity_penalty(
                cos_pred, max_capacity, penalty_weight
            )
            
            # Combined loss
            loss = mse + capacity_penalty
            
            total_loss += loss.item()
            total_capacity_penalty += capacity_penalty.item()
            total_mse += mse.item()
            total_mae += mae.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'mae': total_mae / num_batches,
        'capacity_penalty': total_capacity_penalty / num_batches
    }


def evaluate_dqn_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_episodes: int = 100
) -> Dict[str, float]:
    """
    Comprehensive evaluation of DQN model.
    
    Args:
        model: DQN model to evaluate
        data_loader: Validation data loader
        device: Device to run evaluation on
        num_episodes: Number of evaluation episodes
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    episode_rewards = []
    episode_lengths = []
    action_distribution = defaultdict(int)
    
    with torch.no_grad():
        for episode in range(num_episodes):
            episode_reward = 0.0
            episode_length = 0
            
            for batch in data_loader:
                # Get state from batch (simplified - would need proper environment)
                states = batch['states'].to(device)
                
                # Select action (greedy)
                q_values = model(states)
                actions = q_values.argmax(dim=1)
                
                # Track action distribution
                for action in actions:
                    action_distribution[action.item()] += 1
                
                # Get rewards (simplified)
                rewards = batch['rewards'].to(device)
                episode_reward += rewards.mean().item()
                episode_length += 1
                
                if episode_length >= 100:  # Max episode length
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'action_distribution': dict(action_distribution)
    }
