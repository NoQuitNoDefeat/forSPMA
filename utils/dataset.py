"""
Dataset utilities for SPMA ML training.

This module provides sliding window sequence datasets with capacity constraint
penalties and data preprocessing for both TCN and DQN training.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
import warnings


class SPMADataset(Dataset):
    """
    SPMA dataset with sliding window sequences and capacity constraints.
    
    Features:
    - Sliding window sequence generation for temporal modeling
    - Capacity constraint penalty computation
    - Normalization and preprocessing
    - Support for both TCN and DQN training formats
    
    Args:
        data_path: Path to CSV data file
        window_size: Size of sliding window
        sequence_stride: Stride for sliding window
        normalize: Whether to normalize features
        cos_dim: Number of COS (Channel Occupancy Status) features
        n_prio: Number of priority features
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 32,
        sequence_stride: int = 1,
        normalize: bool = True,
        cos_dim: int = 14,
        n_prio: int = 2,
        max_capacity: float = 1.0,
        capacity_penalty_weight: float = 10.0
    ):
        self.window_size = window_size
        self.sequence_stride = sequence_stride
        self.normalize = normalize
        self.cos_dim = cos_dim
        self.n_prio = n_prio
        self.max_capacity = max_capacity
        self.capacity_penalty_weight = capacity_penalty_weight
        
        # Load and preprocess data
        self.data = self._load_data(data_path)
        self.sequences = self._create_sequences()
        
        # Initialize scaler for normalization
        self.scaler = StandardScaler() if normalize else None
        if self.normalize:
            self._fit_scaler()
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV file and perform basic preprocessing."""
        try:
            data = pd.read_csv(data_path)
            
            # Validate required columns
            required_cols = [f'cos_{i}' for i in range(self.cos_dim)]
            required_cols.extend(['backoff_action', 'reward', 'capacity_used'])
            
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Sort by timestamp if available
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Ensure COS values are in [0, 1] range
            cos_cols = [f'cos_{i}' for i in range(self.cos_dim)]
            data[cos_cols] = data[cos_cols].clip(0, 1)
            
            # Ensure capacity is in [0, 1] range
            data['capacity_used'] = data['capacity_used'].clip(0, 1)
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {data_path}: {str(e)}")
    
    def _create_sequences(self) -> List[Dict[str, Any]]:
        """Create sliding window sequences from the data."""
        sequences = []
        n_samples = len(self.data)
        
        for start_idx in range(0, n_samples - self.window_size + 1, self.sequence_stride):
            end_idx = start_idx + self.window_size
            
            # Extract sequence data
            sequence_data = self.data.iloc[start_idx:end_idx]
            
            # Prepare features: COS + priority info
            cos_features = sequence_data[[f'cos_{i}' for i in range(self.cos_dim)]].values
            
            # Generate priority features (placeholder - could be derived from other data)
            priority_features = np.random.rand(self.window_size, self.n_prio)  # TODO: Use actual priority data
            
            # Combine features
            features = np.concatenate([cos_features, priority_features], axis=1)
            
            # Prepare targets
            actions = sequence_data['backoff_action'].values
            rewards = sequence_data['reward'].values
            capacity_used = sequence_data['capacity_used'].values
            
            # Compute capacity constraint penalty
            capacity_penalty = self._compute_capacity_penalty(cos_features)
            
            sequence = {
                'features': features,
                'actions': actions,
                'rewards': rewards,
                'capacity_used': capacity_used,
                'capacity_penalty': capacity_penalty,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            
            sequences.append(sequence)
        
        return sequences
    
    def _compute_capacity_penalty(self, cos_features: np.ndarray) -> float:
        """
        Compute capacity constraint penalty.
        
        Penalty is applied when sum(COS) > max_capacity, encouraging
        the system to stay within capacity limits.
        """
        # Sum COS across all channels for each time step
        total_capacity = np.sum(cos_features, axis=1)
        
        # Compute penalty for exceeding capacity
        excess_capacity = np.maximum(0, total_capacity - self.max_capacity)
        penalty = np.mean(excess_capacity) * self.capacity_penalty_weight
        
        return penalty
    
    def _fit_scaler(self):
        """Fit scaler on all feature data."""
        if not self.normalize or self.scaler is None:
            return
        
        # Collect all feature data for fitting
        all_features = []
        for seq in self.sequences:
            all_features.append(seq['features'])
        
        all_features = np.vstack(all_features)
        self.scaler.fit(all_features)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence for training."""
        sequence = self.sequences[idx]
        
        # Extract data
        features = sequence['features'].astype(np.float32)
        actions = sequence['actions'].astype(np.int64)
        rewards = sequence['rewards'].astype(np.float32)
        capacity_used = sequence['capacity_used'].astype(np.float32)
        capacity_penalty = sequence['capacity_penalty']
        
        # Normalize features if enabled
        if self.normalize and self.scaler is not None:
            original_shape = features.shape
            features = features.reshape(-1, features.shape[-1])
            features = self.scaler.transform(features)
            features = features.reshape(original_shape)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        actions_tensor = torch.from_numpy(actions)
        rewards_tensor = torch.from_numpy(rewards)
        capacity_tensor = torch.from_numpy(capacity_used)
        
        return {
            'features': features_tensor,
            'actions': actions_tensor,
            'rewards': rewards_tensor,
            'capacity_used': capacity_tensor,
            'capacity_penalty': torch.tensor(capacity_penalty, dtype=torch.float32)
        }
    
    def get_tcn_batch(self, batch_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch formatted for TCN training.
        
        Returns:
            inputs: (batch_size, seq_len, input_dim)
            targets: (batch_size, seq_len, cos_dim)
        """
        batch_data = [self[idx] for idx in batch_indices]
        
        inputs = torch.stack([item['features'] for item in batch_data])
        
        # For TCN, target is the COS features (excluding priority)
        targets = inputs[:, :, :self.cos_dim]
        
        return inputs, targets
    
    def get_dqn_batch(self, batch_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get batch formatted for DQN training.
        
        Returns:
            states: (batch_size, state_dim)
            actions: (batch_size,)
            rewards: (batch_size,)
        """
        batch_data = [self[idx] for idx in batch_indices]
        
        # For DQN, state is the last timestep features + additional context
        states = []
        actions = []
        rewards = []
        
        for item in batch_data:
            # Use last timestep as primary state
            last_features = item['features'][-1].numpy()
            last_action = item['actions'][-1].item()
            last_reward = item['rewards'][-1].item()
            
            # Add context: capacity usage, recent reward history, etc.
            capacity_context = item['capacity_used'][-1].item()
            reward_history = torch.mean(item['rewards'][-5:]).item()  # Last 5 rewards
            
            # Combine into state vector (pad/truncate to state_dim)
            state = np.concatenate([
                last_features,
                [capacity_context, reward_history]
            ])
            
            # Ensure state_dim
            if len(state) > 10:  # Assuming state_dim=10
                state = state[:10]
            elif len(state) < 10:
                state = np.pad(state, (0, 10 - len(state)), 'constant')
            
            states.append(state)
            actions.append(last_action)
            rewards.append(last_reward)
        
        return np.array(states), np.array(actions), np.array(rewards)
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get dataset statistics for analysis."""
        all_cos = []
        all_rewards = []
        all_capacity = []
        
        for seq in self.sequences:
            cos_features = seq['features'][:, :self.cos_dim]
            all_cos.append(cos_features)
            all_rewards.extend(seq['rewards'])
            all_capacity.extend(seq['capacity_used'])
        
        all_cos = np.vstack(all_cos)
        
        stats = {
            'num_sequences': len(self.sequences),
            'sequence_length': self.window_size,
            'total_samples': len(self.data),
            'cos_mean': np.mean(all_cos, axis=0),
            'cos_std': np.std(all_cos, axis=0),
            'reward_mean': np.mean(all_rewards),
            'reward_std': np.std(all_rewards),
            'capacity_mean': np.mean(all_capacity),
            'capacity_std': np.std(all_capacity)
        }
        
        return stats


def create_data_loaders(
    train_path: str,
    val_path: str,
    config: Dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_path: Path to training data CSV
        val_path: Path to validation data CSV
        config: Configuration dictionary
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Extract relevant config
    tcn_config = config.get('tcn', {})
    data_config = config.get('data', {})
    
    # Create datasets
    train_dataset = SPMADataset(
        train_path,
        window_size=tcn_config.get('window_size', 32),
        sequence_stride=data_config.get('sequence_stride', 1),
        normalize=data_config.get('normalize_inputs', True),
        cos_dim=tcn_config.get('input_dim', 14) - tcn_config.get('n_prio', 2),
        n_prio=tcn_config.get('n_prio', 2),
        max_capacity=data_config.get('max_capacity', 1.0),
        capacity_penalty_weight=data_config.get('capacity_penalty_weight', 10.0)
    )
    
    val_dataset = SPMADataset(
        val_path,
        window_size=tcn_config.get('window_size', 32),
        sequence_stride=data_config.get('sequence_stride', 1),
        normalize=data_config.get('normalize_inputs', True),
        cos_dim=tcn_config.get('input_dim', 14) - tcn_config.get('n_prio', 2),
        n_prio=tcn_config.get('n_prio', 2),
        max_capacity=data_config.get('max_capacity', 1.0),
        capacity_penalty_weight=data_config.get('capacity_penalty_weight', 10.0)
    )
    
    # Use train dataset's scaler for validation (important!)
    if train_dataset.normalize and val_dataset.normalize:
        val_dataset.scaler = train_dataset.scaler
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader
