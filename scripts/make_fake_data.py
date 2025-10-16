#!/usr/bin/env python3
"""
Generate fake data for SPMA ML training and testing.

This script generates realistic synthetic data that mimics the characteristics
of real SPMA protocol data including channel occupancy patterns, capacity
constraints, and reward signals.

Usage:
    python scripts/make_fake_data.py --output-dir data/ --num-samples 10000
"""

import argparse
import numpy as np
import pandas as pd
import os
from typing import Dict, Any, Tuple
import yaml


def generate_cos_sequence(
    num_samples: int,
    cos_dim: int,
    correlation_strength: float = 0.3,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate realistic Channel Occupancy Status (COS) sequence.
    
    Args:
        num_samples: Number of samples to generate
        cos_dim: Number of channels
        correlation_strength: Strength of correlation between channels
        noise_level: Level of random noise
    
    Returns:
        cos_sequence: Generated COS sequence (num_samples, cos_dim)
    """
    # Base patterns for each channel
    cos_sequence = np.zeros((num_samples, cos_dim))
    
    for i in range(cos_dim):
        # Different channels have different occupancy patterns
        if i < cos_dim // 3:
            # High occupancy channels
            base_level = np.random.beta(3, 2)  # Skewed towards higher values
        elif i < 2 * cos_dim // 3:
            # Medium occupancy channels
            base_level = np.random.beta(2, 2)  # Symmetric
        else:
            # Low occupancy channels
            base_level = np.random.beta(2, 3)  # Skewed towards lower values
        
        # Add temporal patterns
        time_factor = np.arange(num_samples) / 100.0
        periodic_component = 0.2 * np.sin(2 * np.pi * time_factor / (50 + i * 10))
        trend_component = 0.1 * np.sin(2 * np.pi * time_factor / (200 + i * 20))
        
        # Generate base sequence
        cos_sequence[:, i] = base_level + periodic_component + trend_component
        
        # Add noise
        noise = np.random.normal(0, noise_level, num_samples)
        cos_sequence[:, i] += noise
    
    # Add correlation between channels
    if correlation_strength > 0:
        correlation_matrix = np.eye(cos_dim)
        for i in range(cos_dim):
            for j in range(i + 1, cos_dim):
                correlation_matrix[i, j] = correlation_strength * np.random.uniform(0.5, 1.0)
                correlation_matrix[j, i] = correlation_matrix[i, j]
        
        # Apply correlation
        cos_sequence = np.dot(cos_sequence, correlation_matrix)
    
    # Ensure values are in [0, 1] range
    cos_sequence = np.clip(cos_sequence, 0, 1)
    
    return cos_sequence


def generate_backoff_actions(
    num_samples: int,
    cos_sequence: np.ndarray,
    capacity_threshold: float = 0.8
) -> np.ndarray:
    """
    Generate realistic backoff actions based on COS patterns.
    
    Args:
        num_samples: Number of samples
        cos_sequence: COS sequence
        capacity_threshold: Capacity threshold for action selection
    
    Returns:
        actions: Generated backoff actions
    """
    actions = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # Compute total capacity usage
        total_capacity = np.sum(cos_sequence[i])
        
        # Select action based on capacity usage and COS variance
        cos_variance = np.var(cos_sequence[i])
        
        if total_capacity > capacity_threshold:
            # High capacity - more conservative actions
            if cos_variance > 0.1:
                actions[i] = np.random.choice([0, 1], p=[0.7, 0.3])  # -2, -1
            else:
                actions[i] = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])  # -2, -1, 0
        elif total_capacity < 0.3:
            # Low capacity - more aggressive actions
            if cos_variance < 0.05:
                actions[i] = np.random.choice([3, 4], p=[0.3, 0.7])  # +1, +2
            else:
                actions[i] = np.random.choice([2, 3, 4], p=[0.2, 0.3, 0.5])  # 0, +1, +2
        else:
            # Medium capacity - balanced actions
            actions[i] = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])  # -1, 0, +1
    
    return actions


def generate_rewards(
    cos_sequence: np.ndarray,
    actions: np.ndarray,
    capacity_threshold: float = 0.8
) -> np.ndarray:
    """
    Generate realistic reward signals based on COS and actions.
    
    Args:
        cos_sequence: COS sequence
        actions: Backoff actions
        capacity_threshold: Capacity threshold for reward computation
    
    Returns:
        rewards: Generated reward signals
    """
    num_samples = len(actions)
    rewards = np.zeros(num_samples)
    
    # Action mapping: {0: -2, 1: -1, 2: 0, 3: +1, 4: +2}
    action_mapping = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
    
    for i in range(num_samples):
        total_capacity = np.sum(cos_sequence[i])
        cos_variance = np.var(cos_sequence[i])
        backoff_step = action_mapping[actions[i]]
        
        # Base reward from capacity utilization
        capacity_reward = 1.0 - abs(total_capacity - 0.7)  # Optimal at 70%
        
        # Fairness reward (lower variance is better)
        fairness_reward = 1.0 - cos_variance
        
        # Action consistency reward
        action_reward = 1.0 - abs(backoff_step) / 4.0  # Neutral actions preferred
        
        # Capacity constraint penalty
        capacity_penalty = 0.0
        if total_capacity > capacity_threshold:
            capacity_penalty = -2.0 * (total_capacity - capacity_threshold)
        
        # Combine rewards
        total_reward = (
            0.4 * capacity_reward +
            0.3 * fairness_reward +
            0.2 * action_reward +
            capacity_penalty
        )
        
        # Add small random component
        total_reward += np.random.normal(0, 0.05)
        
        # Clamp to reasonable range
        rewards[i] = np.clip(total_reward, -1.0, 1.0)
    
    return rewards


def generate_capacity_usage(
    cos_sequence: np.ndarray,
    noise_level: float = 0.05
) -> np.ndarray:
    """
    Generate capacity usage values based on COS sequence.
    
    Args:
        cos_sequence: COS sequence
        noise_level: Level of noise to add
    
    Returns:
        capacity_usage: Generated capacity usage values
    """
    # Base capacity is sum of COS values
    base_capacity = np.sum(cos_sequence, axis=1)
    
    # Add some temporal correlation and noise
    capacity_usage = base_capacity.copy()
    
    # Add temporal smoothing
    for i in range(1, len(capacity_usage)):
        capacity_usage[i] = 0.7 * capacity_usage[i] + 0.3 * capacity_usage[i-1]
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(capacity_usage))
    capacity_usage += noise
    
    # Ensure values are in [0, 1] range
    capacity_usage = np.clip(capacity_usage, 0, 1)
    
    return capacity_usage


def generate_fake_data(
    num_samples: int,
    cos_dim: int,
    correlation_strength: float = 0.3,
    noise_level: float = 0.1,
    capacity_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Generate complete fake dataset for SPMA ML training.
    
    Args:
        num_samples: Number of samples to generate
        cos_dim: Number of COS channels
        correlation_strength: Strength of correlation between channels
        noise_level: Level of random noise
        capacity_threshold: Capacity threshold for action selection
    
    Returns:
        dataframe: Generated dataset as pandas DataFrame
    """
    print(f"Generating {num_samples} samples with {cos_dim} COS channels...")
    
    # Generate COS sequence
    cos_sequence = generate_cos_sequence(
        num_samples, cos_dim, correlation_strength, noise_level
    )
    
    # Generate backoff actions
    actions = generate_backoff_actions(num_samples, cos_sequence, capacity_threshold)
    
    # Generate rewards
    rewards = generate_rewards(cos_sequence, actions, capacity_threshold)
    
    # Generate capacity usage
    capacity_usage = generate_capacity_usage(cos_sequence)
    
    # Create DataFrame
    data = {
        'timestamp': np.arange(num_samples) * 0.1  # 0.1 second intervals
    }
    
    # Add COS columns
    for i in range(cos_dim):
        data[f'cos_{i}'] = cos_sequence[:, i]
    
    # Add other columns
    data['backoff_action'] = actions
    data['reward'] = rewards
    data['capacity_used'] = capacity_usage
    
    dataframe = pd.DataFrame(data)
    
    print(f"Generated dataset with shape: {dataframe.shape}")
    print(f"COS range: [{dataframe[[f'cos_{i}' for i in range(cos_dim)]].values.min():.3f}, "
          f"{dataframe[[f'cos_{i}' for i in range(cos_dim)]].values.max():.3f}]")
    print(f"Reward range: [{dataframe['reward'].min():.3f}, {dataframe['reward'].max():.3f}]")
    print(f"Capacity range: [{dataframe['capacity_used'].min():.3f}, {dataframe['capacity_used'].max():.3f}]")
    print(f"Action distribution: {dataframe['backoff_action'].value_counts().sort_index().to_dict()}")
    
    return dataframe


def split_data(dataframe: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets.
    
    Args:
        dataframe: Input dataframe
        train_ratio: Ratio of data for training
    
    Returns:
        train_data: Training data
        val_data: Validation data
    """
    num_samples = len(dataframe)
    train_size = int(num_samples * train_ratio)
    
    train_data = dataframe.iloc[:train_size].copy()
    val_data = dataframe.iloc[train_size:].copy()
    
    print(f"Split data: {len(train_data)} training, {len(val_data)} validation samples")
    
    return train_data, val_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate fake data for SPMA ML')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for generated data')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--cos-dim', type=int, default=14,
                       help='Number of COS channels')
    parser.add_argument('--correlation-strength', type=float, default=0.3,
                       help='Strength of correlation between channels')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Level of random noise')
    parser.add_argument('--capacity-threshold', type=float, default=0.8,
                       help='Capacity threshold for action selection')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate fake data
    dataframe = generate_fake_data(
        args.num_samples,
        args.cos_dim,
        args.correlation_strength,
        args.noise_level,
        args.capacity_threshold
    )
    
    # Split into train/validation
    train_data, val_data = split_data(dataframe, args.train_ratio)
    
    # Save data
    train_path = os.path.join(args.output_dir, 'train.csv')
    val_path = os.path.join(args.output_dir, 'val.csv')
    
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    
    print(f"Saved training data to {train_path}")
    print(f"Saved validation data to {val_path}")
    
    # Generate summary statistics
    summary = {
        'num_samples': args.num_samples,
        'cos_dim': args.cos_dim,
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'cos_range': [float(dataframe[[f'cos_{i}' for i in range(args.cos_dim)]].values.min()),
                      float(dataframe[[f'cos_{i}' for i in range(args.cos_dim)]].values.max())],
        'reward_range': [float(dataframe['reward'].min()), float(dataframe['reward'].max())],
        'capacity_range': [float(dataframe['capacity_used'].min()), float(dataframe['capacity_used'].max())],
        'action_distribution': dataframe['backoff_action'].value_counts().sort_index().to_dict()
    }
    
    summary_path = os.path.join(args.output_dir, 'data_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    print(f"Saved data summary to {summary_path}")
    print("Fake data generation completed successfully!")


if __name__ == '__main__':
    main()
