#!/usr/bin/env python3
"""
DQN Training Script for SPMA ML

This script trains the Deep Q-Network (DQN) to choose optimal backoff actions
in the SPMA environment with experience replay and target network updates.

Usage:
    python train_dqn.py --config config.yaml --env-type stub
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from typing import Dict, Any, Tuple
import time
from tqdm import tqdm

from models.dqn import DQN, create_dqn_from_config
from envs.spma_stub_env import SPMASubEnvironment, create_spma_stub_env
from envs.ns3_bridge_env import NS3BridgeEnvironment, create_ns3_bridge_env
from utils.logger import setup_logger, TrainingLogger, log_model_info
from utils.metrics import MetricsTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN for SPMA backoff action selection')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--env-type', type=str, default='stub', choices=['stub', 'ns3'],
                       help='Environment type to use for training')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (auto/cpu/cuda)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num-episodes', type=int, default=None,
                       help='Number of training episodes (overrides config)')
    
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


def create_environment(env_type: str, config: Dict[str, Any]):
    """Create training environment."""
    if env_type == 'stub':
        env = create_spma_stub_env(config)
    elif env_type == 'ns3':
        env = create_ns3_bridge_env(config)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    return env


def create_model(config: Dict[str, Any], device: torch.device) -> DQN:
    """Create DQN model from configuration."""
    dqn_config = config['dqn']
    model = create_dqn_from_config(dqn_config)
    model = model.to(device)
    return model


def train_episode(
    model: DQN,
    env,
    device: torch.device,
    config: Dict[str, Any],
    episode: int
) -> Dict[str, float]:
    """Train DQN for one episode."""
    model.train()
    
    # Reset environment
    state = env.reset()
    done = False
    
    episode_reward = 0.0
    episode_length = 0
    episode_loss = 0.0
    num_updates = 0
    
    while not done:
        # Select action
        action = model.get_action(state, training=True)
        
        # Execute action in environment
        next_state, reward, done, info = env.step(action)
        
        # Store experience in replay buffer
        model.push_experience(state, action, reward, next_state, done)
        
        # Train model if buffer has enough samples
        training_config = config['training']
        if len(model.replay_buffer) >= training_config['min_buffer_size']:
            loss = model.train_step(batch_size=training_config['batch_size'])
            episode_loss += loss
            num_updates += 1
        
        # Update state
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        # Update epsilon
        model.update_epsilon()
    
    # Average loss
    avg_loss = episode_loss / max(num_updates, 1)
    
    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'episode_loss': avg_loss,
        'epsilon': model.epsilon,
        'buffer_size': len(model.replay_buffer)
    }


def evaluate_episode(
    model: DQN,
    env,
    device: torch.device,
    num_eval_episodes: int = 5
) -> Dict[str, float]:
    """Evaluate DQN for multiple episodes."""
    model.eval()
    
    eval_rewards = []
    eval_lengths = []
    
    for _ in range(num_eval_episodes):
        state = env.reset()
        done = False
        
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            # Greedy action selection (no exploration)
            action = model.get_action(state, training=False)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if episode_length >= 1000:  # Safety limit
                break
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
    
    return {
        'eval_reward_mean': np.mean(eval_rewards),
        'eval_reward_std': np.std(eval_rewards),
        'eval_length_mean': np.mean(eval_lengths),
        'eval_length_std': np.std(eval_lengths)
    }


def save_checkpoint(
    model: DQN,
    episode: int,
    metrics: Dict[str, float],
    filepath: str
):
    """Save DQN checkpoint."""
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.q_network.state_dict(),
        'target_network_state_dict': model.target_network.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'epsilon': model.epsilon,
        'update_count': model.update_count,
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: DQN,
    filepath: str
) -> Tuple[int, Dict[str, float]]:
    """Load DQN checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.q_network.load_state_dict(checkpoint['model_state_dict'])
    model.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.epsilon = checkpoint['epsilon']
    model.update_count = checkpoint['update_count']
    
    return checkpoint['episode'], checkpoint['metrics']


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
        model_name='dqn',
        level=config['logging']['level']
    )
    
    logger.logger.info("Starting DQN training")
    logger.logger.info(f"Configuration: {args.config}")
    logger.logger.info(f"Environment type: {args.env_type}")
    logger.logger.info(f"Device: {device}")
    
    # Log configuration
    log_configuration(logger.logger, config, "Training Configuration")
    
    # Create environment
    env = create_environment(args.env_type, config)
    logger.logger.info(f"Environment created: {type(env).__name__}")
    
    # Create model
    model = create_model(config, device)
    
    # Log model info
    log_model_info(logger.logger, model.q_network, "DQN")
    
    # Training parameters
    num_episodes = args.num_episodes or config['training'].get('num_episodes', 1000)
    eval_interval = config['training'].get('eval_interval', 50)
    save_interval = config['training'].get('save_interval', 100)
    
    logger.logger.info(f"Training for {num_episodes} episodes")
    logger.logger.info(f"Evaluation every {eval_interval} episodes")
    
    # Load checkpoint if resuming
    start_episode = 0
    best_eval_reward = float('-inf')
    
    if args.resume:
        start_episode, metrics = load_checkpoint(model, args.resume)
        best_eval_reward = metrics.get('eval_reward_mean', float('-inf'))
        logger.logger.info(f"Resumed from episode {start_episode}")
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    metrics_tracker.start_training()
    
    # Training loop
    logger.start_training()
    
    progress_bar = tqdm(range(start_episode + 1, num_episodes + 1), desc="Training DQN")
    
    for episode in progress_bar:
        # Train episode
        train_metrics = train_episode(model, env, device, config, episode)
        
        # Update metrics tracker
        metrics_tracker.update_metrics(train_metrics)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Reward': f"{train_metrics['episode_reward']:.2f}",
            'Epsilon': f"{train_metrics['epsilon']:.3f}",
            'Buffer': train_metrics['buffer_size']
        })
        
        # Evaluation
        if episode % eval_interval == 0:
            eval_metrics = evaluate_episode(model, env, device)
            
            # Combine metrics
            all_metrics = {**train_metrics, **eval_metrics}
            
            # Log epoch results
            logger.log_epoch_end(episode, train_metrics, eval_metrics)
            
            # Update best model
            if eval_metrics['eval_reward_mean'] > best_eval_reward:
                best_eval_reward = eval_metrics['eval_reward_mean']
                best_model_path = os.path.join(args.output_dir, 'dqn_best_model.pt')
                save_checkpoint(model, episode, all_metrics, best_model_path)
                logger.logger.info(f"New best model saved with eval reward: {best_eval_reward:.4f}")
            
            # Update metrics tracker
            metrics_tracker.update_epoch_metrics(episode, all_metrics)
        
        # Save checkpoint
        if config['logging']['save_checkpoints'] and episode % save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'dqn_checkpoint_episode_{episode}.pt')
            save_checkpoint(model, episode, train_metrics, checkpoint_path)
    
    # Training completed
    logger.log_training_end()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'dqn_final_model.pt')
    save_checkpoint(model, episode, train_metrics, final_model_path)
    
    # Final evaluation
    logger.logger.info("Running final evaluation...")
    final_eval_metrics = evaluate_episode(model, env, device, num_eval_episodes=10)
    
    # Log best metrics
    best_metrics = logger.get_best_metrics()
    logger.logger.info(f"Best evaluation metrics: {best_metrics}")
    logger.logger.info(f"Final evaluation: {final_eval_metrics}")
    
    # Training summary
    training_summary = metrics_tracker.get_training_summary()
    logger.logger.info(f"Training summary: {training_summary}")
    
    # Close environment
    env.close()
    
    logger.logger.info("DQN training completed successfully")


if __name__ == '__main__':
    main()
