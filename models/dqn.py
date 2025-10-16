"""
Deep Q-Network (DQN) for choosing backoff actions in SPMA.

This module implements a compact DQN with experience replay, target network,
and epsilon-greedy exploration for efficient edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple, Optional
from collections import deque
import numpy as np


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """
    Deep Q-Network for backoff action selection.
    
    Architecture:
    - Small MLP network for fast inference on edge devices
    - Experience replay buffer for stable learning
    - Target network for reducing correlation in Q-learning
    - Epsilon-greedy exploration with decay
    
    Args:
        state_dim: Dimension of state vector
        action_dim: Number of discrete actions (backoff steps)
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate for optimizer
        gamma: Discount factor for future rewards
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Exploration rate decay factor
        target_update: Frequency of target network updates
    """
    
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 5,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update: int = 100
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.update_count = 0
        
        # Build Q-network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.q_network = nn.Sequential(*layers)
        
        # Target network (copy of main network)
        self.target_network = nn.Sequential(*layers)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Action mapping: {-2, -1, 0, +1, +2} backoff steps
        self.action_mapping = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
        self.reverse_action_mapping = {v: k for k, v in self.action_mapping.items()}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network."""
        return self.q_network(state)
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            training: Whether in training mode (affects exploration)
        
        Returns:
            action: Selected action (backoff step: -2 to +2)
        """
        if training and random.random() < self.epsilon:
            # Random exploration
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Greedy action selection
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state_tensor)
                action_idx = q_values.argmax().item()
        
        # Convert to actual backoff step
        return self.reverse_action_mapping[action_idx]
    
    def update_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def push_experience(self, state: np.ndarray, action: int, reward: float,
                       next_state: np.ndarray, done: bool):
        """Add experience to replay buffer."""
        # Convert action to index
        action_idx = self.action_mapping[action]
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def train_step(self, batch_size: int = 32) -> float:
        """
        Perform one training step using experience replay.
        
        Args:
            batch_size: Size of training batch
        
        Returns:
            loss: Training loss value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def compute_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions given a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor).squeeze().numpy()
        
        # Convert to action mapping
        action_q_values = {}
        for action, idx in self.action_mapping.items():
            action_q_values[action] = q_values[idx]
        
        return action_q_values
    
    def get_model_size(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
    
    def export_for_mobile(self) -> torch.jit.ScriptModule:
        """Export model optimized for mobile/edge deployment."""
        self.eval()
        
        # Create dummy input for tracing
        dummy_input = torch.randn(1, self.state_dim)
        
        # Trace the model
        traced_model = torch.jit.trace(self.q_network, dummy_input)
        
        return traced_model


def create_dqn_from_config(config: dict) -> DQN:
    """Create DQN model from configuration dictionary."""
    return DQN(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dims=config['hidden_dims'],
        learning_rate=config.get('learning_rate', 0.001),
        gamma=config.get('gamma', 0.99),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.01),
        epsilon_decay=config.get('epsilon_decay', 0.995),
        target_update=config.get('target_update', 100)
    )
