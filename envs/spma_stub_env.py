"""
Stub SPMA environment for DQN training and development.

This module provides a realistic simulation of SPMA (Self-Protecting Multi-Access)
dynamics without requiring ns-3 integration. It simulates channel occupancy,
capacity constraints, and reward computation for DQN training.
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Any, Tuple, Optional, List
import random


class SPMASubEnvironment(gym.Env):
    """
    Stub SPMA environment for DQN training.
    
    This environment simulates:
    - Multi-channel access with capacity constraints
    - Non-stationary channel occupancy patterns
    - Reward computation based on capacity utilization and fairness
    - Backoff action effects on channel access
    
    State space: [cos_features, capacity_usage, recent_rewards, priority_info]
    Action space: Discrete(5) - backoff steps {-2, -1, 0, +1, +2}
    """
    
    def __init__(
        self,
        cos_dim: int = 14,
        state_dim: int = 10,
        max_episode_steps: int = 1000,
        max_capacity: float = 1.0,
        reward_scale: float = 1.0,
        noise_level: float = 0.1
    ):
        super().__init__()
        
        self.cos_dim = cos_dim
        self.state_dim = state_dim
        self.max_episode_steps = max_episode_steps
        self.max_capacity = max_capacity
        self.reward_scale = reward_scale
        self.noise_level = noise_level
        
        # Action space: backoff steps {-2, -1, 0, +1, +2}
        self.action_space = spaces.Discrete(5)
        
        # State space: continuous values
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.current_cos = np.zeros(cos_dim)
        self.current_capacity = 0.0
        self.recent_rewards = np.zeros(5)  # Last 5 rewards
        self.backoff_history = np.zeros(10)  # Last 10 backoff actions
        self.priority_level = 0.5  # Current priority level
        
        # Channel dynamics parameters
        self.cos_drift = np.random.normal(0, 0.01, cos_dim)
        self.cos_periods = np.random.uniform(50, 200, cos_dim)  # Different periods for each channel
        self.cos_phases = np.random.uniform(0, 2*np.pi, cos_dim)
        
        # Capacity dynamics
        self.capacity_trend = 0.0
        self.capacity_noise = 0.0
        
        # Reward computation parameters
        self.fairness_weight = 0.3
        self.efficiency_weight = 0.4
        self.stability_weight = 0.3
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        
        # Initialize COS with realistic patterns
        self.current_cos = self._generate_initial_cos()
        
        # Initialize capacity
        self.current_capacity = np.sum(self.current_cos) * 0.7  # Start at 70% capacity
        
        # Initialize history buffers
        self.recent_rewards = np.zeros(5)
        self.backoff_history = np.zeros(10)
        
        # Initialize priority
        self.priority_level = random.uniform(0.3, 0.8)
        
        # Reset dynamics
        self.cos_drift = np.random.normal(0, 0.01, self.cos_dim)
        self.capacity_trend = random.uniform(-0.001, 0.001)
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Backoff action (0-4 mapping to {-2, -1, 0, +1, +2})
        
        Returns:
            observation: Next state observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # Convert action to backoff step
        backoff_step = action - 2  # Map {0,1,2,3,4} to {-2,-1,0,+1,+2}
        
        # Update environment dynamics
        self._update_cos_dynamics()
        self._update_capacity_dynamics()
        
        # Apply backoff action effect
        self._apply_backoff_action(backoff_step)
        
        # Compute reward
        reward = self._compute_reward(backoff_step)
        
        # Update state
        self.current_step += 1
        self.recent_rewards = np.roll(self.recent_rewards, 1)
        self.recent_rewards[0] = reward
        self.backoff_history = np.roll(self.backoff_history, 1)
        self.backoff_history[0] = backoff_step
        
        # Check if episode is done
        done = self.current_step >= self.max_episode_steps
        
        # Get next observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'cos': self.current_cos.copy(),
            'capacity': self.current_capacity,
            'backoff_step': backoff_step,
            'priority': self.priority_level,
            'step': self.current_step
        }
        
        return observation, reward, done, info
    
    def _generate_initial_cos(self) -> np.ndarray:
        """Generate initial COS values with realistic patterns."""
        # Base COS with some channels more occupied than others
        base_cos = np.random.beta(2, 5, self.cos_dim)  # Skewed towards lower values
        
        # Add some correlation between channels
        correlation_matrix = np.random.uniform(0.1, 0.3, (self.cos_dim, self.cos_dim))
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Apply correlation
        correlated_cos = np.dot(correlation_matrix, base_cos)
        
        # Ensure values are in [0, 1]
        return np.clip(correlated_cos, 0, 1)
    
    def _update_cos_dynamics(self):
        """Update COS values with non-stationary dynamics."""
        # Add periodic variations
        time_factor = self.current_step / 100.0
        periodic_changes = np.sin(
            2 * np.pi * time_factor / self.cos_periods + self.cos_phases
        ) * 0.1
        
        # Add drift
        drift_changes = self.cos_drift * np.random.normal(1, 0.1)
        
        # Add noise
        noise_changes = np.random.normal(0, self.noise_level, self.cos_dim)
        
        # Update COS
        self.current_cos += periodic_changes + drift_changes + noise_changes
        
        # Ensure bounds and capacity constraint
        self.current_cos = np.clip(self.current_cos, 0, 1)
        
        # Apply capacity constraint (soft penalty)
        total_capacity = np.sum(self.current_cos)
        if total_capacity > self.max_capacity:
            # Scale down proportionally
            scale_factor = self.max_capacity / total_capacity
            self.current_cos *= scale_factor
    
    def _update_capacity_dynamics(self):
        """Update capacity usage dynamics."""
        # Update capacity based on COS
        base_capacity = np.sum(self.current_cos)
        
        # Add trend and noise
        self.capacity_trend += np.random.normal(0, 0.0001)
        self.capacity_noise = np.random.normal(0, 0.02)
        
        # Update capacity
        self.current_capacity = base_capacity + self.capacity_trend + self.capacity_noise
        self.current_capacity = np.clip(self.current_capacity, 0, self.max_capacity)
    
    def _apply_backoff_action(self, backoff_step: int):
        """Apply backoff action effect on channel access."""
        # Backoff affects priority and channel access probability
        if backoff_step > 0:  # Positive backoff (more aggressive)
            # Increase access probability but risk interference
            access_boost = backoff_step * 0.1
            self.current_cos = np.clip(self.current_cos + access_boost, 0, 1)
            self.priority_level = min(1.0, self.priority_level + 0.1)
        elif backoff_step < 0:  # Negative backoff (more conservative)
            # Decrease access probability but reduce interference
            access_reduction = abs(backoff_step) * 0.05
            self.current_cos = np.clip(self.current_cos - access_reduction, 0, 1)
            self.priority_level = max(0.0, self.priority_level - 0.05)
        
        # Update priority based on recent performance
        avg_recent_reward = np.mean(self.recent_rewards[:3])
        if avg_recent_reward > 0.7:
            self.priority_level = min(1.0, self.priority_level + 0.02)
        elif avg_recent_reward < 0.3:
            self.priority_level = max(0.0, self.priority_level - 0.02)
    
    def _compute_reward(self, backoff_step: int) -> float:
        """Compute reward based on current state and action."""
        # Efficiency reward: based on capacity utilization
        capacity_efficiency = self.current_capacity / self.max_capacity
        efficiency_reward = capacity_efficiency * self.efficiency_weight
        
        # Fairness reward: based on channel balance
        cos_variance = np.var(self.current_cos)
        fairness_reward = (1.0 - cos_variance) * self.fairness_weight
        
        # Stability reward: based on action consistency
        action_consistency = 1.0 - np.std(self.backoff_history[:5])
        stability_reward = action_consistency * self.stability_weight
        
        # Capacity constraint penalty
        capacity_penalty = 0.0
        if self.current_capacity > self.max_capacity * 0.95:
            capacity_penalty = -10.0 * (self.current_capacity - self.max_capacity * 0.95)
        
        # Priority-based reward
        priority_reward = self.priority_level * 0.1
        
        # Combine rewards
        total_reward = (
            efficiency_reward + 
            fairness_reward + 
            stability_reward + 
            priority_reward + 
            capacity_penalty
        )
        
        # Scale reward
        total_reward *= self.reward_scale
        
        # Add small random component for exploration
        total_reward += np.random.normal(0, 0.01)
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Combine different state components
        obs_components = []
        
        # COS features (first 6 channels)
        obs_components.extend(self.current_cos[:6])
        
        # Capacity usage
        obs_components.append(self.current_capacity)
        
        # Recent reward average
        obs_components.append(np.mean(self.recent_rewards[:3]))
        
        # Priority level
        obs_components.append(self.priority_level)
        
        # Recent backoff action average
        obs_components.append(np.mean(self.backoff_history[:3]))
        
        # Ensure we have exactly state_dim components
        while len(obs_components) < self.state_dim:
            obs_components.append(0.0)
        
        obs = np.array(obs_components[:self.state_dim], dtype=np.float32)
        
        # Ensure observation is in valid range
        return np.clip(obs, 0.0, 1.0)
    
    def render(self, mode: str = 'human'):
        """Render environment state (placeholder for visualization)."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"COS: {self.current_cos[:5]}...")  # Show first 5 channels
            print(f"Capacity: {self.current_capacity:.3f}")
            print(f"Priority: {self.priority_level:.3f}")
            print(f"Recent Reward: {np.mean(self.recent_rewards[:3]):.3f}")
            print("-" * 40)
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]


def create_spma_stub_env(config: Dict[str, Any]) -> SPMASubEnvironment:
    """Create SPMA stub environment from configuration."""
    env_config = config.get('environment', {})
    
    return SPMASubEnvironment(
        cos_dim=config.get('tcn', {}).get('input_dim', 14) - config.get('tcn', {}).get('n_prio', 2),
        state_dim=config.get('dqn', {}).get('state_dim', 10),
        max_episode_steps=env_config.get('max_episode_steps', 1000),
        max_capacity=config.get('data', {}).get('max_capacity', 1.0),
        reward_scale=env_config.get('reward_scale', 1.0),
        noise_level=env_config.get('noise_level', 0.1)
    )
