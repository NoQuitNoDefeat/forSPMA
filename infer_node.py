#!/usr/bin/env python3
"""
SPMA Node Inference Script

This script provides the main inference loop for SPMA node-side ML models.
It combines TCN COS prediction with DQN backoff action selection for
real-time decision making in the SPMA protocol.

Usage:
    python infer_node.py --config config.yaml --models-dir models/ --runtime 60
"""

import argparse
import torch
import torch.jit
import numpy as np
import yaml
import time
import threading
from typing import Dict, Any, Optional, Tuple
from collections import deque
import json

from utils.logger import setup_logger, log_inference_performance


class FeatureCollector:
    """
    Feature collection placeholder for real SPMA node integration.
    
    This class simulates the feature collection process that would normally
    be handled by the SPMA protocol stack. In a real implementation, this
    would interface with the network stack to collect:
    - Channel occupancy status (COS)
    - Priority information
    - Capacity utilization
    - Historical performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cos_dim = config['tcn']['input_dim'] - config['tcn']['n_prio']
        self.n_prio = config['tcn']['n_prio']
        
        # Feature history for temporal modeling
        self.feature_history = deque(maxlen=config['tcn']['window_size'])
        
        # Initialize with random features (placeholder)
        self._initialize_features()
    
    def _initialize_features(self):
        """Initialize feature history with random data."""
        window_size = self.config['tcn']['window_size']
        for _ in range(window_size):
            features = self._generate_random_features()
            self.feature_history.append(features)
    
    def _generate_random_features(self) -> np.ndarray:
        """Generate random features (placeholder for real data)."""
        # Generate realistic COS values
        cos_features = np.random.beta(2, 5, self.cos_dim)
        
        # Generate priority features
        priority_features = np.random.uniform(0.3, 0.8, self.n_prio)
        
        # Combine features
        features = np.concatenate([cos_features, priority_features])
        
        return features.astype(np.float32)
    
    def collect_features(self) -> np.ndarray:
        """
        Collect current features for inference.
        
        Returns:
            features: Current feature vector (cos_features + priority_features)
        """
        # In real implementation, this would:
        # 1. Query network stack for current COS
        # 2. Get priority information from protocol
        # 3. Collect capacity utilization metrics
        # 4. Update feature history
        
        # For now, generate new features and update history
        new_features = self._generate_random_features()
        self.feature_history.append(new_features)
        
        return new_features
    
    def get_sequence_features(self) -> np.ndarray:
        """
        Get feature sequence for TCN input.
        
        Returns:
            sequence: Feature sequence of shape (window_size, input_dim)
        """
        return np.array(list(self.feature_history))


class ActionMapper:
    """
    Maps DQN actions to SPMA protocol backoff actions.
    
    This class handles the translation between DQN output actions
    and the actual backoff steps used by the SPMA protocol.
    """
    
    def __init__(self):
        # Action mapping: {-2, -1, 0, +1, +2} backoff steps
        self.action_mapping = {
            0: -2,  # Most conservative
            1: -1,  # Conservative
            2: 0,   # Neutral
            3: 1,   # Aggressive
            4: 2    # Most aggressive
        }
        
        # Action descriptions for logging
        self.action_descriptions = {
            -2: "Most Conservative (-2)",
            -1: "Conservative (-1)",
            0: "Neutral (0)",
            1: "Aggressive (+1)",
            2: "Most Aggressive (+2)"
        }
    
    def map_action(self, action_idx: int) -> Tuple[int, str]:
        """
        Map DQN action index to backoff step and description.
        
        Args:
            action_idx: DQN action index (0-4)
        
        Returns:
            backoff_step: Backoff step for SPMA protocol
            description: Human-readable action description
        """
        backoff_step = self.action_mapping.get(action_idx, 0)
        description = self.action_descriptions[backoff_step]
        
        return backoff_step, description


class StateBuilder:
    """
    Builds DQN state vector from TCN predictions and system context.
    
    This class combines TCN COS predictions with additional system state
    information to create the state vector used by the DQN for action selection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_dim = config['dqn']['state_dim']
        self.cos_dim = config['tcn']['input_dim'] - config['tcn']['n_prio']
        
        # State history for context
        self.recent_cos_predictions = deque(maxlen=5)
        self.recent_actions = deque(maxlen=5)
        self.recent_rewards = deque(maxlen=5)
        
        # Initialize with default values
        self._initialize_state_history()
    
    def _initialize_state_history(self):
        """Initialize state history with default values."""
        for _ in range(5):
            self.recent_cos_predictions.append(np.zeros(self.cos_dim))
            self.recent_actions.append(0)
            self.recent_rewards.append(0.0)
    
    def build_state(
        self,
        cos_prediction: np.ndarray,
        capacity_usage: float,
        current_action: int
    ) -> np.ndarray:
        """
        Build DQN state vector from current context.
        
        Args:
            cos_prediction: TCN COS prediction
            capacity_usage: Current capacity usage
            current_action: Current backoff action
        
        Returns:
            state: DQN state vector
        """
        # Update history
        self.recent_cos_predictions.append(cos_prediction)
        self.recent_actions.append(current_action)
        
        # Build state components
        state_components = []
        
        # Current COS prediction (first 6 channels)
        state_components.extend(cos_prediction[:6])
        
        # Capacity usage
        state_components.append(capacity_usage)
        
        # Recent reward average (placeholder)
        avg_recent_reward = np.mean(list(self.recent_rewards))
        state_components.append(avg_recent_reward)
        
        # Priority level (derived from COS variance)
        cos_variance = np.var(cos_prediction)
        priority_level = 1.0 - cos_variance  # Higher variance = lower priority
        state_components.append(priority_level)
        
        # Recent action consistency
        action_consistency = 1.0 - np.std(list(self.recent_actions))
        state_components.append(action_consistency)
        
        # Ensure we have exactly state_dim components
        while len(state_components) < self.state_dim:
            state_components.append(0.0)
        
        state = np.array(state_components[:self.state_dim], dtype=np.float32)
        
        # Ensure state is in valid range
        return np.clip(state, 0.0, 1.0)
    
    def update_reward(self, reward: float):
        """Update recent reward history."""
        self.recent_rewards.append(reward)


class SPMAInferenceNode:
    """
    Main SPMA inference node combining TCN and DQN models.
    
    This class orchestrates the inference pipeline:
    1. Collect features from network stack
    2. Predict COS using TCN
    3. Build state for DQN
    4. Select backoff action using DQN
    5. Execute action in SPMA protocol
    """
    
    def __init__(self, config: Dict[str, Any], models_dir: str):
        self.config = config
        self.models_dir = models_dir
        
        # Setup logging
        self.logger = setup_logger('spma_inference', level='INFO')
        
        # Initialize components
        self.feature_collector = FeatureCollector(config)
        self.action_mapper = ActionMapper()
        self.state_builder = StateBuilder(config)
        
        # Load models
        self.tcn_model = None
        self.dqn_model = None
        self._load_models()
        
        # Inference statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.action_distribution = {i: 0 for i in range(5)}
        
        # Performance monitoring
        self.inference_times = []
        self.max_inference_times = 1000  # Keep last 1000 inference times
    
    def _load_models(self):
        """Load TCN and DQN models."""
        self.logger.info("Loading models...")
        
        # Load TCN model
        tcn_path = os.path.join(self.models_dir, 'tcn_int8.ptl')
        if os.path.exists(tcn_path):
            self.tcn_model = torch.jit.load(tcn_path, map_location='cpu')
            self.logger.info(f"Loaded TCN model from {tcn_path}")
        else:
            self.logger.error(f"TCN model not found at {tcn_path}")
            raise FileNotFoundError(f"TCN model not found at {tcn_path}")
        
        # Load DQN model
        dqn_path = os.path.join(self.models_dir, 'dqn_int8.ptl')
        if os.path.exists(dqn_path):
            self.dqn_model = torch.jit.load(dqn_path, map_location='cpu')
            self.logger.info(f"Loaded DQN model from {dqn_path}")
        else:
            self.logger.error(f"DQN model not found at {dqn_path}")
            raise FileNotFoundError(f"DQN model not found at {dqn_path}")
        
        # Set models to evaluation mode
        self.tcn_model.eval()
        self.dqn_model.eval()
        
        self.logger.info("Models loaded successfully")
    
    def predict_cos(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict COS using TCN model.
        
        Args:
            features: Feature sequence for TCN input
        
        Returns:
            cos_prediction: Predicted COS values
            capacity_usage: Predicted capacity usage
        """
        # Prepare input tensor
        input_tensor = torch.from_numpy(features).unsqueeze(0).float()
        
        # Run TCN inference
        with torch.no_grad():
            cos_pred, capacity_pred = self.tcn_model(input_tensor)
            
            # Get predictions for last timestep
            cos_prediction = cos_pred[0, -1, :].numpy()
            capacity_usage = capacity_pred[0, -1, 0].item()
        
        return cos_prediction, capacity_usage
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select backoff action using DQN model.
        
        Args:
            state: DQN state vector
        
        Returns:
            action_idx: Selected action index (0-4)
        """
        # Prepare input tensor
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()
        
        # Run DQN inference
        with torch.no_grad():
            q_values = self.dqn_model(state_tensor)
            action_idx = q_values.argmax().item()
        
        return action_idx
    
    def run_inference_cycle(self) -> Dict[str, Any]:
        """
        Run one complete inference cycle.
        
        Returns:
            result: Dictionary containing inference results
        """
        start_time = time.time()
        
        # 1. Collect features
        current_features = self.feature_collector.collect_features()
        sequence_features = self.feature_collector.get_sequence_features()
        
        # 2. Predict COS using TCN
        cos_prediction, capacity_usage = self.predict_cos(sequence_features)
        
        # 3. Build state for DQN
        current_action = 0  # Default action
        state = self.state_builder.build_state(cos_prediction, capacity_usage, current_action)
        
        # 4. Select action using DQN
        action_idx = self.select_action(state)
        backoff_step, action_description = self.action_mapper.map_action(action_idx)
        
        # 5. Compute reward (placeholder)
        reward = self._compute_reward(cos_prediction, capacity_usage, backoff_step)
        
        # Update state builder with reward
        self.state_builder.update_reward(reward)
        
        # Record inference time
        inference_time = time.time() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # Update statistics
        self.action_distribution[action_idx] += 1
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_inference_times:
            self.inference_times.pop(0)
        
        # Prepare result
        result = {
            'inference_time': inference_time,
            'cos_prediction': cos_prediction.tolist(),
            'capacity_usage': capacity_usage,
            'action_idx': action_idx,
            'backoff_step': backoff_step,
            'action_description': action_description,
            'reward': reward,
            'state': state.tolist()
        }
        
        return result
    
    def _compute_reward(self, cos_prediction: np.ndarray, capacity_usage: float, backoff_step: int) -> float:
        """Compute reward for current state and action (placeholder)."""
        # Simple reward based on capacity utilization and action consistency
        capacity_reward = 1.0 - abs(capacity_usage - 0.7)  # Prefer 70% capacity
        action_reward = 1.0 - abs(backoff_step) / 4.0  # Prefer neutral actions
        
        total_reward = (capacity_reward + action_reward) / 2.0
        return total_reward
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        if not self.inference_times:
            return {}
        
        times_ms = [t * 1000 for t in self.inference_times]
        
        stats = {
            'total_inferences': self.inference_count,
            'avg_inference_time_ms': np.mean(times_ms),
            'std_inference_time_ms': np.std(times_ms),
            'min_inference_time_ms': np.min(times_ms),
            'max_inference_time_ms': np.max(times_ms),
            'p95_inference_time_ms': np.percentile(times_ms, 95),
            'p99_inference_time_ms': np.percentile(times_ms, 99),
            'action_distribution': self.action_distribution
        }
        
        return stats
    
    def run_continuous_inference(self, runtime_seconds: int, log_interval: int = 10):
        """
        Run continuous inference for specified runtime.
        
        Args:
            runtime_seconds: Total runtime in seconds
            log_interval: Logging interval in seconds
        """
        self.logger.info(f"Starting continuous inference for {runtime_seconds} seconds")
        
        start_time = time.time()
        last_log_time = start_time
        
        while time.time() - start_time < runtime_seconds:
            # Run inference cycle
            result = self.run_inference_cycle()
            
            # Log periodically
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                stats = self.get_performance_stats()
                self.logger.info(f"Inference #{self.inference_count}: "
                               f"Action={result['action_description']}, "
                               f"Capacity={result['capacity_usage']:.3f}, "
                               f"Time={result['inference_time']*1000:.2f}ms")
                last_log_time = current_time
        
        # Final statistics
        final_stats = self.get_performance_stats()
        self.logger.info("Inference completed")
        self.logger.info(f"Total inferences: {final_stats['total_inferences']}")
        self.logger.info(f"Average inference time: {final_stats['avg_inference_time_ms']:.2f}ms")
        self.logger.info(f"P95 inference time: {final_stats['p95_inference_time_ms']:.2f}ms")
        
        # Log performance
        log_inference_performance(
            self.logger, self.inference_times, "SPMA Node Inference", "CPU"
        )
        
        return final_stats


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='SPMA Node Inference')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--runtime', type=int, default=60,
                       help='Runtime in seconds')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval in seconds')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for inference results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create inference node
    inference_node = SPMAInferenceNode(config, args.models_dir)
    
    # Run continuous inference
    stats = inference_node.run_continuous_inference(
        args.runtime, args.log_interval
    )
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
