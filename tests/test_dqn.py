"""
Test suite for DQN model implementation.

This module tests the Deep Q-Network (DQN) model including forward pass,
experience replay, target network updates, and action selection.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os

from models.dqn import DQN, create_dqn_from_config, ReplayBuffer


class TestReplayBuffer:
    """Test ReplayBuffer class."""
    
    def test_replay_buffer_creation(self):
        """Test replay buffer creation."""
        buffer = ReplayBuffer(capacity=1000)
        
        assert buffer.capacity == 1000
        assert len(buffer) == 0
    
    def test_replay_buffer_push(self):
        """Test adding experiences to buffer."""
        buffer = ReplayBuffer(capacity=1000)
        
        # Add experience
        state = np.random.randn(10)
        action = 2
        reward = 0.5
        next_state = np.random.randn(10)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
    
    def test_replay_buffer_sample(self):
        """Test sampling from replay buffer."""
        buffer = ReplayBuffer(capacity=1000)
        
        # Add multiple experiences
        for i in range(10):
            state = np.random.randn(10)
            action = i % 5
            reward = np.random.random()
            next_state = np.random.randn(10)
            done = i % 3 == 0
            
            buffer.push(state, action, reward, next_state, done)
        
        # Sample batch
        batch_size = 5
        batch = buffer.sample(batch_size)
        
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (batch_size, 10)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 10)
        assert dones.shape == (batch_size,)
    
    def test_replay_buffer_capacity_limit(self):
        """Test replay buffer capacity limit."""
        buffer = ReplayBuffer(capacity=5)
        
        # Add more experiences than capacity
        for i in range(10):
            state = np.random.randn(10)
            action = i % 5
            reward = np.random.random()
            next_state = np.random.randn(10)
            done = False
            
            buffer.push(state, action, reward, next_state, done)
        
        # Should only keep last 5 experiences
        assert len(buffer) == 5


class TestDQN:
    """Test DQN model."""
    
    def test_dqn_creation(self):
        """Test DQN model creation."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        assert model.state_dim == 10
        assert model.action_dim == 5
        assert model.epsilon == 1.0  # Initial epsilon
    
    def test_dqn_forward_pass(self):
        """Test DQN forward pass."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        # Test input
        state = torch.randn(2, 10)
        q_values = model(state)
        
        assert q_values.shape == (2, 5)
    
    def test_dqn_get_action_training(self):
        """Test action selection in training mode."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.99
        )
        
        state = np.random.randn(10)
        
        # In training mode with high epsilon, should get random actions
        actions = []
        for _ in range(100):
            action = model.get_action(state, training=True)
            actions.append(action)
        
        # Should get variety of actions due to exploration
        unique_actions = set(actions)
        assert len(unique_actions) > 1
    
    def test_dqn_get_action_evaluation(self):
        """Test action selection in evaluation mode."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.99
        )
        
        state = np.random.randn(10)
        
        # In evaluation mode, should get consistent actions
        actions = []
        for _ in range(10):
            action = model.get_action(state, training=False)
            actions.append(action)
        
        # Should get same action every time (greedy)
        assert all(action == actions[0] for action in actions)
    
    def test_dqn_epsilon_decay(self):
        """Test epsilon decay."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.9
        )
        
        initial_epsilon = model.epsilon
        
        # Decay epsilon
        model.update_epsilon()
        
        assert model.epsilon < initial_epsilon
        assert model.epsilon >= model.epsilon_end
    
    def test_dqn_push_experience(self):
        """Test adding experiences to replay buffer."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            buffer_size=1000
        )
        
        state = np.random.randn(10)
        action = 2
        reward = 0.5
        next_state = np.random.randn(10)
        done = False
        
        model.push_experience(state, action, reward, next_state, done)
        
        assert len(model.replay_buffer) == 1
    
    def test_dqn_train_step(self):
        """Test DQN training step."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            buffer_size=1000,
            min_buffer_size=10
        )
        
        # Add experiences to buffer
        for i in range(20):
            state = np.random.randn(10)
            action = i % 5
            reward = np.random.random()
            next_state = np.random.randn(10)
            done = i % 5 == 0
            
            model.push_experience(state, action, reward, next_state, done)
        
        # Train step
        loss = model.train_step(batch_size=10)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_dqn_target_network_update(self):
        """Test target network update."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            target_update=5
        )
        
        # Get initial target network weights
        initial_target_weights = model.target_network[0].weight.clone()
        
        # Update main network
        state = torch.randn(1, 10)
        target = torch.randn(1, 5)
        loss = torch.nn.functional.mse_loss(model(state), target)
        loss.backward()
        model.optimizer.step()
        
        # Update target network
        for _ in range(5):
            model.update_count += 1
            if model.update_count % model.target_update == 0:
                model.target_network.load_state_dict(model.q_network.state_dict())
                break
        
        # Target network should be updated
        assert not torch.equal(model.target_network[0].weight, initial_target_weights)
    
    def test_dqn_compute_q_values(self):
        """Test Q-value computation."""
        model = DQN(
            state_dim=10,
            action_dim=5
        )
        
        state = np.random.randn(10)
        q_values = model.compute_q_values(state)
        
        # Should return dictionary with action -> q_value mapping
        assert isinstance(q_values, dict)
        assert len(q_values) == 5
        
        # Check action keys
        expected_actions = [-2, -1, 0, 1, 2]
        assert all(action in q_values for action in expected_actions)
        
        # Check that values are numeric
        assert all(isinstance(q_val, (int, float)) for q_val in q_values.values())
    
    def test_dqn_model_size(self):
        """Test DQN model size calculation."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        model_size = model.get_model_size()
        assert model_size > 0
        assert isinstance(model_size, int)
    
    def test_dqn_checkpoint_save_load(self):
        """Test DQN checkpoint saving and loading."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        # Modify model state
        model.epsilon = 0.5
        model.update_count = 100
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
        
        model.save_checkpoint(checkpoint_path)
        
        # Create new model and load checkpoint
        new_model = DQN(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        new_model.load_checkpoint(checkpoint_path)
        
        # Check that state was restored
        assert new_model.epsilon == 0.5
        assert new_model.update_count == 100
        
        # Clean up
        os.unlink(checkpoint_path)
    
    def test_dqn_mobile_export(self):
        """Test DQN mobile export functionality."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        # Test mobile export
        traced_model = model.export_for_mobile()
        
        # Test inference with traced model
        x = torch.randn(1, 10)
        
        with torch.no_grad():
            original_output = model.q_network(x)
            traced_output = traced_model(x)
        
        # Check that outputs are similar
        assert torch.allclose(original_output, traced_output, atol=1e-5)
    
    def test_dqn_action_mapping(self):
        """Test action mapping between indices and backoff steps."""
        model = DQN(
            state_dim=10,
            action_dim=5
        )
        
        # Test action mapping
        assert model.action_mapping == {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
        assert model.reverse_action_mapping == {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
    
    def test_dqn_gradient_flow(self):
        """Test gradient flow through DQN."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        state = torch.randn(2, 10, requires_grad=True)
        q_values = model(state)
        
        # Compute loss
        target = torch.randn(2, 5)
        loss = torch.nn.functional.mse_loss(q_values, target)
        loss.backward()
        
        # Check that gradients exist
        assert state.grad is not None
        assert torch.any(state.grad != 0)
        
        # Check gradient flow to model parameters
        for param in model.q_network.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_dqn_different_state_sizes(self):
        """Test DQN with different state sizes."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            state = torch.randn(batch_size, 10)
            q_values = model(state)
            
            assert q_values.shape == (batch_size, 5)
    
    def test_dqn_edge_cases(self):
        """Test DQN edge cases."""
        model = DQN(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        # Test with zero state
        state = torch.zeros(1, 10)
        q_values = model(state)
        
        assert q_values.shape == (1, 5)
        
        # Test with very small state
        state = torch.randn(1, 10) * 1e-6
        q_values = model(state)
        
        assert q_values.shape == (1, 5)
    
    def test_create_dqn_from_config(self):
        """Test DQN creation from configuration."""
        config = {
            'state_dim': 10,
            'action_dim': 5,
            'hidden_dims': [64, 32],
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 0.9,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update': 50
        }
        
        model = create_dqn_from_config(config)
        
        assert model.state_dim == 10
        assert model.action_dim == 5
        assert model.epsilon == 0.9
        assert model.gamma == 0.99


if __name__ == '__main__':
    pytest.main([__file__])
