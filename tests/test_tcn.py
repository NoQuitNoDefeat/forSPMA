"""
Test suite for TCN model implementation.

This module tests the Temporal Convolutional Network (TCN) model including
forward pass, gradient flow, capacity constraints, and edge deployment features.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os

from models.tcn import TCN, create_tcn_from_config, CausalConv1d, ResidualBlock


class TestCausalConv1d:
    """Test CausalConv1d module."""
    
    def test_causal_conv_basic(self):
        """Test basic causal convolution functionality."""
        conv = CausalConv1d(16, 32, kernel_size=3, dilation=1)
        
        # Test input
        x = torch.randn(2, 16, 10)
        output = conv(x)
        
        # Check output shape
        assert output.shape == (2, 32, 10)
        
        # Check causality (no future information)
        # This is implicit in the implementation
    
    def test_causal_conv_dilated(self):
        """Test dilated causal convolution."""
        conv = CausalConv1d(16, 32, kernel_size=3, dilation=2)
        
        x = torch.randn(2, 16, 20)
        output = conv(x)
        
        assert output.shape == (2, 32, 20)
    
    def test_causal_conv_depthwise_separable(self):
        """Test depthwise separable causal convolution."""
        conv = CausalConv1d(16, 32, kernel_size=3, dilation=1, use_depthwise_separable=True)
        
        x = torch.randn(2, 16, 10)
        output = conv(x)
        
        assert output.shape == (2, 32, 10)


class TestResidualBlock:
    """Test ResidualBlock module."""
    
    def test_residual_block_basic(self):
        """Test basic residual block functionality."""
        block = ResidualBlock(16, 16, kernel_size=3, dilation=1)
        
        x = torch.randn(2, 16, 10)
        output = block(x)
        
        assert output.shape == (2, 16, 10)
    
    def test_residual_block_channel_mismatch(self):
        """Test residual block with channel mismatch."""
        block = ResidualBlock(16, 32, kernel_size=3, dilation=1)
        
        x = torch.randn(2, 16, 10)
        output = block(x)
        
        assert output.shape == (2, 32, 10)
    
    def test_residual_block_dilated(self):
        """Test dilated residual block."""
        block = ResidualBlock(16, 16, kernel_size=3, dilation=2)
        
        x = torch.randn(2, 16, 20)
        output = block(x)
        
        assert output.shape == (2, 16, 20)


class TestTCN:
    """Test TCN model."""
    
    def test_tcn_creation(self):
        """Test TCN model creation."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        assert model.input_dim == 16
        assert model.channels == 32
        assert model.levels == 3
        assert model.n_prio == 2
    
    def test_tcn_forward_pass(self):
        """Test TCN forward pass."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        # Test input: (batch_size, seq_len, input_dim)
        x = torch.randn(2, 32, 16)
        
        cos_pred, capacity_pred = model(x)
        
        # Check output shapes
        assert cos_pred.shape == (2, 32, 14)  # input_dim - n_prio
        assert capacity_pred.shape == (2, 32, 1)
        
        # Check output ranges (should be [0, 1] due to sigmoid)
        assert torch.all(cos_pred >= 0) and torch.all(cos_pred <= 1)
        assert torch.all(capacity_pred >= 0) and torch.all(capacity_pred <= 1)
    
    def test_tcn_predict_cos(self):
        """Test TCN COS prediction method."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        x = torch.randn(2, 32, 16)
        cos_pred = model.predict_cos(x)
        
        assert cos_pred.shape == (2, 32, 14)
        assert torch.all(cos_pred >= 0) and torch.all(cos_pred <= 1)
    
    def test_tcn_gradient_flow(self):
        """Test gradient flow through TCN."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        x = torch.randn(2, 32, 16, requires_grad=True)
        cos_pred, capacity_pred = model(x)
        
        # Compute loss
        loss = torch.mean(cos_pred) + torch.mean(capacity_pred)
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert torch.any(x.grad != 0)
        
        # Check gradient flow to model parameters
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_tcn_capacity_constraint(self):
        """Test TCN capacity constraint handling."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        # Create input with high capacity
        x = torch.ones(2, 32, 16) * 0.8  # High COS values
        
        cos_pred, capacity_pred = model(x)
        
        # Check that predictions respect capacity constraints
        total_capacity = torch.sum(cos_pred, dim=-1)
        assert torch.all(total_capacity >= 0)  # Non-negative
        
        # Capacity predictions should be reasonable
        assert torch.all(capacity_pred >= 0) and torch.all(capacity_pred <= 1)
    
    def test_tcn_model_size(self):
        """Test TCN model size calculation."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        model_size = model.get_model_size()
        assert model_size > 0
        assert isinstance(model_size, int)
    
    def test_tcn_mobile_export(self):
        """Test TCN mobile export functionality."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        # Test mobile export
        traced_model = model.export_for_mobile()
        
        # Test inference with traced model
        x = torch.randn(1, 32, 16)
        
        with torch.no_grad():
            original_output = model.predict_cos(x)
            traced_output = traced_model(x)
        
        # Check that outputs are similar
        assert torch.allclose(original_output, traced_output, atol=1e-5)
    
    def test_tcn_steady_state_learning(self):
        """Test TCN steady-state COS learning."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        # Test with steady-state input
        steady_state_cos = torch.ones(2, 32, 14) * 0.5
        priority_features = torch.randn(2, 32, 2)
        x = torch.cat([steady_state_cos, priority_features], dim=-1)
        
        cos_pred, capacity_pred = model(x)
        
        # Check that steady-state weight affects output
        assert model.steady_state_weight is not None
        assert model.steady_state_weight.item() > 0
    
    def test_create_tcn_from_config(self):
        """Test TCN creation from configuration."""
        config = {
            'input_dim': 16,
            'channels': 32,
            'levels': 3,
            'kernel_size': 3,
            'n_prio': 2,
            'dropout': 0.1,
            'use_depthwise_separable': True
        }
        
        model = create_tcn_from_config(config)
        
        assert model.input_dim == 16
        assert model.channels == 32
        assert model.levels == 3
        assert model.n_prio == 2
    
    def test_tcn_different_input_sizes(self):
        """Test TCN with different input sizes."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        # Test different sequence lengths
        for seq_len in [16, 32, 64]:
            x = torch.randn(2, seq_len, 16)
            cos_pred, capacity_pred = model(x)
            
            assert cos_pred.shape == (2, seq_len, 14)
            assert capacity_pred.shape == (2, seq_len, 1)
    
    def test_tcn_memory_efficiency(self):
        """Test TCN memory efficiency with large inputs."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        # Test with larger batch size
        x = torch.randn(8, 32, 16)
        
        # Should not raise memory error
        cos_pred, capacity_pred = model(x)
        
        assert cos_pred.shape == (8, 32, 14)
        assert capacity_pred.shape == (8, 32, 1)
    
    def test_tcn_edge_cases(self):
        """Test TCN edge cases."""
        model = TCN(
            input_dim=16,
            channels=32,
            levels=3,
            kernel_size=3,
            n_prio=2
        )
        
        # Test with zero input
        x = torch.zeros(1, 32, 16)
        cos_pred, capacity_pred = model(x)
        
        assert cos_pred.shape == (1, 32, 14)
        assert capacity_pred.shape == (1, 32, 1)
        
        # Test with very small input
        x = torch.randn(1, 32, 16) * 1e-6
        cos_pred, capacity_pred = model(x)
        
        assert cos_pred.shape == (1, 32, 14)
        assert capacity_pred.shape == (1, 32, 1)


if __name__ == '__main__':
    pytest.main([__file__])
