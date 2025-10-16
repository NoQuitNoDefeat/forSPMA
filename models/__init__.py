"""
SPMA ML Models Package

This package contains the neural network models for SPMA (Self-Protecting Multi-Access):
- TCN: Temporal Convolutional Network for predicting non-stationary COS
- DQN: Deep Q-Network for choosing backoff actions

Both models are optimized for edge deployment with INT8 quantization support.
"""

from .tcn import TCN
from .dqn import DQN

__all__ = ['TCN', 'DQN']
