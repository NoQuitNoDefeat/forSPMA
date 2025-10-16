"""
Temporal Convolutional Network (TCN) for predicting non-stationary COS.

This module implements a causal, dilated TCN with residual connections and
optional depthwise separable convolutions for efficient edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalConv1d(nn.Module):
    """Causal 1D convolution ensuring temporal causality."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        dilation: int = 1,
        use_depthwise_separable: bool = False
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.dilation = dilation
        
        if use_depthwise_separable:
            # Depthwise separable convolution for efficiency
            self.depthwise = nn.Conv1d(
                in_channels, in_channels, kernel_size, 
                dilation=dilation, groups=in_channels, padding=self.padding
            )
            self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation, padding=self.padding
            )
        
        self.use_depthwise_separable = use_depthwise_separable
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_depthwise_separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        
        # Remove future information (causal padding)
        return x[:, :, :-self.padding] if self.padding > 0 else x


class ResidualBlock(nn.Module):
    """Residual block with dilated causal convolution and dropout."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
        use_depthwise_separable: bool = False
    ):
        super().__init__()
        
        # First causal convolution
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation,
            use_depthwise_separable
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second causal convolution  
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation,
            use_depthwise_separable
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if channel mismatch)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.residual is not None:
            residual = self.residual(residual)
        
        # First convolution + batch norm + activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second convolution + batch norm
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection + activation
        out = out + residual
        out = self.activation(out)
        
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network for COS prediction.
    
    Architecture:
    - Causal dilated convolutions ensure temporal causality
    - Residual connections accelerate training and improve gradient flow
    - Optional depthwise separable convolutions reduce parameters for edge deployment
    - Outputs non-stationary COS predictions for capacity-aware backoff decisions
    
    Args:
        input_dim: Number of input features (COS channels + priority info)
        channels: Base number of channels in the network
        levels: Number of TCN levels (dilation layers)
        kernel_size: Size of convolution kernels
        dropout: Dropout rate for regularization
        use_depthwise_separable: Use depthwise separable convolutions for efficiency
        n_prio: Number of priority features
    """
    
    def __init__(
        self,
        input_dim: int = 14,
        channels: int = 16,
        levels: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_depthwise_separable: bool = True,
        n_prio: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.channels = channels
        self.levels = levels
        self.kernel_size = kernel_size
        self.n_prio = n_prio
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, channels)
        
        # Build residual blocks with exponentially increasing dilation
        self.residual_blocks = nn.ModuleList()
        for i in range(levels):
            dilation = 2 ** i
            self.residual_blocks.append(
                ResidualBlock(
                    channels, channels, kernel_size, dilation, dropout,
                    use_depthwise_separable
                )
            )
        
        # Output layers for COS prediction and capacity estimation
        self.cos_head = nn.Linear(channels, input_dim - n_prio)  # COS channels only
        self.capacity_head = nn.Linear(channels, 1)  # Capacity usage prediction
        
        # Steady-state COS learning (residual learning for non-stationarity)
        # This learns the deviation from steady-state COS patterns
        self.steady_state_proj = nn.Linear(input_dim - n_prio, channels)
        self.steady_state_weight = nn.Parameter(torch.tensor(0.1))  # Learnable mixing weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of TCN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               Expected format: [cos_features, priority_features]
        
        Returns:
            cos_pred: Predicted COS values (batch_size, seq_len, cos_dim)
            capacity_pred: Predicted capacity usage (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Separate COS features and priority features
        cos_features = x[:, :, :self.input_dim - self.n_prio]  # COS channels
        prio_features = x[:, :, self.input_dim - self.n_prio:]  # Priority info
        
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch_size, seq_len, channels)
        
        # Transpose for 1D convolution: (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Transpose back: (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Residual learning: incorporate steady-state COS patterns
        # This helps the model learn deviations from steady-state behavior
        steady_state_features = self.steady_state_proj(cos_features)
        x = x + self.steady_state_weight * steady_state_features
        
        # Predict COS and capacity
        cos_pred = torch.sigmoid(self.cos_head(x))  # Clamp to [0,1]
        capacity_pred = torch.sigmoid(self.capacity_head(x))  # Clamp to [0,1]
        
        return cos_pred, capacity_pred
    
    def predict_cos(self, x: torch.Tensor) -> torch.Tensor:
        """Predict COS values only (for inference)."""
        cos_pred, _ = self.forward(x)
        return cos_pred
    
    def get_model_size(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def export_for_mobile(self) -> torch.jit.ScriptModule:
        """Export model optimized for mobile/edge deployment."""
        self.eval()
        
        # Create dummy input for tracing
        dummy_input = torch.randn(1, 32, self.input_dim)
        
        # Trace the model
        traced_model = torch.jit.trace(self.predict_cos, dummy_input)
        
        return traced_model


def create_tcn_from_config(config: dict) -> TCN:
    """Create TCN model from configuration dictionary."""
    return TCN(
        input_dim=config['input_dim'],
        channels=config['channels'],
        levels=config['levels'],
        kernel_size=config['kernel_size'],
        dropout=config.get('dropout', 0.1),
        use_depthwise_separable=config.get('use_depthwise_separable', True),
        n_prio=config['n_prio']
    )
