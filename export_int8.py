#!/usr/bin/env python3
"""
INT8 Export Script for SPMA ML Models

This script exports trained TCN and DQN models to INT8 quantized TorchScript
for efficient edge deployment. Supports multiple export formats and optimization
levels for different deployment targets.

Usage:
    python export_int8.py --model tcn --checkpoint models/tcn_best_model.pt --output models/tcn_int8.ptl
    python export_int8.py --model dqn --checkpoint models/dqn_best_model.pt --output models/dqn_int8.ptl
"""

import argparse
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.jit import ScriptModule
import numpy as np
import yaml
import os
from typing import Dict, Any, Optional, Tuple
import time

from models.tcn import TCN, create_tcn_from_config
from models.dqn import DQN, create_dqn_from_config
from utils.logger import setup_logger, log_inference_performance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export SPMA ML models to INT8 TorchScript')
    parser.add_argument('--model', type=str, choices=['tcn', 'dqn'], required=True,
                       help='Model type to export')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for exported model')
    parser.add_argument('--calibration-steps', type=int, default=100,
                       help='Number of calibration steps for quantization')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmark after export')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for export and benchmarking')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_inputs(model_type: str, config: Dict[str, Any]) -> torch.Tensor:
    """Create dummy inputs for model tracing and calibration."""
    if model_type == 'tcn':
        tcn_config = config['tcn']
        seq_len = tcn_config.get('window_size', 32)
        input_dim = tcn_config['input_dim']
        dummy_input = torch.randn(1, seq_len, input_dim)
    elif model_type == 'dqn':
        dqn_config = config['dqn']
        state_dim = dqn_config['state_dim']
        dummy_input = torch.randn(1, state_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return dummy_input


def load_model_from_checkpoint(
    model_type: str,
    checkpoint_path: str,
    config: Dict[str, Any]
) -> nn.Module:
    """Load trained model from checkpoint."""
    if model_type == 'tcn':
        model = create_tcn_from_config(config['tcn'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'dqn':
        model = create_dqn_from_config(config['dqn'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.q_network.load_state_dict(checkpoint['model_state_dict'])
        # Use only the Q-network for inference
        model = model.q_network
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model


def prepare_model_for_quantization(
    model: nn.Module,
    model_type: str,
    dummy_input: torch.Tensor
) -> nn.Module:
    """
    Prepare model for quantization by fusing modules and setting quantization config.
    
    Args:
        model: Model to prepare
        model_type: Type of model (tcn/dqn)
        dummy_input: Dummy input for tracing
    
    Returns:
        prepared_model: Model prepared for quantization
    """
    # Set model to evaluation mode
    model.eval()
    
    # Fuse compatible modules for efficiency
    if model_type == 'tcn':
        # Fuse Conv1d + BatchNorm + ReLU modules
        model = torch.quantization.fuse_modules(model, [
            ['residual_blocks.0.conv1', 'residual_blocks.0.bn1'],
            ['residual_blocks.0.conv2', 'residual_blocks.0.bn2'],
        ])
        
        # Add quantization stubs
        model = torch.quantization.QuantStub()(model)
        model = torch.quantization.DeQuantStub()(model)
    
    elif model_type == 'dqn':
        # Fuse Linear + ReLU modules
        model = torch.quantization.fuse_modules(model, [
            ['0', '1'],  # Linear + ReLU
            ['3', '4'],  # Linear + ReLU
        ])
        
        # Add quantization stubs
        model = torch.quantization.QuantStub()(model)
        model = torch.quantization.DeQuantStub()(model)
    
    # Set quantization configuration
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    
    return model


def calibrate_model(
    model: nn.Module,
    model_type: str,
    config: Dict[str, Any],
    calibration_steps: int,
    device: torch.device
) -> nn.Module:
    """
    Calibrate model for quantization using representative data.
    
    Args:
        model: Model to calibrate
        model_type: Type of model (tcn/dqn)
        config: Configuration dictionary
        calibration_steps: Number of calibration steps
    
    Returns:
        calibrated_model: Calibrated model ready for quantization
    """
    model.eval()
    
    # Prepare model for quantization
    dummy_input = create_dummy_inputs(model_type, config)
    model = prepare_model_for_quantization(model, model_type, dummy_input)
    
    # Move to device
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Prepare model for calibration
    model = quantization.prepare(model)
    
    # Run calibration with representative data
    logger = setup_logger('export', level='INFO')
    logger.info(f"Running calibration with {calibration_steps} steps...")
    
    with torch.no_grad():
        for step in range(calibration_steps):
            # Generate representative input data
            if model_type == 'tcn':
                # Generate realistic COS data
                input_data = torch.rand_like(dummy_input)
                # Add some correlation between channels
                input_data = input_data * 0.8 + 0.1
                # Ensure capacity constraints
                cos_sum = torch.sum(input_data, dim=-1)
                if torch.any(cos_sum > 1.0):
                    scale = 1.0 / cos_sum
                    input_data = input_data * scale.unsqueeze(-1)
            else:  # dqn
                # Generate realistic state data
                input_data = torch.rand_like(dummy_input)
                input_data = torch.clamp(input_data, 0.0, 1.0)
            
            # Forward pass for calibration
            _ = model(input_data)
            
            if step % 20 == 0:
                logger.info(f"Calibration step {step}/{calibration_steps}")
    
    # Convert to quantized model
    quantized_model = quantization.convert(model)
    
    logger.info("Calibration completed successfully")
    return quantized_model


def export_torchscript(
    model: nn.Module,
    model_type: str,
    config: Dict[str, Any],
    output_path: str,
    device: torch.device
) -> ScriptModule:
    """
    Export model to TorchScript format.
    
    Args:
        model: Model to export
        model_type: Type of model (tcn/dqn)
        config: Configuration dictionary
        output_path: Path to save exported model
        device: Device for export
    
    Returns:
        traced_model: Traced TorchScript model
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = create_dummy_inputs(model_type, config)
    dummy_input = dummy_input.to(device)
    
    logger = setup_logger('export', level='INFO')
    logger.info(f"Tracing {model_type.upper()} model...")
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    # Optimize for mobile deployment
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save traced model
    traced_model.save(output_path)
    
    logger.info(f"Model exported successfully to {output_path}")
    
    # Log model size
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Exported model size: {model_size_mb:.2f} MB")
    
    return traced_model


def benchmark_inference(
    model: ScriptModule,
    model_type: str,
    config: Dict[str, Any],
    device: torch.device,
    num_runs: int = 1000
) -> Dict[str, float]:
    """
    Benchmark inference performance of exported model.
    
    Args:
        model: Traced TorchScript model
        model_type: Type of model (tcn/dqn)
        config: Configuration dictionary
        device: Device for benchmarking
        num_runs: Number of benchmark runs
    
    Returns:
        benchmark_results: Dictionary of benchmark metrics
    """
    model.eval()
    
    # Create dummy input
    dummy_input = create_dummy_inputs(model_type, config)
    dummy_input = dummy_input.to(device)
    
    # Warmup runs
    logger = setup_logger('export', level='INFO')
    logger.info("Running warmup...")
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark runs
    logger.info(f"Running benchmark with {num_runs} iterations...")
    
    inference_times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            if i % 100 == 0:
                logger.info(f"Benchmark progress: {i}/{num_runs}")
    
    # Compute statistics
    times_ms = [t * 1000 for t in inference_times]
    
    benchmark_results = {
        'mean_ms': np.mean(times_ms),
        'std_ms': np.std(times_ms),
        'min_ms': np.min(times_ms),
        'max_ms': np.max(times_ms),
        'p50_ms': np.percentile(times_ms, 50),
        'p95_ms': np.percentile(times_ms, 95),
        'p99_ms': np.percentile(times_ms, 99)
    }
    
    # Log results
    log_inference_performance(
        logger, inference_times, f"{model_type.upper()} (INT8)", device.type.upper()
    )
    
    return benchmark_results


def verify_model_accuracy(
    original_model: nn.Module,
    quantized_model: ScriptModule,
    model_type: str,
    config: Dict[str, Any],
    device: torch.device
) -> float:
    """
    Verify that quantized model maintains acceptable accuracy.
    
    Args:
        original_model: Original float model
        quantized_model: Quantized TorchScript model
        model_type: Type of model (tcn/dqn)
        config: Configuration dictionary
        device: Device for verification
    
    Returns:
        accuracy_score: Accuracy comparison score
    """
    original_model.eval()
    quantized_model.eval()
    
    # Create test inputs
    dummy_input = create_dummy_inputs(model_type, config)
    dummy_input = dummy_input.to(device)
    
    # Get predictions from both models
    with torch.no_grad():
        original_output = original_model(dummy_input)
        quantized_output = quantized_model(dummy_input)
    
    # Compute accuracy metric
    if model_type == 'tcn':
        # For TCN, compare COS predictions
        mse = torch.nn.functional.mse_loss(quantized_output, original_output)
        accuracy_score = 1.0 - mse.item()
    else:  # dqn
        # For DQN, compare Q-value predictions
        mse = torch.nn.functional.mse_loss(quantized_output, original_output)
        accuracy_score = 1.0 - mse.item()
    
    logger = setup_logger('export', level='INFO')
    logger.info(f"Model accuracy score: {accuracy_score:.4f}")
    
    return accuracy_score


def main():
    """Main export function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device)
    
    # Setup logging
    logger = setup_logger('export', level='INFO')
    logger.info(f"Exporting {args.model.upper()} model to INT8 TorchScript")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {device}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load original model
    logger.info("Loading original model...")
    original_model = load_model_from_checkpoint(args.model, args.checkpoint, config)
    
    # Log original model info
    original_size = sum(p.numel() for p in original_model.parameters())
    logger.info(f"Original model parameters: {original_size:,}")
    logger.info(f"Original model size: {original_size * 4 / (1024 * 1024):.2f} MB")
    
    # Calibrate and quantize model
    logger.info("Calibrating model for quantization...")
    quantized_model = calibrate_model(
        original_model, args.model, config, args.calibration_steps, device
    )
    
    # Export to TorchScript
    logger.info("Exporting to TorchScript...")
    traced_model = export_torchscript(
        quantized_model, args.model, config, args.output, device
    )
    
    # Verify accuracy
    logger.info("Verifying model accuracy...")
    accuracy = verify_model_accuracy(
        original_model, traced_model, args.model, config, device
    )
    
    # Run benchmark if requested
    if args.benchmark:
        logger.info("Running inference benchmark...")
        benchmark_results = benchmark_inference(
            traced_model, args.model, config, device
        )
        
        # Check performance targets
        if args.model == 'tcn':
            target_ms = config['benchmark']['targets']['tcn_inference']
        else:
            target_ms = config['benchmark']['targets']['dqn_inference']
        
        meets_target = benchmark_results['mean_ms'] < target_ms
        logger.info(f"Performance target ({target_ms}ms): {'✓' if meets_target else '✗'}")
    
    logger.info(f"Export completed successfully!")
    logger.info(f"Model saved to: {args.output}")
    logger.info(f"Accuracy score: {accuracy:.4f}")
    
    if args.benchmark:
        logger.info(f"Average inference time: {benchmark_results['mean_ms']:.2f}ms")


if __name__ == '__main__':
    main()
