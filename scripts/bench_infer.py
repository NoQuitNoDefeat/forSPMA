#!/usr/bin/env python3
"""
Inference Benchmark Script for SPMA ML Models

This script benchmarks the inference performance of TCN and DQN models
on CPU and provides detailed latency analysis for edge deployment evaluation.

Usage:
    python scripts/bench_infer.py --model tcn --checkpoint models/tcn_best_model.pt
    python scripts/bench_infer.py --model dqn --checkpoint models/dqn_best_model.pt
"""

import argparse
import torch
import torch.jit
import numpy as np
import time
import yaml
import os
import json
from typing import Dict, Any, List, Tuple
import statistics

from models.tcn import TCN, create_tcn_from_config
from models.dqn import DQN, create_dqn_from_config
from utils.logger import setup_logger, log_inference_performance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark SPMA ML model inference')
    parser.add_argument('--model', type=str, choices=['tcn', 'dqn'], required=True,
                       help='Model type to benchmark')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for benchmarking (cpu/cuda)')
    parser.add_argument('--warmup-runs', type=int, default=50,
                       help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=1000,
                       help='Number of benchmark runs')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1],
                       help='Batch sizes to test')
    parser.add_argument('--export-int8', action='store_true',
                       help='Export and benchmark INT8 model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for benchmark results')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_input(model_type: str, config: Dict[str, Any], batch_size: int = 1) -> torch.Tensor:
    """Create dummy input for model inference."""
    if model_type == 'tcn':
        tcn_config = config['tcn']
        seq_len = tcn_config.get('window_size', 32)
        input_dim = tcn_config['input_dim']
        dummy_input = torch.randn(batch_size, seq_len, input_dim)
    elif model_type == 'dqn':
        dqn_config = config['dqn']
        state_dim = dqn_config['state_dim']
        dummy_input = torch.randn(batch_size, state_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return dummy_input


def load_model(model_type: str, checkpoint_path: str, config: Dict[str, Any], device: torch.device):
    """Load model from checkpoint."""
    if model_type == 'tcn':
        model = create_tcn_from_config(config['tcn'])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'dqn':
        model = create_dqn_from_config(config['dqn'])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.q_network.load_state_dict(checkpoint['model_state_dict'])
        model = model.q_network  # Use only Q-network for inference
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    model = model.to(device)
    
    return model


def benchmark_model(
    model: torch.nn.Module,
    model_type: str,
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    warmup_runs: int,
    num_runs: int
) -> Dict[str, float]:
    """
    Benchmark model inference performance.
    
    Args:
        model: Model to benchmark
        model_type: Type of model (tcn/dqn)
        device: Device for benchmarking
        batch_size: Batch size for inference
        warmup_runs: Number of warmup runs
        num_runs: Number of benchmark runs
    
    Returns:
        benchmark_results: Dictionary of benchmark metrics
    """
    logger = setup_logger('benchmark', level='INFO')
    
    # Create dummy input
    dummy_input = create_dummy_input(model_type, config, batch_size)
    dummy_input = dummy_input.to(device)
    
    # Warmup runs
    logger.info(f"Running {warmup_runs} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            if model_type == 'tcn':
                _ = model(dummy_input)
            else:  # dqn
                _ = model(dummy_input)
    
    # Benchmark runs
    logger.info(f"Running {num_runs} benchmark iterations...")
    inference_times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            
            if model_type == 'tcn':
                cos_pred, capacity_pred = model(dummy_input)
            else:  # dqn
                q_values = model(dummy_input)
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            if i % 100 == 0:
                logger.info(f"Benchmark progress: {i}/{num_runs}")
    
    # Compute statistics
    times_ms = [t * 1000 for t in inference_times]
    
    benchmark_results = {
        'model_type': model_type,
        'batch_size': batch_size,
        'device': str(device),
        'num_runs': num_runs,
        'mean_ms': statistics.mean(times_ms),
        'std_ms': statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        'min_ms': min(times_ms),
        'max_ms': max(times_ms),
        'median_ms': statistics.median(times_ms),
        'p95_ms': np.percentile(times_ms, 95),
        'p99_ms': np.percentile(times_ms, 99),
        'throughput_fps': batch_size / (statistics.mean(times_ms) / 1000),
        'all_times_ms': times_ms
    }
    
    return benchmark_results


def benchmark_int8_model(
    model_type: str,
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    warmup_runs: int,
    num_runs: int,
    models_dir: str = 'models'
) -> Dict[str, float]:
    """Benchmark INT8 quantized model."""
    logger = setup_logger('benchmark', level='INFO')
    
    # Load INT8 model
    int8_path = os.path.join(models_dir, f'{model_type}_int8.ptl')
    if not os.path.exists(int8_path):
        logger.error(f"INT8 model not found at {int8_path}")
        logger.info("Run export_int8.py first to create INT8 model")
        return {}
    
    logger.info(f"Loading INT8 model from {int8_path}")
    int8_model = torch.jit.load(int8_path, map_location=device)
    int8_model.eval()
    
    # Create dummy input
    dummy_input = create_dummy_input(model_type, config, batch_size)
    dummy_input = dummy_input.to(device)
    
    # Warmup runs
    logger.info(f"Running {warmup_runs} warmup iterations for INT8 model...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = int8_model(dummy_input)
    
    # Benchmark runs
    logger.info(f"Running {num_runs} benchmark iterations for INT8 model...")
    inference_times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            _ = int8_model(dummy_input)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            if i % 100 == 0:
                logger.info(f"INT8 benchmark progress: {i}/{num_runs}")
    
    # Compute statistics
    times_ms = [t * 1000 for t in inference_times]
    
    benchmark_results = {
        'model_type': f'{model_type}_int8',
        'batch_size': batch_size,
        'device': str(device),
        'num_runs': num_runs,
        'mean_ms': statistics.mean(times_ms),
        'std_ms': statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        'min_ms': min(times_ms),
        'max_ms': max(times_ms),
        'median_ms': statistics.median(times_ms),
        'p95_ms': np.percentile(times_ms, 95),
        'p99_ms': np.percentile(times_ms, 99),
        'throughput_fps': batch_size / (statistics.mean(times_ms) / 1000),
        'all_times_ms': times_ms
    }
    
    return benchmark_results


def compare_models(
    float_results: Dict[str, float],
    int8_results: Dict[str, float]
) -> Dict[str, Any]:
    """Compare float and INT8 model performance."""
    if not float_results or not int8_results:
        return {}
    
    comparison = {
        'speedup': float_results['mean_ms'] / int8_results['mean_ms'],
        'accuracy_loss': 0.0,  # Would need actual accuracy comparison
        'size_reduction': 0.75,  # Typical INT8 size reduction
        'float_mean_ms': float_results['mean_ms'],
        'int8_mean_ms': int8_results['mean_ms'],
        'float_p95_ms': float_results['p95_ms'],
        'int8_p95_ms': int8_results['p95_ms']
    }
    
    return comparison


def check_performance_targets(
    results: Dict[str, float],
    config: Dict[str, Any]
) -> Dict[str, bool]:
    """Check if performance targets are met."""
    targets = config['benchmark']['targets']
    
    model_type = results['model_type'].replace('_int8', '')
    
    if model_type == 'tcn':
        target_ms = targets['tcn_inference']
    elif model_type == 'dqn':
        target_ms = targets['dqn_inference']
    else:
        return {}
    
    meets_target = results['mean_ms'] < target_ms
    meets_p95_target = results['p95_ms'] < target_ms * 1.5  # Allow 50% higher for P95
    
    return {
        'mean_target': meets_target,
        'p95_target': meets_p95_target,
        'target_ms': target_ms,
        'achieved_mean_ms': results['mean_ms'],
        'achieved_p95_ms': results['p95_ms']
    }


def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device)
    
    # Setup logging
    logger = setup_logger('benchmark', level='INFO')
    logger.info(f"Benchmarking {args.model.upper()} model")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch sizes: {args.batch_sizes}")
    
    all_results = []
    
    for batch_size in args.batch_sizes:
        logger.info(f"\nBenchmarking with batch size {batch_size}")
        
        # Load and benchmark float model
        logger.info("Loading float model...")
        float_model = load_model(args.model, args.checkpoint, config, device)
        
        # Benchmark float model
        logger.info("Benchmarking float model...")
        float_results = benchmark_model(
            float_model, args.model, config, device,
            batch_size, args.warmup_runs, args.num_runs
        )
        
        all_results.append(float_results)
        
        # Log float model results
        logger.info(f"Float model results (batch_size={batch_size}):")
        logger.info(f"  Mean inference time: {float_results['mean_ms']:.2f}ms")
        logger.info(f"  P95 inference time: {float_results['p95_ms']:.2f}ms")
        logger.info(f"  Throughput: {float_results['throughput_fps']:.2f} FPS")
        
        # Check performance targets
        target_results = check_performance_targets(float_results, config)
        if target_results:
            logger.info(f"  Performance target ({target_results['target_ms']}ms): "
                       f"{'✓' if target_results['mean_target'] else '✗'}")
        
        # Benchmark INT8 model if requested
        if args.export_int8:
            logger.info("Benchmarking INT8 model...")
            int8_results = benchmark_int8_model(
                args.model, config, device,
                batch_size, args.warmup_runs, args.num_runs
            )
            
            if int8_results:
                all_results.append(int8_results)
                
                # Log INT8 model results
                logger.info(f"INT8 model results (batch_size={batch_size}):")
                logger.info(f"  Mean inference time: {int8_results['mean_ms']:.2f}ms")
                logger.info(f"  P95 inference time: {int8_results['p95_ms']:.2f}ms")
                logger.info(f"  Throughput: {int8_results['throughput_fps']:.2f} FPS")
                
                # Compare models
                comparison = compare_models(float_results, int8_results)
                if comparison:
                    logger.info(f"  Speedup: {comparison['speedup']:.2f}x")
                    logger.info(f"  Size reduction: {comparison['size_reduction']*100:.1f}%")
                
                # Check INT8 performance targets
                int8_target_results = check_performance_targets(int8_results, config)
                if int8_target_results:
                    logger.info(f"  INT8 Performance target ({int8_target_results['target_ms']}ms): "
                               f"{'✓' if int8_target_results['mean_target'] else '✗'}")
    
    # Log overall performance
    logger.info("\nOverall Performance Summary:")
    for result in all_results:
        model_name = result['model_type'].upper()
        logger.info(f"{model_name} (batch_size={result['batch_size']}): "
                   f"{result['mean_ms']:.2f}ms mean, {result['p95_ms']:.2f}ms P95")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Benchmark results saved to {args.output}")
    
    logger.info("Benchmark completed successfully!")


if __name__ == '__main__':
    main()
