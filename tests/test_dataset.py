"""
Test suite for dataset utilities.

This module tests the SPMA dataset implementation including data loading,
preprocessing, capacity constraints, and sequence generation.
"""

import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
import os

from utils.dataset import SPMADataset, create_data_loaders


class TestSPMADataset:
    """Test SPMADataset class."""
    
    def create_sample_data(self, num_samples: int = 100, cos_dim: int = 14):
        """Create sample data for testing."""
        data = {
            'timestamp': np.arange(num_samples) * 0.1
        }
        
        # Add COS columns
        for i in range(cos_dim):
            data[f'cos_{i}'] = np.random.uniform(0, 1, num_samples)
        
        # Add other columns
        data['backoff_action'] = np.random.randint(-2, 3, num_samples)
        data['reward'] = np.random.uniform(-1, 1, num_samples)
        data['capacity_used'] = np.random.uniform(0, 1, num_samples)
        
        return pd.DataFrame(data)
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data()
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            dataset = SPMADataset(
                tmp_path,
                window_size=32,
                cos_dim=14,
                n_prio=2
            )
            
            assert len(dataset) > 0
            assert dataset.cos_dim == 14
            assert dataset.n_prio == 2
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data(num_samples=100)
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            dataset = SPMADataset(
                tmp_path,
                window_size=32,
                cos_dim=14,
                n_prio=2
            )
            
            # Get first item
            item = dataset[0]
            
            assert 'features' in item
            assert 'actions' in item
            assert 'rewards' in item
            assert 'capacity_used' in item
            assert 'capacity_penalty' in item
            
            # Check shapes
            assert item['features'].shape == (32, 16)  # window_size, input_dim
            assert item['actions'].shape == (32,)
            assert item['rewards'].shape == (32,)
            assert item['capacity_used'].shape == (32,)
            assert isinstance(item['capacity_penalty'], torch.Tensor)
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_sequence_creation(self):
        """Test sliding window sequence creation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data(num_samples=100)
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            dataset = SPMADataset(
                tmp_path,
                window_size=32,
                sequence_stride=1,
                cos_dim=14,
                n_prio=2
            )
            
            # Check number of sequences
            expected_sequences = 100 - 32 + 1  # num_samples - window_size + 1
            assert len(dataset) == expected_sequences
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_sequence_stride(self):
        """Test dataset with different sequence strides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data(num_samples=100)
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            # Test with stride=2
            dataset = SPMADataset(
                tmp_path,
                window_size=32,
                sequence_stride=2,
                cos_dim=14,
                n_prio=2
            )
            
            # Check number of sequences
            expected_sequences = (100 - 32) // 2 + 1  # With stride=2
            assert len(dataset) == expected_sequences
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_capacity_penalty(self):
        """Test capacity penalty computation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data(num_samples=100)
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            dataset = SPMADataset(
                tmp_path,
                window_size=32,
                cos_dim=14,
                n_prio=2,
                max_capacity=1.0,
                capacity_penalty_weight=10.0
            )
            
            # Get item and check capacity penalty
            item = dataset[0]
            penalty = item['capacity_penalty']
            
            assert isinstance(penalty, torch.Tensor)
            assert penalty >= 0  # Penalty should be non-negative
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_normalization(self):
        """Test dataset normalization."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data(num_samples=100)
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            # Test with normalization
            dataset_normalized = SPMADataset(
                tmp_path,
                window_size=32,
                cos_dim=14,
                n_prio=2,
                normalize=True
            )
            
            # Test without normalization
            dataset_not_normalized = SPMADataset(
                tmp_path,
                window_size=32,
                cos_dim=14,
                n_prio=2,
                normalize=False
            )
            
            # Get items from both datasets
            item_norm = dataset_normalized[0]
            item_not_norm = dataset_not_normalized[0]
            
            # Features should be different due to normalization
            assert not torch.allclose(item_norm['features'], item_not_norm['features'])
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_tcn_batch(self):
        """Test TCN batch format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data(num_samples=100)
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            dataset = SPMADataset(
                tmp_path,
                window_size=32,
                cos_dim=14,
                n_prio=2
            )
            
            # Get TCN batch
            batch_indices = [0, 1, 2]
            inputs, targets = dataset.get_tcn_batch(batch_indices)
            
            assert inputs.shape == (3, 32, 16)  # batch_size, seq_len, input_dim
            assert targets.shape == (3, 32, 14)  # batch_size, seq_len, cos_dim
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_dqn_batch(self):
        """Test DQN batch format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data(num_samples=100)
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            dataset = SPMADataset(
                tmp_path,
                window_size=32,
                cos_dim=14,
                n_prio=2
            )
            
            # Get DQN batch
            batch_indices = [0, 1, 2]
            states, actions, rewards = dataset.get_dqn_batch(batch_indices)
            
            assert states.shape == (3, 10)  # batch_size, state_dim
            assert actions.shape == (3,)
            assert rewards.shape == (3,)
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_data_stats(self):
        """Test dataset statistics computation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data(num_samples=100)
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            dataset = SPMADataset(
                tmp_path,
                window_size=32,
                cos_dim=14,
                n_prio=2
            )
            
            stats = dataset.get_data_stats()
            
            assert 'num_sequences' in stats
            assert 'sequence_length' in stats
            assert 'total_samples' in stats
            assert 'cos_mean' in stats
            assert 'cos_std' in stats
            assert 'reward_mean' in stats
            assert 'reward_std' in stats
            assert 'capacity_mean' in stats
            assert 'capacity_std' in stats
            
            assert stats['num_sequences'] == len(dataset)
            assert stats['sequence_length'] == 32
            assert stats['total_samples'] == 100
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_edge_cases(self):
        """Test dataset edge cases."""
        # Test with very small dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            data = self.create_sample_data(num_samples=10)
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            dataset = SPMADataset(
                tmp_path,
                window_size=32,  # Larger than dataset
                cos_dim=14,
                n_prio=2
            )
            
            # Should handle gracefully
            assert len(dataset) == 0
            
        finally:
            os.unlink(tmp_path)
    
    def test_dataset_missing_columns(self):
        """Test dataset with missing required columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            # Create data with missing columns
            data = pd.DataFrame({
                'timestamp': np.arange(10),
                'cos_0': np.random.uniform(0, 1, 10)
                # Missing other required columns
            })
            data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            # Should raise ValueError for missing columns
            with pytest.raises(ValueError):
                dataset = SPMADataset(
                    tmp_path,
                    window_size=32,
                    cos_dim=14,
                    n_prio=2
                )
            
        finally:
            os.unlink(tmp_path)


class TestCreateDataLoaders:
    """Test create_data_loaders function."""
    
    def create_sample_data(self, num_samples: int = 100, cos_dim: int = 14):
        """Create sample data for testing."""
        data = {
            'timestamp': np.arange(num_samples) * 0.1
        }
        
        # Add COS columns
        for i in range(cos_dim):
            data[f'cos_{i}'] = np.random.uniform(0, 1, num_samples)
        
        # Add other columns
        data['backoff_action'] = np.random.randint(-2, 3, num_samples)
        data['reward'] = np.random.uniform(-1, 1, num_samples)
        data['capacity_used'] = np.random.uniform(0, 1, num_samples)
        
        return pd.DataFrame(data)
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        # Create temporary CSV files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as train_file:
            train_data = self.create_sample_data(num_samples=100)
            train_data.to_csv(train_file.name, index=False)
            train_path = train_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as val_file:
            val_data = self.create_sample_data(num_samples=50)
            val_data.to_csv(val_file.name, index=False)
            val_path = val_file.name
        
        try:
            config = {
                'tcn': {
                    'window_size': 32,
                    'input_dim': 16,
                    'n_prio': 2
                },
                'data': {
                    'sequence_stride': 1,
                    'normalize_inputs': True,
                    'max_capacity': 1.0,
                    'capacity_penalty_weight': 10.0
                },
                'training': {
                    'batch_size': 8,
                    'num_workers': 0,  # Use 0 for testing
                    'pin_memory': False
                }
            }
            
            train_loader, val_loader = create_data_loaders(
                train_path, val_path, config,
                batch_size=8,
                num_workers=0,
                pin_memory=False
            )
            
            # Check loaders
            assert train_loader is not None
            assert val_loader is not None
            
            # Check batch
            for batch in train_loader:
                assert 'features' in batch
                assert 'actions' in batch
                assert 'rewards' in batch
                assert 'capacity_used' in batch
                assert 'capacity_penalty' in batch
                
                # Check shapes
                assert batch['features'].shape[0] <= 8  # batch_size
                assert batch['features'].shape[1] == 32  # window_size
                assert batch['features'].shape[2] == 16  # input_dim
                
                break  # Only check first batch
            
        finally:
            os.unlink(train_path)
            os.unlink(val_path)


if __name__ == '__main__':
    pytest.main([__file__])
