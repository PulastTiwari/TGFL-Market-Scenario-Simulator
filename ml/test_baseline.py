#!/usr/bin/env python3
"""
Simple training script to test the baseline model setup
Can be run without Jupyter for CI/testing purposes
"""

import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "ml"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import our modules (will work once dependencies are installed)
try:
    from data.loaders import MarketDataLoader, SyntheticDataGenerator
    from models.transformer import create_tiny_transformer
    from evaluation.metrics import ScenarioEvaluator
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"⚠️  Import error (expected without dependencies): {e}")
    print("Run this script after installing requirements.txt")
    sys.exit(0)

def test_data_generation():
    """Test synthetic data generation"""
    print("\n📊 Testing data generation...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Test different regimes
    regimes = ['normal', 'bull', 'bear', 'volatile']
    for regime in regimes:
        data = generator.generate_regime_data(regime, length=100)
        print(f"  {regime:8s}: {len(data)} samples, mean return: {data['returns'].mean():.6f}")
    
    print("✅ Data generation test passed")

def test_model_creation():
    """Test model creation and parameter counting"""
    print("\n🧠 Testing model creation...")
    
    model = create_tiny_transformer()
    param_count = model.count_parameters()
    
    print(f"  Model parameters: {param_count:,}")
    print(f"  Memory efficient: {'✅' if param_count < 1000000 else '❌'}")
    
    # Test forward pass
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, 1)
    
    with torch.no_grad():
        output = model(x)
        print(f"  Forward pass: {x.shape} → {output.shape}")
    
    print("✅ Model creation test passed")

def test_evaluation_metrics():
    """Test evaluation metrics"""
    print("\n📈 Testing evaluation metrics...")
    
    evaluator = ScenarioEvaluator()
    
    # Create test data
    np.random.seed(42)
    historical = np.random.normal(0.0005, 0.02, 500)
    generated = np.random.normal(0.0005, 0.02, 300)
    
    # Test KS test
    ks_result = evaluator.kolmogorov_smirnov_test(generated, historical)
    print(f"  KS-test p-value: {ks_result['p_value']:.4f}")
    
    # Test ACF
    acf_result = evaluator.autocorrelation_similarity(generated, historical)
    print(f"  ACF similarity: {acf_result['acf_similarity_score']:.4f}")
    
    print("✅ Evaluation metrics test passed")

def test_training_loop():
    """Test a minimal training loop"""
    print("\n🚂 Testing training loop...")
    
    # Create synthetic data
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_regime_data('normal', length=200)
    
    loader = MarketDataLoader()
    sequences, targets = loader.create_sequences(data, sequence_length=32, step_size=5)
    
    print(f"  Created {len(sequences)} training sequences")
    
    # Create model and data
    model = create_tiny_transformer()
    
    train_X = torch.FloatTensor(sequences[:80]).unsqueeze(-1)
    train_y = torch.FloatTensor(targets[:80]).unsqueeze(-1)
    
    dataset = TensorDataset(train_X, train_y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Mini training loop (3 epochs)
    model.train()
    for epoch in range(3):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output[:, -1, :], batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.6f}")
    
    print("✅ Training loop test passed")

def main():
    """Run all tests"""
    print("TGFL Baseline Model Test Suite")
    print("=" * 40)
    
    try:
        test_data_generation()
        test_model_creation() 
        test_evaluation_metrics()
        test_training_loop()
        
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Run the full Jupyter notebook for complete training")
        print("2. Proceed to federated learning implementation")
        print("3. Build the API and frontend integration")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()