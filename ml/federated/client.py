"""
Federated Learning Client Implementation for TGFL Market Scenario Simulator
Implements Flower client that wraps the transformer model for federated training
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np  # type: ignore[import-not-found]
from flwr.client import NumPyClient  # type: ignore[import-not-found]
from flwr.common import NDArrays, Scalar  # type: ignore[import-not-found]

from ml.models.transformer import create_tiny_transformer
from ml.data.loaders import MarketDataLoader, SyntheticDataGenerator
from ml.evaluation.metrics import ScenarioEvaluator


class TGFLClient(NumPyClient):
    """
    Flower federated learning client for transformer-based market scenario generation
    """
    
    def __init__(
        self,
        client_id: str,
        model_config: Dict,
        data_partition: Dict,
        device: torch.device = torch.device("cpu"),
        local_epochs: int = 1,
        batch_size: int = 16
    ):
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        
        # Initialize model (tiny transformer has fixed architecture)
        self.model = create_tiny_transformer().to(self.device)
        
        # Store data partition
        self.train_data = data_partition.get('train', [])
        self.test_data = data_partition.get('test', [])
        
        # Initialize optimizer and criterion
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=0.01
        )
        self.criterion = nn.MSELoss()
        
        # Data loaders
        self.loader = MarketDataLoader()
        
        print(f"Client {client_id} initialized with {self.model.count_parameters()} parameters")
        print(f"  Train sequences: {len(self.train_data)}")
        print(f"  Test sequences: {len(self.test_data)}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return model parameters as numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Update model parameters from numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data"""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Get training configuration
        epoch = int(config.get("epoch", 1))
        
        if not self.train_data:
            print(f"Client {self.client_id}: No training data available")
            return self.get_parameters({}), 0, {"train_loss": 0.0}
        
        # Convert data to tensors
        train_sequences = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in self.train_data])
        train_targets = train_sequences[:, 1:]  # Next-token prediction
        train_inputs = train_sequences[:, :-1]
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for local_epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(train_loader)
        
        avg_loss = total_loss / self.local_epochs
        
        print(f"Client {self.client_id} - Round {epoch}: Local training loss = {avg_loss:.6f}")
        
        return self.get_parameters({}), len(train_inputs), {"train_loss": avg_loss}
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local test data"""
        # Set parameters from server
        self.set_parameters(parameters)
        
        if not self.test_data:
            print(f"Client {self.client_id}: No test data available")
            return 0.0, 0, {"test_loss": 0.0}
        
        # Convert data to tensors
        test_sequences = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in self.test_data])
        test_targets = test_sequences[:, 1:]
        test_inputs = test_sequences[:, :-1]
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
        test_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # Evaluation loop
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in test_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
                
                total_loss += loss.item() * len(batch_inputs)
                num_samples += len(batch_inputs)
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        print(f"Client {self.client_id}: Test loss = {avg_loss:.6f}")
        
        return avg_loss, num_samples, {"test_loss": avg_loss}


def create_federated_clients(
    num_clients: int,
    model_config: Dict,
    data_partitions: List[Dict],
    device: torch.device = torch.device("cpu"),
    **client_kwargs
) -> List[TGFLClient]:
    """
    Create multiple federated clients with partitioned data
    
    Args:
        num_clients: Number of clients to create
        model_config: Model configuration dictionary
        data_partitions: List of data partitions for each client
        device: Computing device (CPU/GPU)
        **client_kwargs: Additional client configuration
    
    Returns:
        List of initialized TGFLClient instances
    """
    clients = []
    
    for i in range(num_clients):
        client_id = f"client_{i+1}"
        partition = data_partitions[i] if i < len(data_partitions) else {'train': [], 'test': []}
        
        client = TGFLClient(
            client_id=client_id,
            model_config=model_config,
            data_partition=partition,
            device=device,
            **client_kwargs
        )
        clients.append(client)
    
    return clients


# Example usage and testing
if __name__ == "__main__":
    print("Testing TGFL Federated Client...")
    
    # Model configuration
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'sequence_length': 256,
        'dropout': 0.1
    }
    
    # Generate synthetic data for testing
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_regime_data('normal', length=1000)
    
    loader = MarketDataLoader()
    sequences, targets = loader.create_sequences(data, sequence_length=50)
    
    # Create mock data partitions
    mid_point = len(sequences) // 2
    partitions = [
        {
            'train': sequences[:mid_point].tolist(),
            'test': sequences[mid_point:mid_point+10].tolist()
        },
        {
            'train': sequences[mid_point:].tolist(), 
            'test': sequences[-10:].tolist()
        }
    ]
    
    # Create clients
    clients = create_federated_clients(
        num_clients=2,
        model_config=config,
        data_partitions=partitions,
        local_epochs=1,
        batch_size=8
    )
    
    print(f"\nCreated {len(clients)} federated clients")
    print("Federated client implementation ready!")