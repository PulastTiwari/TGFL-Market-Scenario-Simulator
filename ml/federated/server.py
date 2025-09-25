"""
Federated Learning Server Implementation for TGFL Market Scenario Simulator
Implements Flower server with FedAvg aggregation strategy
"""

import torch  # type: ignore
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg
from flwr.common import NDArrays, Parameters, Scalar, FitRes, EvaluateRes
from flwr.common.typing import GetParametersIns
import pickle
from pathlib import Path

from ml.models.transformer import create_tiny_transformer
from ml.evaluation.metrics import ScenarioEvaluator


class TGFLStrategy(FedAvg):
    """
    Custom federated averaging strategy for TGFL transformer training
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        save_path: str = "data/models/federated_model.pth"
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters
        )
        
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Track training history
        self.round_losses = []
        self.round_accuracies = []
        
        print(f"TGFL Federated Strategy initialized")
        print(f"  Min fit clients: {min_fit_clients}")
        print(f"  Min evaluate clients: {min_evaluate_clients}")
        print(f"  Model save path: {save_path}")
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training"""
        config = {
            "epoch": server_round,
            "local_epochs": 1,
            "batch_size": 16
        }
        
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        
        # Update config for all clients
        for client_proxy, fit_instruction in fit_ins:
            fit_instruction.config.update(config)
        
        return fit_ins
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, FitRes]],
        failures: List[Any]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results from clients"""
        
        if not results:
            return None, {}
        
        # Call parent aggregation (FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Save aggregated model
            self._save_model(aggregated_parameters, server_round)
            
            # Calculate average training loss
            train_losses = [res.metrics.get("train_loss", 0.0) for _, res in results]
            avg_train_loss = np.mean(train_losses) if train_losses else 0.0
            
            self.round_losses.append(avg_train_loss)
            
            print(f"Round {server_round} aggregation:")
            print(f"  Participants: {len(results)}")
            print(f"  Average train loss: {avg_train_loss:.6f}")
            print(f"  Failures: {len(failures)}")
            
            aggregated_metrics["avg_train_loss"] = avg_train_loss
            aggregated_metrics["participants"] = len(results)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, EvaluateRes]],
        failures: List[Any]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients"""
        
        if not results:
            return None, {}
        
        # Calculate weighted average loss
        total_examples = sum([res.num_examples for _, res in results])
        if total_examples == 0:
            return None, {}
        
        weighted_losses = [
            res.loss * res.num_examples for _, res in results
        ]
        avg_loss = sum(weighted_losses) / total_examples
        
        # Collect test losses from all clients
        test_losses = [res.metrics.get("test_loss", res.loss) for _, res in results]
        
        print(f"Round {server_round} evaluation:")
        print(f"  Average test loss: {avg_loss:.6f}")
        print(f"  Participants: {len(results)}")
        
        metrics = {
            "avg_test_loss": avg_loss,
            "participants": len(results),
            "min_test_loss": float(np.min(test_losses)),
            "max_test_loss": float(np.max(test_losses))
        }
        
        return avg_loss, metrics
    
    def _save_model(self, parameters: Parameters, round_num: int):
        """Save the aggregated model parameters"""
        try:
            # Convert parameters back to PyTorch model
            model = create_tiny_transformer()
            
            # Convert NDArrays back to tensors
            params_dict = {}
            param_names = list(model.state_dict().keys())
            
            for i, param_array in enumerate(parameters.tensors):
                if i < len(param_names):
                    params_dict[param_names[i]] = torch.from_numpy(param_array)
            
            model.load_state_dict(params_dict)
            
            # Save model checkpoint
            checkpoint = {
                'round': round_num,
                'model_state_dict': model.state_dict(),
                'round_losses': self.round_losses,
                'parameters': len(list(model.parameters()))
            }
            
            torch.save(checkpoint, self.save_path)
            
            print(f"  → Model saved to {self.save_path}")
            
        except Exception as e:
            print(f"  → Failed to save model: {e}")


def get_initial_parameters() -> Parameters:
    """Get initial model parameters for federated training"""
    model = create_tiny_transformer()
    
    # Convert model parameters to NDArrays
    ndarrays = [param.detach().cpu().numpy() for param in model.parameters()]
    
    return Parameters(tensors=ndarrays, tensor_type="numpy.ndarray")


def start_federated_server(
    num_rounds: int = 10,
    min_clients: int = 2,
    server_address: str = "[::]:8080",
    save_path: str = "data/models/federated_model.pth"
) -> None:
    """
    Start the federated learning server
    
    Args:
        num_rounds: Number of federated training rounds
        min_clients: Minimum number of clients required
        server_address: Server address and port
        save_path: Path to save the final model
    """
    
    # Initialize strategy
    strategy = TGFLStrategy(
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=get_initial_parameters(),
        save_path=save_path
    )
    
    # Server configuration
    config = ServerConfig(num_rounds=num_rounds)
    
    print(f"Starting TGFL Federated Server...")
    print(f"  Address: {server_address}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Min clients: {min_clients}")
    print(f"  Model save path: {save_path}")
    print()
    
    # Start server
    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy
    )


def evaluate_federated_model(
    model_path: str = "data/models/federated_model.pth",
    test_data: Optional[List] = None
) -> Dict:
    """
    Evaluate the trained federated model
    
    Args:
        model_path: Path to the saved federated model
        test_data: Test data for evaluation (optional)
    
    Returns:
        Evaluation results dictionary
    """
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = create_tiny_transformer()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        results = {
            'model_path': model_path,
            'training_rounds': checkpoint.get('round', 0),
            'parameters': checkpoint.get('parameters', 0),
            'round_losses': checkpoint.get('round_losses', [])
        }
        
        if test_data and len(test_data) > 0:
            # Run evaluation on test data
            criterion = torch.nn.MSELoss()
            total_loss = 0.0
            num_samples = 0
            
            with torch.no_grad():
                for sequence in test_data:
                    if len(sequence) > 1:
                        inputs = torch.tensor(sequence[:-1], dtype=torch.float32).unsqueeze(0)
                        targets = torch.tensor(sequence[1:], dtype=torch.float32).unsqueeze(0)
                        
                        outputs = model(inputs.unsqueeze(-1))
                        loss = criterion(outputs, targets.unsqueeze(-1))
                        
                        total_loss += loss.item()
                        num_samples += 1
            
            if num_samples > 0:
                results['test_loss'] = total_loss / num_samples
                results['test_samples'] = num_samples
        
        print(f"Federated model evaluation complete:")
        print(f"  Training rounds: {results.get('training_rounds', 'Unknown')}")
        print(f"  Parameters: {results.get('parameters', 'Unknown')}")
        if 'test_loss' in results:
            print(f"  Test loss: {results['test_loss']:.6f}")
        
        return results
        
    except Exception as e:
        print(f"Failed to evaluate federated model: {e}")
        return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    print("TGFL Federated Server Test")
    print("=" * 40)
    
    # Test initial parameters generation
    initial_params = get_initial_parameters()
    print(f"Initial parameters: {len(initial_params.tensors)} tensors")
    
    # Note: To actually start the server, uncomment the line below
    # start_federated_server(num_rounds=5, min_clients=2)
    
    print("\nFederated server implementation ready!")
    print("To start training:")
    print("1. Run this script to start the server")
    print("2. Run client instances to connect and train")