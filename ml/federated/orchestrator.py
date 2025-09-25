"""
Federated Learning Orchestrator for TGFL Market Scenario Simulator
Coordinates federated training simulation with multiple clients and data partitioning
"""

import torch
import numpy as np
import multiprocessing
import threading
import time
import subprocess
import signal
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

from ml.federated.client import create_federated_clients, TGFLClient
from ml.federated.server import start_federated_server
from ml.data.loaders import SyntheticDataGenerator, create_federated_partitions
from ml.models.transformer import create_tiny_transformer


class FederatedOrchestrator:
    """
    Orchestrates federated learning simulation with multiple clients and server
    """
    
    def __init__(
        self,
        num_clients: int = 4,
        server_address: str = "localhost:8080",
        data_path: Optional[str] = None,
        results_path: str = "data/results/federated",
        partition_strategy: str = "iid"
    ):
        self.num_clients = num_clients
        self.server_address = server_address
        self.data_path = data_path
        self.results_path = Path(results_path)
        self.partition_strategy = partition_strategy
        
        # Create results directory
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Process management
        self.server_process = None
        self.client_processes = []
        self.simulation_running = False
        
        # Training configuration
        self.config = {
            'num_rounds': 10,
            'local_epochs': 1,
            'batch_size': 16,
            'learning_rate': 0.001,
            'min_clients': 2
        }
        
        print(f"Federated Orchestrator initialized:")
        print(f"  Clients: {num_clients}")
        print(f"  Server: {server_address}")
        print(f"  Partition strategy: {partition_strategy}")
        print(f"  Results path: {results_path}")
    
    def prepare_data(self, total_samples: int = 1000) -> List[List]:
        """
        Generate and partition data for federated clients
        
        Args:
            total_samples: Total number of market scenarios to generate
            
        Returns:
            List of data partitions for each client
        """
        print(f"Preparing data for {self.num_clients} clients...")
        
        # Generate synthetic market data
        generator = SyntheticDataGenerator()
        
        if self.data_path and Path(self.data_path).exists():
            # Load existing data
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                market_sequences = data.get('sequences', [])
        else:
            # Generate mixed regime data for diverse scenarios
            regimes = ['normal', 'bull', 'bear', 'volatile']
            sequences_per_regime = total_samples // len(regimes)
            
            market_sequences = []
            for regime in regimes:
                df = generator.generate_regime_data(regime, length=sequences_per_regime)
                # Convert to sequences (simple time series) using sliding windows
                arr = df['returns'].to_numpy()
                regime_sequences = []
                if len(arr) >= 10:
                    for i in range(len(arr) - 9):
                        window = arr[i:i+10].tolist()
                        regime_sequences.append(window)
                market_sequences.extend(regime_sequences[:sequences_per_regime])
            
            # Ensure we have enough sequences
            while len(market_sequences) < total_samples:
                df = generator.generate_regime_data('normal', length=100)
                arr = df['returns'].to_numpy()
                extra_sequences = []
                if len(arr) >= 10:
                    for i in range(len(arr) - 9):
                        extra_sequences.append(arr[i:i+10].tolist())
                market_sequences.extend(extra_sequences[:total_samples - len(market_sequences)])
            
            # Save generated data
            data_file = self.results_path / "market_data.json"
            with open(data_file, 'w') as f:
                json.dump({
                    'sequences': market_sequences[:total_samples],
                    'num_clients': self.num_clients,
                    'total_samples': len(market_sequences[:total_samples])
                }, f)
            print(f"  - Generated data saved to {data_file}")
        
        # Simple partition data across clients
        sequences = market_sequences[:total_samples]
        partition_size = len(sequences) // self.num_clients
        
        client_partitions = []
        for i in range(self.num_clients):
            start_idx = i * partition_size
            if i == self.num_clients - 1:  # Last client gets remainder
                end_idx = len(sequences)
            else:
                end_idx = (i + 1) * partition_size

            client_partitions.append(sequences[start_idx:end_idx])

        print(f"  - Data partitioned: {[len(p) for p in client_partitions]} samples per client")

        # Save partition information
        partition_file = self.results_path / "data_partitions.json"
        with open(partition_file, 'w') as f:
            json.dump({
                'strategy': self.partition_strategy,
                'num_clients': self.num_clients,
                'partition_sizes': [len(p) for p in client_partitions],
                'total_samples': sum(len(p) for p in client_partitions)
            }, f)

        return client_partitions
    
    def start_server(self, num_rounds: int = 10) -> bool:
        """
        Start the federated server in a separate process
        
        Args:
            num_rounds: Number of federated training rounds
            
        Returns:
            True if server started successfully
        """
        try:
            print(f"Starting federated server (rounds: {num_rounds})...")
            
            # Create server script
            server_script = f"""
import sys
sys.path.append('.')
from ml.federated.server import start_federated_server

if __name__ == "__main__":
    start_federated_server(
        num_rounds={num_rounds},
        min_clients={self.config['min_clients']},
        server_address="{self.server_address}",
        save_path="{self.results_path}/federated_model.pth"
    )
"""
            
            script_path = self.results_path / "start_server.py"
            with open(script_path, 'w') as f:
                f.write(server_script)
            
            # Start server process
            self.server_process = subprocess.Popen([
                "python", str(script_path)
            ], cwd=Path.cwd())
            
            # Wait for server to start
            time.sleep(3)
            
            if self.server_process.poll() is None:
                print(f"  - Server started (PID: {self.server_process.pid})")
                return True
            else:
                print("  - Server failed to start")
                return False
                
        except Exception as e:
            print(f"  → Server startup error: {e}")
            return False
    
    def start_clients(self, client_partitions: List[List]) -> bool:
        """
        Start federated clients in separate processes
        
        Args:
            client_partitions: Data partitions for each client
            
        Returns:
            True if all clients started successfully
        """
        try:
            print(f"Starting {self.num_clients} federated clients...")
            
            for client_id in range(self.num_clients):
                # Save client data partition
                partition_file = self.results_path / f"client_{client_id}_data.json"
                with open(partition_file, 'w') as f:
                    json.dump({
                        'client_id': client_id,
                        'data': client_partitions[client_id]
                    }, f)
                
                # Create client script
                client_script = f"""
import sys
sys.path.append('.')
import json
from pathlib import Path
from ml.federated.client import TGFLClient
from flwr.client import start_numpy_client

def load_client_data(client_id):
    base_dir = Path(__file__).resolve().parent
    with open(base_dir / f"client_{{client_id}}_data.json", 'r') as f:
        return json.load(f)['data']

if __name__ == "__main__":
    client_id = {client_id}
    data = load_client_data(client_id)
    
    # Create client (model is built internally by TGFLClient)
    model_config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'sequence_length': 256,
        'dropout': 0.1,
    }
    client = TGFLClient(
        client_id=f"client_{{client_id}}",
        model_config=model_config,
        data_partition={
            'train': data,
            'test': data[: min(10, len(data))]
        },
        local_epochs=1,
        batch_size=16,
    )
    
    # Start client
    start_numpy_client(
        server_address="{self.server_address}",
        client=client
    )
"""
                
                script_path = self.results_path / f"start_client_{client_id}.py"
                with open(script_path, 'w') as f:
                    f.write(client_script)
                
                # Start client process
                client_process = subprocess.Popen([
                    "python", str(script_path)
                ], cwd=Path.cwd())
                
                self.client_processes.append(client_process)
                print(f"  - Client {client_id} started (PID: {client_process.pid})")
                
                # Small delay between client starts
                time.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"  → Client startup error: {e}")
            return False
    
    def run_simulation(
        self,
        num_rounds: int = 10,
        total_samples: int = 1000,
        wait_for_completion: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete federated learning simulation
        
        Args:
            num_rounds: Number of federated training rounds
            total_samples: Total number of samples to generate
            wait_for_completion: Whether to wait for training completion
            
        Returns:
            Simulation results dictionary
        """
        print("=" * 60)
        print("TGFL Federated Learning Simulation")
        print("=" * 60)
        
        simulation_start = time.time()
        self.simulation_running = True
        
        try:
            # Step 1: Prepare data
            client_partitions = self.prepare_data(total_samples)
            
            # Step 2: Start server
            if not self.start_server(num_rounds):
                raise Exception("Failed to start federated server")
            
            # Step 3: Start clients
            if not self.start_clients(client_partitions):
                raise Exception("Failed to start federated clients")
            
            # Step 4: Monitor simulation
            if wait_for_completion:
                print(f"\nMonitoring federated training ({num_rounds} rounds)...")
                self._monitor_simulation(num_rounds)
            
            simulation_time = time.time() - simulation_start
            
            # Collect results
            results = {
                'simulation_time': simulation_time,
                'num_clients': self.num_clients,
                'num_rounds': num_rounds,
                'total_samples': total_samples,
                'partition_strategy': self.partition_strategy,
                'server_address': self.server_address,
                'results_path': str(self.results_path),
                'success': True
            }
            
            # Save simulation results
            results_file = self.results_path / "simulation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nSimulation completed successfully!")
            print(f"  Total time: {simulation_time:.1f} seconds")
            print(f"  Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            print(f"\nSimulation failed: {e}")
            return {
                'error': str(e),
                'success': False,
                'simulation_time': time.time() - simulation_start
            }
        
        finally:
            self.cleanup()
    
    def _monitor_simulation(self, num_rounds: int, check_interval: int = 10):
        """Monitor the federated simulation progress"""
        
        start_time = time.time()
        max_wait_time = num_rounds * 60  # 1 minute per round max
        
        while time.time() - start_time < max_wait_time:
            # Check if server is still running
            if self.server_process and self.server_process.poll() is not None:
                print("  - Server process completed")
                break

            # Check client processes
            active_clients = sum(1 for p in self.client_processes if p.poll() is None)

            elapsed = time.time() - start_time
            print(f"  - Elapsed: {elapsed:.0f}s, Active clients: {active_clients}")

            time.sleep(check_interval)

        print(f"  - Monitoring complete after {time.time() - start_time:.1f} seconds")
    
    def cleanup(self):
        """Clean up processes and resources"""
        print("\nCleaning up simulation processes...")
        
        self.simulation_running = False
        
        # Stop client processes
        for i, client_process in enumerate(self.client_processes):
            if client_process.poll() is None:
                try:
                    client_process.terminate()
                    client_process.wait(timeout=5)
                    print(f"  - Client {i} stopped")
                except:
                    client_process.kill()
        
        # Stop server process
        if self.server_process and self.server_process.poll() is None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print(f"  - Server stopped")
            except:
                self.server_process.kill()
        
        self.client_processes.clear()
        self.server_process = None
        
    print("  - Cleanup complete")
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        
        server_running = (self.server_process is not None and 
                         self.server_process.poll() is None)
        
        active_clients = sum(1 for p in self.client_processes if p.poll() is None)
        
        return {
            'simulation_running': self.simulation_running,
            'server_running': server_running,
            'active_clients': active_clients,
            'total_clients': len(self.client_processes),
            'server_address': self.server_address
        }


# Utility functions for federated simulation

def run_quick_simulation(
    num_clients: int = 3,
    num_rounds: int = 5,
    samples_per_client: int = 100
) -> Dict[str, Any]:
    """
    Run a quick federated learning simulation for testing
    
    Args:
        num_clients: Number of federated clients
        num_rounds: Number of training rounds
        samples_per_client: Samples per client
        
    Returns:
        Simulation results
    """
    total_samples = num_clients * samples_per_client
    
    orchestrator = FederatedOrchestrator(
        num_clients=num_clients,
        results_path="data/results/federated_quick"
    )
    
    return orchestrator.run_simulation(
        num_rounds=num_rounds,
        total_samples=total_samples,
        wait_for_completion=True
    )


def run_comprehensive_simulation() -> Dict[str, Any]:
    """
    Run a comprehensive federated learning simulation
    
    Returns:
        Simulation results
    """
    orchestrator = FederatedOrchestrator(
        num_clients=5,
        results_path="data/results/federated_comprehensive",
        partition_strategy="non_iid"
    )
    
    return orchestrator.run_simulation(
        num_rounds=20,
        total_samples=2000,
        wait_for_completion=True
    )


# Example usage
if __name__ == "__main__":
    print("TGFL Federated Orchestrator Test")
    print("=" * 40)
    
    # Run quick test simulation
    print("\nRunning quick test simulation...")
    results = run_quick_simulation(
        num_clients=2,
        num_rounds=3,
        samples_per_client=50
    )
    
    if results['success']:
        print("[SUCCESS] Quick simulation completed successfully!")
    else:
        print(f"[FAIL] Quick simulation failed: {results.get('error', 'Unknown error')}")
    
    print("\nFederated orchestrator implementation complete!")