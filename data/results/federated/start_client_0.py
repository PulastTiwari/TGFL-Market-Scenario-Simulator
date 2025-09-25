import sys
sys.path.append('.')
import json
from pathlib import Path
from ml.federated.client import TGFLClient
from flwr.client import start_numpy_client

def load_client_data(client_id: int):
    base_dir = Path(__file__).resolve().parent
    with open(base_dir / f"client_{client_id}_data.json", 'r') as f:
        return json.load(f)['data']

if __name__ == "__main__":
    client_id = 0
    data = load_client_data(client_id)

    # Minimal model config (TGFLClient constructs its own tiny transformer)
    model_config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'sequence_length': 256,
        'dropout': 0.1,
    }

    client = TGFLClient(
        client_id=f"client_{client_id}",
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
        server_address="localhost:8080",
        client=client,
    )
