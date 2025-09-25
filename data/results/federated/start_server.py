
import sys
sys.path.append('.')
from ml.federated.server import start_federated_server

if __name__ == "__main__":
    start_federated_server(
        num_rounds=3,
        min_clients=2,
        server_address="localhost:8080",
        save_path="data/results/federated/federated_model.pth"
    )
