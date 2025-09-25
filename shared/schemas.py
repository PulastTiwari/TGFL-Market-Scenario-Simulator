"""
Shared schemas and types for TGFL Market Scenario Simulator
Used by both API (Pydantic) and frontend (TypeScript via code generation)
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

# Training related schemas
class TrainingRequest(BaseModel):
    """Request to start a federated training run"""
    num_clients: int = Field(default=5, ge=2, le=10, description="Number of federated clients")
    num_rounds: int = Field(default=10, ge=1, le=50, description="Number of training rounds")
    model_size: Literal["tiny", "small", "medium"] = Field(default="small", description="Model size")
    dataset: Literal["cached", "live", "synthetic"] = Field(default="cached", description="Dataset type")
    learning_rate: float = Field(default=0.001, gt=0, le=1, description="Learning rate")

class TrainingStatus(BaseModel):
    """Current status of a training run"""
    run_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    current_round: Optional[int] = None
    total_rounds: int
    progress: float = Field(ge=0.0, le=1.0, description="Progress from 0 to 1")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None

# Scenario generation schemas  
class ScenarioRequest(BaseModel):
    """Request to generate market scenarios"""
    regime: Literal["bull", "bear", "volatile", "normal"] = Field(description="Market regime")
    length: int = Field(default=252, ge=10, le=1000, description="Scenario length in days")
    num_scenarios: int = Field(default=1, ge=1, le=100, description="Number of scenarios")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    asset_symbol: str = Field(default="SPY", description="Asset symbol for scenario")

class ScenarioResponse(BaseModel):
    """Generated market scenario data"""
    scenario_id: str
    regime: str
    length: int
    asset_symbol: str
    data: List[List[float]] = Field(description="List of scenarios with time series values")
    timestamps: List[str] = Field(description="ISO timestamp strings")
    metadata: Dict[str, Any]
    created_at: datetime

# Evaluation schemas
class EvaluationMetrics(BaseModel):
    """Statistical evaluation metrics for scenarios"""
    ks_test_p_value: float = Field(description="Kolmogorov-Smirnov test p-value")
    ks_test_statistic: float = Field(description="KS test statistic")
    acf_similarity_score: float = Field(description="Autocorrelation function similarity")
    volatility_similarity_score: float = Field(description="Rolling volatility similarity")
    mean_return: float = Field(description="Mean daily return")
    volatility: float = Field(description="Annualized volatility")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown: float = Field(description="Maximum drawdown")

# API response schemas
class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "unhealthy"]
    message: str
    version: str
    uptime_seconds: float

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str
    timestamp: datetime
    request_id: Optional[str] = None

# Model configuration schemas
class ModelConfig(BaseModel):
    """Configuration for the transformer model"""
    vocab_size: int = Field(default=1000, description="Vocabulary size")
    d_model: int = Field(default=128, description="Model dimension")
    n_heads: int = Field(default=8, description="Number of attention heads")
    n_layers: int = Field(default=4, description="Number of transformer layers")
    sequence_length: int = Field(default=256, description="Maximum sequence length")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")

class FederatedConfig(BaseModel):
    """Configuration for federated learning"""
    aggregation_strategy: Literal["fedavg", "fedprox", "scaffold"] = Field(default="fedavg")
    client_fraction: float = Field(default=1.0, ge=0.1, le=1.0, description="Fraction of clients per round")
    local_epochs: int = Field(default=1, ge=1, le=10, description="Local epochs per client")
    min_fit_clients: int = Field(default=2, ge=1, description="Minimum clients for training")
    min_evaluate_clients: int = Field(default=2, ge=1, description="Minimum clients for evaluation")

# Data schemas
class DatasetInfo(BaseModel):
    """Information about available datasets"""
    name: str
    description: str
    asset_symbols: List[str]
    date_range: tuple[str, str]  # (start_date, end_date)
    num_samples: int
    cached: bool
    file_path: Optional[str] = None

# Federated Learning Schemas
class FederatedTrainingRequest(BaseModel):
    """Request to start federated training simulation"""
    num_clients: int = Field(default=4, ge=2, le=10, description="Number of federated clients")
    num_rounds: int = Field(default=10, ge=1, le=50, description="Number of federated rounds") 
    total_samples: int = Field(default=1000, ge=100, le=10000, description="Total data samples")
    partition_strategy: Literal["iid", "non_iid", "temporal"] = Field(default="iid", description="Data partitioning strategy")
    server_address: str = Field(default="localhost:8080", description="Federated server address")
    wait_for_completion: bool = Field(default=True, description="Wait for training completion")
    config: Optional[FederatedConfig] = Field(default=None, description="Additional federated config")

class FederatedTrainingStatus(BaseModel):
    """Status of federated training simulation"""
    simulation_id: str
    status: Literal["pending", "preparing", "running", "completed", "failed", "stopped"]
    current_round: Optional[int] = None
    total_rounds: int
    progress: float = Field(ge=0.0, le=1.0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    simulation_time: Optional[float] = None
    server_running: bool = False
    active_clients: int = 0
    total_clients: int
    error_message: Optional[str] = None
    results_path: Optional[str] = None

class FederatedMetrics(BaseModel):
    """Federated training specific metrics"""
    round_losses: List[float] = Field(description="Training loss per round")
    round_accuracies: List[float] = Field(description="Accuracy per round", default_factory=list)
    client_metrics: Dict[str, List[float]] = Field(description="Per-client metrics", default_factory=dict)
    convergence_round: Optional[int] = Field(description="Round where convergence achieved")
    avg_client_samples: float = Field(description="Average samples per client")
    data_distribution: Dict[str, int] = Field(description="Data distribution across clients")
    communication_cost: Optional[float] = Field(description="Total communication cost")

class ClientInfo(BaseModel):
    """Information about a federated client"""
    client_id: int
    data_samples: int
    training_loss: Optional[float] = None
    test_loss: Optional[float] = None
    status: Literal["idle", "training", "evaluating", "disconnected"] = "idle"
    last_update: Optional[datetime] = None

class FederatedSimulationResult(BaseModel):
    """Complete federated simulation results"""
    simulation_id: str
    success: bool
    simulation_time: float
    num_clients: int
    num_rounds: int
    total_samples: int
    partition_strategy: str
    server_address: str
    results_path: str
    final_model_path: Optional[str] = None
    metrics: Optional[FederatedMetrics] = None
    client_info: List[ClientInfo] = Field(default_factory=list)
    error_message: Optional[str] = None