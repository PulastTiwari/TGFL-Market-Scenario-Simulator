/**
 * Shared TypeScript schemas for TGFL Market Scenario Simulator
 * Keep in sync with shared/schemas.py
 */

// Training related types
export interface TrainingRequest {
  num_clients: number; // 2-10
  num_rounds: number; // 1-50  
  model_size: 'tiny' | 'small' | 'medium';
  dataset: 'cached' | 'live' | 'synthetic';
  learning_rate: number; // 0-1
}

export interface TrainingStatus {
  run_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  current_round?: number;
  total_rounds: number;
  progress: number; // 0.0-1.0
  start_time?: string; // ISO datetime
  end_time?: string; // ISO datetime  
  metrics?: Record<string, number>;
  error_message?: string;
}

// Scenario generation types
export interface ScenarioRequest {
  regime: 'bull' | 'bear' | 'volatile' | 'normal';
  length: number; // 10-1000 days
  num_scenarios: number; // 1-100
  seed?: number;
  asset_symbol: string;
}

export interface ScenarioResponse {
  scenario_id: string;
  regime: string;
  length: number;
  asset_symbol: string;
  data: number[][]; // Array of scenarios with time series values
  timestamps: string[]; // ISO timestamp strings
  metadata: Record<string, any>;
  created_at: string; // ISO datetime
}

// Evaluation types
export interface EvaluationMetrics {
  ks_test_p_value: number;
  ks_test_statistic: number;
  acf_similarity_score: number;
  volatility_similarity_score: number;
  mean_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
}

// API response types
export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  message: string;
  version: string;
  uptime_seconds: number;
}

export interface ErrorResponse {
  error: string;
  detail: string;
  timestamp: string; // ISO datetime
  request_id?: string;
}

// Model configuration types
export interface ModelConfig {
  vocab_size: number;
  d_model: number;
  n_heads: number;
  n_layers: number;
  sequence_length: number;
  dropout: number; // 0.0-1.0
}

export interface FederatedConfig {
  aggregation_strategy: 'fedavg' | 'fedprox' | 'scaffold';
  client_fraction: number; // 0.1-1.0
  local_epochs: number; // 1-10
  min_fit_clients: number;
  min_evaluate_clients: number;
}

// Data types
export interface DatasetInfo {
  name: string;
  description: string;
  asset_symbols: string[];
  date_range: [string, string]; // [start_date, end_date]
  num_samples: number;
  cached: boolean;
  file_path?: string;
}

// Chart data types for visualization
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  scenario?: number;
}

export interface MetricsChartData {
  round: number;
  loss: number;
  accuracy?: number;
  client_id?: string;
}

// UI state types
export interface TrainingState {
  isTraining: boolean;
  currentRun?: TrainingStatus;
  runs: TrainingStatus[];
  error?: string;
}

export interface ScenarioState {
  scenarios: ScenarioResponse[];
  isGenerating: boolean;
  selectedScenario?: ScenarioResponse;
  error?: string;
}

// Form validation schemas (for use with zod)
export const TRAINING_REQUEST_DEFAULTS: TrainingRequest = {
  num_clients: 5,
  num_rounds: 10,
  model_size: 'small',
  dataset: 'cached',
  learning_rate: 0.001,
};

export const SCENARIO_REQUEST_DEFAULTS: ScenarioRequest = {
  regime: 'normal',
  length: 252,
  num_scenarios: 1,
  asset_symbol: 'SPY',
};