"""
Data loading and preprocessing utilities for TGFL Market Scenario Simulator
Handles yfinance data fetching, caching, and synthetic data generation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataLoader:
    """Handles loading and caching of market data"""
    
    def __init__(self, cache_dir: str = "../data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_stock_data(
        self, 
        symbols: List[str], 
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance with caching
        
        Args:
            symbols: List of stock symbols (e.g., ['SPY', 'AAPL'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (None for today)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping symbol to DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        cache_file = self.cache_dir / f"market_data_{start_date}_{end_date}.pkl"
        
        # Try to load from cache first
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, fetching fresh data")
        
        # Fetch fresh data
        logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}")
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty:
                    # Clean and standardize data
                    df = df.dropna()
                    df.index.name = 'Date'
                    df.columns = [col.replace(' ', '_').lower() for col in df.columns]
                    
                    # Add derived features
                    df['returns'] = df['close'].pct_change()
                    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
                    
                    data[symbol] = df
                    logger.info(f"Fetched {len(df)} rows for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
        
        # Cache the results
        if data:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Cached data to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        
        return data
    
    def create_sequences(
        self, 
        data: pd.DataFrame, 
        sequence_length: int = 256,
        target_column: str = 'log_returns',
        step_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling
        
        Args:
            data: DataFrame with time series data
            sequence_length: Length of input sequences
            target_column: Column to use as target values
            step_size: Step size for sliding window
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        # Get target series and remove NaN values
        series = data[target_column].dropna().values
        
        if len(series) < sequence_length + 1:
            raise ValueError(f"Not enough data: {len(series)} < {sequence_length + 1}")
        
        sequences = []
        targets = []
        
        for i in range(0, len(series) - sequence_length, step_size):
            seq = series[i:i + sequence_length]
            target = series[i + sequence_length]
            sequences.append(seq)
            targets.append(target)

        sequences_arr = np.array(sequences)
        targets_arr = np.array(targets)

        # Ensure at least one sequence is returned
        return sequences_arr, targets_arr

class SyntheticDataGenerator:
    """Generate synthetic market data for different regimes"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def generate_regime_data(
        self, 
        regime: str, 
        length: int = 1000,
        base_price: float = 100.0
    ) -> pd.DataFrame:
        """
        Generate synthetic price data for different market regimes
        
        Args:
            regime: One of 'bull', 'bear', 'volatile', 'normal'
            length: Number of time steps
            base_price: Starting price
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        # Regime parameters
        regime_params = {
            'normal': {'drift': 0.0002, 'volatility': 0.015, 'vol_of_vol': 0.1},
            'bull': {'drift': 0.0008, 'volatility': 0.012, 'vol_of_vol': 0.08},
            'bear': {'drift': -0.0005, 'volatility': 0.018, 'vol_of_vol': 0.12},
            'volatile': {'drift': 0.0001, 'volatility': 0.025, 'vol_of_vol': 0.2}
        }
        
        if regime not in regime_params:
            raise ValueError(f"Unknown regime: {regime}")
        
        params = regime_params[regime]
        
        # Generate stochastic volatility
        log_vol = np.zeros(length)
        log_vol[0] = np.log(params['volatility'])
        
        for t in range(1, length):
            log_vol[t] = 0.95 * log_vol[t-1] + params['vol_of_vol'] * self.rng.normal()
        
        volatility = np.exp(log_vol)
        
        # Generate returns with stochastic volatility
        returns = []
        for t in range(length):
            return_t = params['drift'] + volatility[t] * self.rng.normal()
            returns.append(return_t)
        
        returns = np.array(returns)
        
        # Generate price path
        log_prices = np.cumsum(returns)
        prices = base_price * np.exp(log_prices)
        
        # Create OHLCV data (simplified)
        dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        
        # Simple OHLC approximation
        daily_range = volatility * prices * 0.5  # Approximate daily range
        df['high'] = df['close'] + daily_range * self.rng.uniform(0, 1, length)
        df['low'] = df['close'] - daily_range * self.rng.uniform(0, 1, length)
        df['open'] = df['low'] + (df['high'] - df['low']) * self.rng.uniform(0, 1, length)
        
        # Volume (synthetic)
        base_volume = 1000000
        df['volume'] = base_volume * (1 + 0.5 * self.rng.normal(size=length))
        df['volume'] = df['volume'].clip(lower=0)
        
        # Derived features
        df['returns'] = returns
        df['log_returns'] = returns  # Already log returns
        df['volatility'] = volatility * np.sqrt(252)  # Annualized
        
        return df
    
    def create_mixed_regime_data(
        self, 
        regime_lengths: Dict[str, int],
        base_price: float = 100.0
    ) -> pd.DataFrame:
        """
        Create data with multiple regime switches
        
        Args:
            regime_lengths: Dict mapping regime name to length
            base_price: Starting price
            
        Returns:
            DataFrame with mixed regime data
        """
        """
        Create a concatenated time series made from multiple regimes.

        Args:
            regime_lengths: mapping of regime name -> length in time steps
            base_price: starting price for the first regime

        Returns:
            pd.DataFrame with synthetic OHLCV-like columns and a continuous price series
        """
        frames = []
        current_base = base_price
        for regime, length in regime_lengths.items():
            df = self.generate_regime_data(regime, length=length, base_price=current_base)
            # update base for next regime to prevent large jumps
            current_base = float(df['close'].iloc[-1])
            frames.append(df)

        if not frames:
            raise ValueError("regime_lengths must contain at least one regime with positive length")

        combined = pd.concat(frames, ignore_index=True)
        # Recompute log returns for the combined series
        combined['log_returns'] = np.log(combined['close']).diff()
        combined = combined.fillna(0.0)
        return combined

def create_federated_partitions(
    data: Dict[str, pd.DataFrame], 
    num_clients: int = 5,
    partition_strategy: str = 'temporal'
) -> List[Dict[str, pd.DataFrame]]:
    """
    Partition data across federated clients
    
    Args:
        data: Dictionary of symbol -> DataFrame
        num_clients: Number of federated clients
        partition_strategy: 'temporal', 'asset', or 'mixed'
        
    Returns:
        List of data partitions for each client
    """
    if partition_strategy == 'temporal':
        # Split by time periods
        partitions = []
        for symbol, df in data.items():
            n_samples = len(df)
            chunk_size = n_samples // num_clients
            
            for i in range(num_clients):
                start_idx = i * chunk_size
                if i == num_clients - 1:  # Last client gets remainder
                    end_idx = n_samples
                else:
                    end_idx = (i + 1) * chunk_size
                
                if i >= len(partitions):
                    partitions.append({})
                
                partitions[i][symbol] = df.iloc[start_idx:end_idx].copy()
        
        return partitions
    
    elif partition_strategy == 'asset':
        # Each client gets different assets (if multiple assets)
        symbols = list(data.keys())
        partitions = [{}] * num_clients
        
        for i, symbol in enumerate(symbols):
            client_idx = i % num_clients
            if not partitions[client_idx]:
                partitions[client_idx] = {}
            partitions[client_idx][symbol] = data[symbol].copy()
        
        return partitions
    
    else:  # mixed strategy
        # Combine temporal and asset splits
        # Implementation depends on specific requirements
        return create_federated_partitions(data, num_clients, 'temporal')

# Example usage and testing
if __name__ == "__main__":
    # Test data loading
    loader = MarketDataLoader()
    
    # Test synthetic data generation  
    generator = SyntheticDataGenerator()
    
    print("Testing synthetic data generation...")
    for regime in ['normal', 'bull', 'bear', 'volatile']:
        df = generator.generate_regime_data(regime, length=100)
        print(f"{regime}: mean return = {df['returns'].mean():.6f}, volatility = {df['volatility'].mean():.4f}")
    
    print("\nTesting sequence creation...")
    test_df = generator.generate_regime_data('normal', length=300)
    sequences, targets = loader.create_sequences(test_df, sequence_length=50)
    print(f"Created {len(sequences)} sequences of length {sequences.shape[1]}")