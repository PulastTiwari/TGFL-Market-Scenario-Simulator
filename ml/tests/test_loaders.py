import numpy as np
from ml.data.loaders import SyntheticDataGenerator, MarketDataLoader


def test_generate_regime_and_sequences():
    gen = SyntheticDataGenerator(seed=0)
    df = gen.generate_regime_data('normal', length=120, base_price=100.0)
    assert 'close' in df.columns
    loader = MarketDataLoader()
    sequences, targets = loader.create_sequences(df, sequence_length=20)
    assert sequences.shape[0] == targets.shape[0]
    assert sequences.shape[1] == 20


def test_create_mixed_regime_data():
    gen = SyntheticDataGenerator(seed=1)
    combined = gen.create_mixed_regime_data({'normal': 50, 'bull': 50}, base_price=100.0)
    assert len(combined) == 100
    assert 'log_returns' in combined.columns
