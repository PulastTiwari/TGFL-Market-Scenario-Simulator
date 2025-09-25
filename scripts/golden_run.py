"""
Deterministic "golden run" generator for demo:
- Creates/loads a baseline price series (GBM fallback).
- Generates a single synthetic scenario via AR(1) fitted to baseline returns.
- Computes KS p-value, ACF MAE, rolling-vol MAE.
- Writes:
  data/results/golden/metrics.json
  data/results/golden/samples.csv
  data/results/golden/plot.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf

# Deterministic seed for reproducibility
SEED = 12345
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results" / "golden"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters (safe for CPU+8GB)
NUM_POINTS = 512
ROLLING_WINDOW = 20
ACF_LAGS = 10

def load_baseline():
    """Load baseline price series from cache or generate synthetic GBM"""
    # Attempt to load a cached baseline CSV/parquet in data/cache/
    cache_dir = ROOT / "data" / "cache"
    if cache_dir.exists():
        for p in cache_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(p)
                if "close" in df.columns:
                    return df["close"].dropna().astype(float).to_numpy()[:NUM_POINTS]
            except Exception:
                continue
        for p in cache_dir.glob("*.csv"):
            try:
                df = pd.read_csv(p)
                if "close" in df.columns:
                    return df["close"].dropna().astype(float).to_numpy()[:NUM_POINTS]
            except Exception:
                continue
    
    # Fallback: synthetic GBM baseline
    print("No cached data found, generating synthetic baseline...")
    mu = 0.0005  # daily drift
    sigma = 0.015  # daily volatility
    dt = 1.0
    s0 = 100.0
    eps = np.random.normal(loc=0.0, scale=1.0, size=NUM_POINTS)
    returns = mu * dt + sigma * np.sqrt(dt) * eps
    prices = s0 * np.exp(np.cumsum(returns))
    return prices

def returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Calculate log returns from price series"""
    r = np.diff(np.log(prices))
    return r

def fit_ar1_and_generate(r_baseline: np.ndarray, length: int) -> np.ndarray:
    """Fit AR(1) model to baseline returns and generate new series"""
    x = r_baseline[:-1]
    y = r_baseline[1:]
    
    if len(x) < 10:
        phi = 0.0
        resid_sigma = np.std(y) if len(y) > 0 else 1e-3
    else:
        # Simple OLS estimation: phi = (X'Y) / (X'X)
        phi = np.sum(x * y) / np.sum(x * x)
        phi = max(-0.99, min(0.99, phi))  # Keep stationary
        resid = y - phi * x
        resid_sigma = np.std(resid)
    
    # Generate new return series
    out = np.empty(length)
    out[0] = r_baseline[-1] if len(r_baseline) > 0 else 0.0  # warm-start
    
    for t in range(1, length):
        out[t] = phi * out[t-1] + np.random.normal(scale=resid_sigma)
    
    return out

def compute_metrics(baseline_prices, synth_prices):
    """Compute KS test, ACF similarity, and rolling volatility metrics"""
    r_base = returns_from_prices(baseline_prices)
    r_synth = returns_from_prices(synth_prices)
    
    # KS test on returns
    try:
        ks_res = ks_2samp(r_base, r_synth)
        ks_pvalue = float(ks_res.pvalue)
    except Exception:
        ks_pvalue = 0.0
    
    # ACF similarity (Mean Absolute Error)
    try:
        acf_base = acf(r_base, nlags=ACF_LAGS, fft=False)
        acf_synth = acf(r_synth, nlags=ACF_LAGS, fft=False)
        acf_mae = float(np.mean(np.abs(np.asarray(acf_base) - np.asarray(acf_synth))))
    except Exception:
        acf_mae = 1.0
    
    # Rolling volatility similarity
    try:
        roll_base = pd.Series(r_base).rolling(window=ROLLING_WINDOW, min_periods=1).std().to_numpy()
        roll_synth = pd.Series(r_synth).rolling(window=ROLLING_WINDOW, min_periods=1).std().to_numpy()
        # Remove NaN values and align lengths
        roll_base = roll_base[~np.isnan(roll_base)]
        roll_synth = roll_synth[~np.isnan(roll_synth)]
        n = min(len(roll_base), len(roll_synth))
        if n > 0:
            rolling_mae = float(np.mean(np.abs(roll_base[-n:] - roll_synth[-n:])))
        else:
            rolling_mae = 0.0
    except Exception:
        rolling_mae = 1.0
    
    return {
        "ks_pvalue": ks_pvalue,
        "acf_mae": acf_mae,
        "rolling_vol_mae": rolling_mae,
        "num_points": len(synth_prices),
        "baseline_return_std": float(np.std(r_base)),
        "synth_return_std": float(np.std(r_synth))
    }

def save_samples_csv(baseline: np.ndarray, synth: np.ndarray, path: Path):
    """Save both baseline and synthetic price series to CSV"""
    n = min(len(baseline), len(synth))
    df = pd.DataFrame({
        "t": np.arange(n),
        "baseline_price": baseline[:n],
        "synth_price": synth[:n]
    })
    df.to_csv(path, index=False)

def save_plot(baseline, synth, path: Path):
    """Create and save comparison plot"""
    plt.figure(figsize=(10, 6))
    
    # Price series
    plt.subplot(2, 1, 1)
    plt.plot(baseline, label="Baseline", alpha=0.8, linewidth=1.2)
    plt.plot(synth, label="Generated", alpha=0.7, linewidth=1.2)
    plt.title("Price Series Comparison")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Returns comparison
    plt.subplot(2, 1, 2)
    r_base = returns_from_prices(baseline)
    r_synth = returns_from_prices(synth)
    plt.plot(r_base, label="Baseline Returns", alpha=0.8, linewidth=1)
    plt.plot(r_synth, label="Generated Returns", alpha=0.7, linewidth=1)
    plt.title("Returns Comparison")
    plt.ylabel("Log Returns")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()

def main():
    """Main golden run execution"""
    print("Starting golden run generation...")
    
    # Load or generate baseline
    baseline = load_baseline()
    print(f"Loaded baseline with {len(baseline)} points")
    
    # Generate synthetic scenario deterministically
    r_baseline = returns_from_prices(baseline)
    r_synth = fit_ar1_and_generate(r_baseline, len(baseline))
    
    # Convert returns back to price series (start from baseline initial value)
    s0 = baseline[0]
    synth_prices = s0 * np.exp(np.cumsum(np.concatenate(([0], r_synth))))
    
    # Compute evaluation metrics
    metrics = compute_metrics(baseline, synth_prices)
    
    # Add generation metadata
    metrics.update({
        "seed": SEED,
        "generation_method": "AR1_fitted_to_baseline",
        "generated_at": pd.Timestamp.now().isoformat(),
        "model_version": "golden_v1"
    })
    
    # Save artifacts
    metrics_path = RESULTS_DIR / "metrics.json"
    samples_path = RESULTS_DIR / "samples.csv"
    plot_path = RESULTS_DIR / "plot.png"
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    save_samples_csv(baseline, synth_prices, samples_path)
    save_plot(baseline, synth_prices, plot_path)
    
    print(f"Golden run artifacts saved to: {RESULTS_DIR}")
    print("Generated files:")
    print(f"  - {metrics_path}")
    print(f"  - {samples_path}")
    print(f"  - {plot_path}")
    print("\nMetrics summary:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()