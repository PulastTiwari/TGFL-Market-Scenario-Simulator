"""
Evaluation metrics for generated market scenarios
Implements statistical tests to validate realism of synthetic data
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ScenarioEvaluator:
    """Evaluates quality of generated market scenarios"""
    
    def __init__(self):
        self.metrics_cache = {}
        
    def kolmogorov_smirnov_test(
        self, 
        generated_returns: np.ndarray, 
        historical_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform Kolmogorov-Smirnov test for distributional similarity
        
        Args:
            generated_returns: Array of generated daily returns
            historical_returns: Array of historical daily returns
            
        Returns:
            Dictionary with KS statistic and p-value
        """
        # Remove any NaN or infinite values
        generated_clean = generated_returns[np.isfinite(generated_returns)]
        historical_clean = historical_returns[np.isfinite(historical_returns)]
        
        if len(generated_clean) == 0 or len(historical_clean) == 0:
            return {'ks_statistic': np.nan, 'p_value': 0.0}
        
        # Perform KS test (ensure safe unpacking and float conversion)
        ks_result = ks_2samp(generated_clean, historical_clean)
        if isinstance(ks_result, tuple) and len(ks_result) >= 2:
            ks_stat_raw, p_value_raw = ks_result[0], ks_result[1]
            try:
                ks_statistic = float(np.asarray(ks_stat_raw).item())
            except Exception:
                ks_statistic = float(ks_stat_raw) if isinstance(ks_stat_raw, (int, float)) else np.nan
            try:
                p_value = float(np.asarray(p_value_raw).item())
            except Exception:
                p_value = float(p_value_raw) if isinstance(p_value_raw, (int, float)) else 0.0
        else:
            try:
                ks_statistic = float(np.asarray(ks_result).item())
            except Exception:
                ks_statistic = np.nan
            p_value = 0.0

        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value
        }
    
    def autocorrelation_similarity(
        self, 
        generated_returns: np.ndarray, 
        historical_returns: np.ndarray,
        max_lags: int = 20
    ) -> Dict[str, Any]:
        """
        Compare autocorrelation functions between generated and historical returns
        
        Args:
            generated_returns: Array of generated daily returns
            historical_returns: Array of historical daily returns
            max_lags: Maximum number of lags to compute
            
        Returns:
            Dictionary with ACF comparison metrics
        """
        try:
            # Clean data
            generated_clean = generated_returns[np.isfinite(generated_returns)]
            historical_clean = historical_returns[np.isfinite(historical_returns)]
            
            if len(generated_clean) < max_lags * 2 or len(historical_clean) < max_lags * 2:
                return {
                    'acf_similarity_score': 0.0,
                    'acf_generated': np.zeros(max_lags + 1),
                    'acf_historical': np.zeros(max_lags + 1),
                    'acf_mse': np.inf
                }
            
            # Compute autocorrelation functions; acf may return (acf, confint) in some versions
            acf_generated_raw = acf(generated_clean, nlags=max_lags, fft=True)
            acf_historical_raw = acf(historical_clean, nlags=max_lags, fft=True)

            # If acf returned a tuple (acf, confint), extract the first element
            if isinstance(acf_generated_raw, tuple):
                acf_generated = np.asarray(acf_generated_raw[0])
            else:
                acf_generated = np.asarray(acf_generated_raw)

            if isinstance(acf_historical_raw, tuple):
                acf_historical = np.asarray(acf_historical_raw[0])
            else:
                acf_historical = np.asarray(acf_historical_raw)
            
            # Compute similarity score (1 - normalized MSE)
            mse = np.mean((acf_generated - acf_historical) ** 2)
            max_var = max(np.var(acf_generated), np.var(acf_historical), 1e-8)
            similarity_score = max(0.0, 1.0 - mse / max_var)
            
            return {
                'acf_similarity_score': float(similarity_score),
                'acf_generated': acf_generated,
                'acf_historical': acf_historical,
                'acf_mse': float(mse)
            }
            
        except Exception as e:
            print(f"ACF computation failed: {e}")
            return {
                'acf_similarity_score': 0.0,
                'acf_generated': np.zeros(max_lags + 1),
                'acf_historical': np.zeros(max_lags + 1), 
                'acf_mse': np.inf
            }
    
    def volatility_clustering_analysis(
        self, 
        generated_returns: np.ndarray, 
        historical_returns: np.ndarray,
        window_size: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze volatility clustering patterns
        
        Args:
            generated_returns: Array of generated daily returns
            historical_returns: Array of historical daily returns
            window_size: Rolling window size for volatility calculation
            
        Returns:
            Dictionary with volatility clustering metrics
        """
        try:
            # Clean data
            generated_clean = generated_returns[np.isfinite(generated_returns)]
            historical_clean = historical_returns[np.isfinite(historical_returns)]
            
            if len(generated_clean) < window_size * 2 or len(historical_clean) < window_size * 2:
                return {
                    'volatility_similarity_score': 0.0,
                    'vol_corr_generated': 0.0,
                    'vol_corr_historical': 0.0,
                    'vol_ks_p_value': 0.0
                }
            
            # Compute rolling volatility
            generated_df = pd.DataFrame({'returns': generated_clean})
            historical_df = pd.DataFrame({'returns': historical_clean})

            gen_vol = generated_df['returns'].rolling(window=window_size).std()
            hist_vol = historical_df['returns'].rolling(window=window_size).std()

            # Remove NaN values from rolling calculation and ensure numpy arrays
            gen_vol_clean = np.asarray(gen_vol.dropna())
            hist_vol_clean = np.asarray(hist_vol.dropna())

            # Ensure there are enough points to compute lag-1 correlation
            if gen_vol_clean.size < 2 or hist_vol_clean.size < 2:
                return {
                    'volatility_similarity_score': 0.0,
                    'vol_corr_generated': 0.0,
                    'vol_corr_historical': 0.0,
                    'vol_ks_p_value': 0.0
                }

            # Compute volatility autocorrelation (volatility clustering measure)
            gen_vol_corr = np.corrcoef(gen_vol_clean[:-1], gen_vol_clean[1:])[0, 1]
            hist_vol_corr = np.corrcoef(hist_vol_clean[:-1], hist_vol_clean[1:])[0, 1]
            
            # Handle NaN correlations
            gen_vol_corr = 0.0 if np.isnan(gen_vol_corr) else gen_vol_corr
            hist_vol_corr = 0.0 if np.isnan(hist_vol_corr) else hist_vol_corr
            
            # KS test on volatility distributions
            vol_ks_result = ks_2samp(np.asarray(gen_vol_clean), np.asarray(hist_vol_clean))
            # ks_2samp returns (statistic, pvalue)
            if isinstance(vol_ks_result, tuple) and len(vol_ks_result) >= 2:
                vol_ks_stat_raw, vol_ks_p_raw = vol_ks_result[0], vol_ks_result[1]
                try:
                    vol_ks_stat = float(np.asarray(vol_ks_stat_raw).item())
                except Exception:
                    vol_ks_stat = float(vol_ks_stat_raw) if isinstance(vol_ks_stat_raw, (int, float)) else 0.0
                try:
                    vol_ks_p = float(np.asarray(vol_ks_p_raw).item())
                except Exception:
                    vol_ks_p = float(vol_ks_p_raw) if isinstance(vol_ks_p_raw, (int, float)) else 0.0
            else:
                try:
                    vol_ks_stat = float(np.asarray(vol_ks_result).item())
                except Exception:
                    vol_ks_stat = 0.0
                vol_ks_p = 0.0
            
            # Similarity score based on correlation difference
            corr_diff = abs(gen_vol_corr - hist_vol_corr)
            vol_similarity = max(0.0, 1.0 - corr_diff)
            
            return {
                'volatility_similarity_score': float(vol_similarity),
                'vol_corr_generated': float(gen_vol_corr),
                'vol_corr_historical': float(hist_vol_corr),
                'vol_ks_p_value': float(vol_ks_p),
                'vol_generated': gen_vol_clean,
                'vol_historical': hist_vol_clean
            }
            
        except Exception as e:
            print(f"Volatility analysis failed: {e}")
            return {
                'volatility_similarity_score': 0.0,
                'vol_corr_generated': 0.0,
                'vol_corr_historical': 0.0,
                'vol_ks_p_value': 0.0
            }
    
    def distributional_moments(
        self, 
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate distributional moments of returns
        
        Args:
            returns: Array of daily returns
            
        Returns:
            Dictionary with statistical moments
        """
        clean_returns = returns[np.isfinite(returns)]
        
        if len(clean_returns) == 0:
            return {
                'mean': np.nan,
                'std': np.nan, 
                'skewness': np.nan,
                'kurtosis': np.nan,
                'min': np.nan,
                'max': np.nan
            }
        
        return {
            'mean': float(np.mean(clean_returns)),
            'std': float(np.std(clean_returns)),
            'skewness': float(stats.skew(clean_returns)),
            'kurtosis': float(stats.kurtosis(clean_returns)),
            'min': float(np.min(clean_returns)),
            'max': float(np.max(clean_returns))
        }
    
    def comprehensive_evaluation(
        self, 
        generated_scenarios: List[np.ndarray], 
        historical_data: np.ndarray,
        scenario_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of multiple scenarios
        
        Args:
            generated_scenarios: List of generated return arrays
            historical_data: Historical return array for comparison
            scenario_labels: Optional labels for scenarios
            
        Returns:
            Comprehensive evaluation results
        """
        if scenario_labels is None:
            scenario_labels = [f"Scenario_{i}" for i in range(len(generated_scenarios))]
        
        results = {
            'summary': {},
            'scenarios': {},
            'historical': {}
        }
        
        # Evaluate historical data
        hist_moments = self.distributional_moments(historical_data)
        results['historical'] = {
            'moments': hist_moments,
            'length': len(historical_data)
        }
        
        # Evaluate each scenario
        ks_p_values = []
        acf_scores = []
        vol_scores = []
        
        for i, (scenario, label) in enumerate(zip(generated_scenarios, scenario_labels)):
            # KS test
            ks_result = self.kolmogorov_smirnov_test(scenario, historical_data)
            
            # ACF analysis
            acf_result = self.autocorrelation_similarity(scenario, historical_data)
            
            # Volatility analysis
            vol_result = self.volatility_clustering_analysis(scenario, historical_data)
            
            # Distributional moments
            scenario_moments = self.distributional_moments(scenario)
            
            results['scenarios'][label] = {
                'ks_test': ks_result,
                'acf_analysis': acf_result,
                'volatility_analysis': vol_result,
                'moments': scenario_moments,
                'length': len(scenario)
            }
            
            # Collect summary statistics
            ks_p_values.append(ks_result['p_value'])
            acf_scores.append(acf_result['acf_similarity_score'])
            vol_scores.append(vol_result['volatility_similarity_score'])
        
        # Summary statistics
        results['summary'] = {
            'num_scenarios': len(generated_scenarios),
            'mean_ks_p_value': float(np.mean(ks_p_values)),
            'min_ks_p_value': float(np.min(ks_p_values)),
            'pass_ks_test_005': int(sum(1 for p in ks_p_values if p >= 0.05)),
            'mean_acf_score': float(np.mean(acf_scores)),
            'mean_vol_score': float(np.mean(vol_scores)),
            'overall_quality_score': float(np.mean([
                np.mean(ks_p_values), 
                np.mean(acf_scores), 
                np.mean(vol_scores)
            ]))
        }
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of evaluation results"""
        print("=" * 60)
        print("SCENARIO EVALUATION SUMMARY")
        print("=" * 60)
        
        summary = results['summary']
        historical = results['historical']
        
        print(f"Number of scenarios evaluated: {summary['num_scenarios']}")
        print(f"Historical data length: {historical['length']}")
        print()
        
        print("STATISTICAL TESTS:")
        print(f"  Mean KS-test p-value: {summary['mean_ks_p_value']:.4f}")
        print(f"  Scenarios passing KS-test (p≥0.05): {summary['pass_ks_test_005']}/{summary['num_scenarios']}")
        print(f"  Mean ACF similarity score: {summary['mean_acf_score']:.4f}")
        print(f"  Mean volatility similarity score: {summary['mean_vol_score']:.4f}")
        print()
        
        print(f"OVERALL QUALITY SCORE: {summary['overall_quality_score']:.4f}")
        
        if summary['mean_ks_p_value'] >= 0.05:
            print("[SUCCESS] PRIMARY SUCCESS CRITERION MET (KS p-value ≥ 0.05)")
        else:
            print("[FAIL] Primary success criterion not met (KS p-value < 0.05)")
        
        print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Test with synthetic data
    evaluator = ScenarioEvaluator()
    
    # Create test data
    np.random.seed(42)
    historical = np.random.normal(0.0005, 0.02, 1000)  # Normal market
    
    # Generate some test scenarios
    scenario1 = np.random.normal(0.0005, 0.02, 500)   # Similar to historical
    scenario2 = np.random.normal(0.002, 0.015, 500)   # Bull market
    scenario3 = np.random.normal(-0.001, 0.025, 500)  # Bear market
    
    scenarios = [scenario1, scenario2, scenario3]
    labels = ['Normal', 'Bull', 'Bear']
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(scenarios, historical, labels)
    
    # Print summary
    evaluator.print_evaluation_summary(results)