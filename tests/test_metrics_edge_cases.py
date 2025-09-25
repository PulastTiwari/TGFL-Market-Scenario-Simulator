"""
Comprehensive unit tests for ML evaluation metrics with edge cases
Tests handling of NaNs, short series, empty data, and other edge conditions
"""

import pytest
import numpy as np
import pandas as pd
from ml.evaluation.metrics import ScenarioEvaluator
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


class TestScenarioEvaluatorEdgeCases:
    """Test ScenarioEvaluator with various edge cases"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.evaluator = ScenarioEvaluator()
        
        # Normal test data
        np.random.seed(42)
        self.normal_returns = np.random.normal(0, 0.02, 1000)
        self.normal_returns2 = np.random.normal(0.001, 0.018, 800)
        
    def test_ks_test_with_empty_arrays(self):
        """Test KS test with empty input arrays"""
        empty_array = np.array([])
        result = self.evaluator.kolmogorov_smirnov_test(empty_array, self.normal_returns)
        
        assert np.isnan(result['ks_statistic'])
        assert result['p_value'] == 0.0
        
    def test_ks_test_with_nans(self):
        """Test KS test with NaN values in data"""
        nan_returns = np.array([0.01, np.nan, -0.02, np.inf, 0.005])
        result = self.evaluator.kolmogorov_smirnov_test(nan_returns, self.normal_returns)
        
        # Should handle NaNs gracefully
        assert isinstance(result['ks_statistic'], float)
        assert isinstance(result['p_value'], float)
        assert result['p_value'] >= 0.0
        
    def test_ks_test_with_all_nans(self):
        """Test KS test with arrays containing only NaN/inf values"""
        bad_array = np.array([np.nan, np.inf, -np.inf])
        result = self.evaluator.kolmogorov_smirnov_test(bad_array, self.normal_returns)
        
        assert np.isnan(result['ks_statistic'])
        assert result['p_value'] == 0.0
        
    def test_ks_test_single_value_arrays(self):
        """Test KS test with single-value arrays"""
        single_val = np.array([0.01])
        result = self.evaluator.kolmogorov_smirnov_test(single_val, self.normal_returns)
        
        # Should handle gracefully
        assert isinstance(result['ks_statistic'], float)
        assert isinstance(result['p_value'], float)
        
    def test_acf_with_short_series(self):
        """Test ACF computation with series shorter than max_lags"""
        short_returns = np.random.normal(0, 0.02, 5)  # Very short series
        max_lags = 20
        
        result = self.evaluator.autocorrelation_similarity(
            short_returns, self.normal_returns, max_lags=max_lags
        )
        
        assert result['acf_similarity_score'] == 0.0
        assert len(result['acf_generated']) == max_lags + 1
        assert len(result['acf_historical']) == max_lags + 1
        assert result['acf_mse'] == np.inf
        
    def test_acf_with_constant_series(self):
        """Test ACF with constant (zero variance) series"""
        constant_series = np.full(100, 0.01)  # All same value
        
        result = self.evaluator.autocorrelation_similarity(
            constant_series, self.normal_returns, max_lags=10
        )
        
        # Should handle gracefully without crashing
        assert isinstance(result['acf_similarity_score'], float)
        assert result['acf_similarity_score'] >= 0.0
        
    def test_acf_with_nans(self):
        """Test ACF computation with NaN values"""
        nan_returns = self.normal_returns.copy()
        nan_returns[::50] = np.nan  # Insert NaNs every 50th element
        
        result = self.evaluator.autocorrelation_similarity(
            nan_returns, self.normal_returns, max_lags=10
        )
        
        # Should clean NaNs and compute successfully
        assert isinstance(result['acf_similarity_score'], float)
        assert result['acf_similarity_score'] >= 0.0
        
    def test_volatility_clustering_short_window(self):
        """Test volatility clustering with window larger than data"""
        short_returns = np.random.normal(0, 0.02, 15)
        large_window = 30
        
        result = self.evaluator.volatility_clustering_analysis(
            short_returns, self.normal_returns, window_size=large_window
        )
        
        assert result['volatility_similarity_score'] == 0.0
        assert result['vol_corr_generated'] == 0.0
        assert result['vol_corr_historical'] == 0.0
        assert result['vol_ks_p_value'] == 0.0
        
    def test_volatility_clustering_with_nans(self):
        """Test volatility clustering with NaN values"""
        nan_returns = self.normal_returns.copy()
        nan_returns[::100] = np.nan
        
        result = self.evaluator.volatility_clustering_analysis(
            nan_returns, self.normal_returns, window_size=20
        )
        
        # Should handle NaNs gracefully
        assert isinstance(result['volatility_similarity_score'], float)
        assert isinstance(result['vol_corr_generated'], float)
        assert isinstance(result['vol_corr_historical'], float)
        
    def test_volatility_clustering_zero_variance(self):
        """Test volatility clustering with zero-variance series"""
        zero_var_returns = np.zeros(100)
        
        result = self.evaluator.volatility_clustering_analysis(
            zero_var_returns, self.normal_returns, window_size=10
        )
        
        # Should handle zero variance without crashing
        assert isinstance(result['volatility_similarity_score'], float)
        assert result['volatility_similarity_score'] >= 0.0
        
    def test_distributional_moments_edge_cases(self):
        """Test distributional moments with edge case data"""
        # Test with very small array
        small_returns = np.array([0.01, -0.02])
        
        result = self.evaluator.distributional_moments(small_returns)
        
        # Should compute basic moments even for small arrays
        assert isinstance(result['mean'], float)
        assert isinstance(result['std'], float)
        assert isinstance(result['skewness'], float)
        assert isinstance(result['kurtosis'], float)
        
    def test_distributional_moments_with_outliers(self):
        """Test distributional moments with extreme outliers"""
        outlier_returns = self.normal_returns.copy()
        outlier_returns[0] = 100.0  # Extreme outlier
        outlier_returns[1] = -50.0  # Extreme negative outlier
        
        result = self.evaluator.distributional_moments(outlier_returns)
        
        # Should handle outliers without crashing
        assert isinstance(result['mean'], float)
        assert isinstance(result['std'], float)
        # Skewness and kurtosis may be extreme but should be finite
        assert np.isfinite(result['skewness'])
        assert np.isfinite(result['kurtosis'])
        
    def test_comprehensive_evaluation_edge_cases(self):
        """Test comprehensive evaluation with various edge cases"""
        # Very short series
        short_gen = np.random.normal(0, 0.02, 50)
        short_hist = np.random.normal(0, 0.02, 60)
        
        result = self.evaluator.comprehensive_evaluation([short_gen], short_hist)
        
        # Should return all expected keys based on actual implementation
        expected_keys = [
            'historical', 'scenarios', 'summary'
        ]
        
        for key in expected_keys:
            assert key in result
            
        # Overall quality score should be between 0 and 1
        assert 0.0 <= result['summary']['overall_quality_score'] <= 1.0
        
    def test_comprehensive_evaluation_identical_series(self):
        """Test comprehensive evaluation with identical series"""
        identical_returns = self.normal_returns.copy()
        
        result = self.evaluator.comprehensive_evaluation([identical_returns], identical_returns)
        
        # Should get perfect or near-perfect scores
        assert result['summary']['overall_quality_score'] > 0.8  # Should be high similarity
        assert len(result['scenarios']) == 1
        
    def test_type_safety_all_methods(self):
        """Test that all methods return proper types and handle type conversion"""
        test_data = np.array([0.01, -0.02, 0.005, -0.01, 0.03])
        
        # Test KS test return types
        ks_result = self.evaluator.kolmogorov_smirnov_test(test_data, self.normal_returns)
        assert isinstance(ks_result['ks_statistic'], (float, np.floating))
        assert isinstance(ks_result['p_value'], (float, np.floating))
        
        # Test ACF return types
        acf_result = self.evaluator.autocorrelation_similarity(test_data, self.normal_returns, max_lags=5)
        assert isinstance(acf_result['acf_similarity_score'], (float, np.floating))
        assert isinstance(acf_result['acf_generated'], np.ndarray)
        assert isinstance(acf_result['acf_historical'], np.ndarray)
        assert isinstance(acf_result['acf_mse'], (float, np.floating))
        
        # Test volatility clustering return types
        vol_result = self.evaluator.volatility_clustering_analysis(test_data, self.normal_returns, window_size=3)
        assert isinstance(vol_result['volatility_similarity_score'], (float, np.floating))
        assert isinstance(vol_result['vol_corr_generated'], (float, np.floating))
        assert isinstance(vol_result['vol_corr_historical'], (float, np.floating))
        assert isinstance(vol_result['vol_ks_p_value'], (float, np.floating))


if __name__ == "__main__":
    # Run a quick smoke test
    test_class = TestScenarioEvaluatorEdgeCases()
    test_class.setup_method()
    
    print("Running edge case tests...")
    
    try:
        test_class.test_ks_test_with_empty_arrays()
        print("✓ Empty arrays test passed")
        
        test_class.test_ks_test_with_nans()
        print("✓ NaN handling test passed")
        
        test_class.test_acf_with_short_series()
        print("✓ Short series ACF test passed")
        
        test_class.test_volatility_clustering_short_window()
        print("✓ Short window volatility test passed")
        
        test_class.test_type_safety_all_methods()
        print("✓ Type safety test passed")
        
        print("\nAll edge case tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise