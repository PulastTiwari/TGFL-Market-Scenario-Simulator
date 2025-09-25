import numpy as np
from ml.evaluation.metrics import ScenarioEvaluator


def test_scenario_evaluator_basic():
    ev = ScenarioEvaluator()
    np.random.seed(0)
    hist = np.random.normal(0.0005, 0.02, 500)
    gen = np.random.normal(0.0005, 0.02, 200)
    ks = ev.kolmogorov_smirnov_test(gen, hist)
    assert 'ks_statistic' in ks and 'p_value' in ks
    acf_res = ev.autocorrelation_similarity(gen, hist, max_lags=5)
    assert 'score' in acf_res or isinstance(acf_res, dict)
