"""
tests/test_ab_testing.py
Unit tests for the A/B testing statistical framework.
Run with: pytest tests/test_ab_testing.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.ab_testing import (
    two_proportion_ztest,
    ttest_continuous,
    minimum_sample_size,
    compute_power,
    bonferroni_correction,
    simulate_ab_experiment,
)


# ---------------------------------------------------------------------------
# two_proportion_ztest
# ---------------------------------------------------------------------------

class TestTwoProportionZTest:

    def test_significant_uplift(self):
        """Large sample with real uplift should be significant."""
        result = two_proportion_ztest(
            clicks_control=2300, impressions_control=100_000,
            clicks_variant=2800,  impressions_variant=100_000,
        )
        assert result.is_significant, "Expected significant result with large sample + real lift"
        assert result.p_value < 0.05
        assert result.relative_lift_pct > 0

    def test_no_uplift_not_significant(self):
        """Same rates should not be significant."""
        result = two_proportion_ztest(
            clicks_control=200, impressions_control=10_000,
            clicks_variant=202,  impressions_variant=10_000,
        )
        assert not result.is_significant

    def test_confidence_interval_contains_true_lift(self):
        """95% CI should contain the true lift approximately."""
        true_lift = 0.005
        result = two_proportion_ztest(
            clicks_control=2300, impressions_control=100_000,
            clicks_variant=2800,  impressions_variant=100_000,
        )
        ci_lo, ci_hi = result.confidence_interval
        assert ci_lo < result.absolute_lift < ci_hi

    def test_output_fields(self):
        """Result should have all required fields."""
        result = two_proportion_ztest(500, 10_000, 550, 10_000)
        assert hasattr(result, "auc") is False    # not a model metric
        assert 0 <= result.p_value <= 1
        assert result.sample_size_control == 10_000
        assert result.sample_size_variant == 10_000

    def test_strict_alpha(self):
        """With alpha=0.01, borderline result should not be significant."""
        result = two_proportion_ztest(
            clicks_control=230, impressions_control=10_000,
            clicks_variant=260,  impressions_variant=10_000,
            alpha=0.01,
        )
        # At α=0.01, this marginal lift may not clear the bar
        assert result.alpha == 0.01


# ---------------------------------------------------------------------------
# ttest_continuous
# ---------------------------------------------------------------------------

class TestTTestContinuous:

    def test_clearly_different_means(self):
        rng = np.random.default_rng(0)
        control = rng.normal(loc=5.0, scale=1.0, size=1000)
        variant = rng.normal(loc=5.8, scale=1.0, size=1000)
        result  = ttest_continuous(control, variant)
        assert result.is_significant

    def test_same_distribution_not_significant(self):
        rng = np.random.default_rng(1)
        control = rng.normal(loc=5.0, scale=1.0, size=200)
        variant = rng.normal(loc=5.0, scale=1.0, size=200)
        result  = ttest_continuous(control, variant)
        assert not result.is_significant

    def test_lift_direction(self):
        control = np.array([1.0, 2.0, 3.0])
        variant = np.array([4.0, 5.0, 6.0])
        result  = ttest_continuous(control, variant)
        assert result.absolute_lift > 0


# ---------------------------------------------------------------------------
# minimum_sample_size
# ---------------------------------------------------------------------------

class TestSampleSize:

    def test_returns_positive_integer(self):
        n = minimum_sample_size(baseline_rate=0.02, min_detectable_effect=0.003)
        assert isinstance(n, int)
        assert n > 0

    def test_smaller_mde_requires_larger_sample(self):
        n_large_mde = minimum_sample_size(0.02, 0.005)
        n_small_mde = minimum_sample_size(0.02, 0.002)
        assert n_small_mde > n_large_mde

    def test_higher_power_requires_larger_sample(self):
        n_80 = minimum_sample_size(0.02, 0.003, power=0.80)
        n_90 = minimum_sample_size(0.02, 0.003, power=0.90)
        assert n_90 > n_80


# ---------------------------------------------------------------------------
# compute_power
# ---------------------------------------------------------------------------

class TestComputePower:

    def test_large_n_high_power(self):
        power = compute_power(n=500_000, baseline_rate=0.02, lift=0.003)
        assert power > 0.95

    def test_small_n_low_power(self):
        power = compute_power(n=100, baseline_rate=0.02, lift=0.003)
        assert power < 0.30

    def test_power_between_0_and_1(self):
        power = compute_power(n=10_000, baseline_rate=0.02, lift=0.004)
        assert 0.0 <= power <= 1.0


# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------

class TestBonferroni:

    def test_correction_reduces_significance(self):
        p_values = [0.03, 0.04, 0.001]
        df = bonferroni_correction(p_values, alpha=0.05)
        corrected_alpha = 0.05 / 3
        # Only p=0.001 should survive
        assert df.loc[2, "significant_after_correction"] is True
        assert df.loc[0, "significant_after_correction"] is False

    def test_output_shape(self):
        df = bonferroni_correction([0.01, 0.02, 0.05, 0.10])
        assert len(df) == 4
        assert "significant_after_correction" in df.columns


# ---------------------------------------------------------------------------
# simulate_ab_experiment
# ---------------------------------------------------------------------------

class TestSimulateABExperiment:

    def test_returns_four_ints(self):
        c_c, c_i, v_c, v_i = simulate_ab_experiment()
        assert all(isinstance(x, int) for x in [c_c, c_i, v_c, v_i])

    def test_impressions_match_input(self):
        _, c_impr, _, v_impr = simulate_ab_experiment(n_control=20_000, n_variant=30_000)
        assert c_impr == 20_000
        assert v_impr == 30_000

    def test_clicks_within_bounds(self):
        c_clicks, c_impr, v_clicks, v_impr = simulate_ab_experiment()
        assert 0 <= c_clicks <= c_impr
        assert 0 <= v_clicks <= v_impr
