"""
ab_testing.py
Statistical A/B testing framework for online advertising experiments.

Covers:
  - Two-proportion z-test (CTR uplift significance)
  - T-test for continuous metrics (e.g. revenue per click)
  - Confidence interval computation
  - Statistical power and minimum detectable effect
  - Multiple comparison correction (Bonferroni)
  - Simulation of A/B experiments
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ABTestResult:
    """Container for A/B test outcome."""
    metric: str
    control_mean: float
    variant_mean: float
    absolute_lift: float
    relative_lift_pct: float
    p_value: float
    confidence_interval: tuple      # 95% CI on the lift
    is_significant: bool
    alpha: float
    test_type: str
    sample_size_control: int
    sample_size_variant: int

    def __str__(self):
        sig = "✓ SIGNIFICANT" if self.is_significant else "✗ not significant"
        return (
            f"\n{'='*55}\n"
            f"  A/B Test Result — {self.metric}\n"
            f"{'='*55}\n"
            f"  Control  : {self.control_mean:.4f}  (n={self.sample_size_control:,})\n"
            f"  Variant  : {self.variant_mean:.4f}  (n={self.sample_size_variant:,})\n"
            f"  Lift     : {self.absolute_lift:+.4f}  ({self.relative_lift_pct:+.2f}%)\n"
            f"  95% CI   : [{self.confidence_interval[0]:+.4f}, {self.confidence_interval[1]:+.4f}]\n"
            f"  p-value  : {self.p_value:.4f}\n"
            f"  Result   : {sig} (α={self.alpha})\n"
            f"{'='*55}\n"
        )


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

def two_proportion_ztest(
    clicks_control: int, impressions_control: int,
    clicks_variant: int, impressions_variant: int,
    alpha: float = 0.05,
    metric_name: str = "CTR",
) -> ABTestResult:
    """
    Two-proportion z-test: test whether CTR of variant differs from control.

    Null hypothesis H0: p_variant = p_control
    Alternative    H1: p_variant ≠ p_control  (two-sided)

    Parameters
    ----------
    clicks_control / variant    : number of clicks observed
    impressions_control/variant : total impressions served
    alpha                       : significance level (default 0.05)
    """
    p_c = clicks_control  / impressions_control
    p_v = clicks_variant  / impressions_variant
    n_c = impressions_control
    n_v = impressions_variant

    # Pooled proportion under H0
    p_pool = (clicks_control + clicks_variant) / (n_c + n_v)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_v))

    z_stat = (p_v - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))    # two-sided

    # 95% CI on the lift (using unpooled SE)
    se_lift = np.sqrt(p_c*(1-p_c)/n_c + p_v*(1-p_v)/n_v)
    z_crit  = stats.norm.ppf(1 - alpha/2)
    ci = (
        round((p_v - p_c) - z_crit * se_lift, 6),
        round((p_v - p_c) + z_crit * se_lift, 6),
    )

    return ABTestResult(
        metric=metric_name,
        control_mean=round(p_c, 6),
        variant_mean=round(p_v, 6),
        absolute_lift=round(p_v - p_c, 6),
        relative_lift_pct=round((p_v - p_c) / p_c * 100, 3),
        p_value=round(p_value, 6),
        confidence_interval=ci,
        is_significant=p_value < alpha,
        alpha=alpha,
        test_type="two-proportion z-test",
        sample_size_control=n_c,
        sample_size_variant=n_v,
    )


def ttest_continuous(
    control_values: np.ndarray,
    variant_values: np.ndarray,
    alpha: float = 0.05,
    metric_name: str = "Revenue per click",
) -> ABTestResult:
    """
    Welch's t-test for continuous metrics (e.g., revenue per user, session duration).
    Does NOT assume equal variance between groups.
    """
    t_stat, p_value = stats.ttest_ind(variant_values, control_values, equal_var=False)

    mean_c = control_values.mean()
    mean_v = variant_values.mean()
    lift    = mean_v - mean_c

    # 95% CI on the difference
    se = np.sqrt(control_values.var(ddof=1)/len(control_values) +
                 variant_values.var(ddof=1)/len(variant_values))
    df_approx = len(control_values) + len(variant_values) - 2
    t_crit = stats.t.ppf(1 - alpha/2, df=df_approx)
    ci = (round(lift - t_crit * se, 4), round(lift + t_crit * se, 4))

    return ABTestResult(
        metric=metric_name,
        control_mean=round(mean_c, 4),
        variant_mean=round(mean_v, 4),
        absolute_lift=round(lift, 4),
        relative_lift_pct=round(lift / mean_c * 100, 3) if mean_c != 0 else 0.0,
        p_value=round(p_value, 6),
        confidence_interval=ci,
        is_significant=p_value < alpha,
        alpha=alpha,
        test_type="Welch t-test",
        sample_size_control=len(control_values),
        sample_size_variant=len(variant_values),
    )


# ---------------------------------------------------------------------------
# Statistical power & sample size
# ---------------------------------------------------------------------------

def minimum_sample_size(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Compute the minimum sample size per group needed to detect a given
    absolute lift in CTR with the desired statistical power.

    Parameters
    ----------
    baseline_rate          : current CTR of control (e.g. 0.02 = 2%)
    min_detectable_effect  : smallest absolute lift we care about (e.g. 0.003)
    alpha                  : Type I error rate (default 0.05)
    power                  : 1 - Type II error rate (default 0.80)
    """
    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta  = stats.norm.ppf(power)

    p_bar = (p1 + p2) / 2
    n = (
        (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
         z_beta  * np.sqrt(p1 * (1-p1) + p2 * (1-p2))) ** 2
        / (p2 - p1) ** 2
    )
    return int(np.ceil(n))


def compute_power(
    n: int,
    baseline_rate: float,
    lift: float,
    alpha: float = 0.05,
) -> float:
    """
    Compute the statistical power of a completed or planned test given n per group.
    """
    p1 = baseline_rate
    p2 = baseline_rate + lift
    p_bar = (p1 + p2) / 2
    z_alpha = stats.norm.ppf(1 - alpha/2)

    se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n)
    se_alt  = np.sqrt((p1*(1-p1) + p2*(1-p2)) / n)

    z = (abs(p2 - p1) - z_alpha * se_null) / se_alt
    return round(stats.norm.cdf(z), 4)


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------

def bonferroni_correction(p_values: list, alpha: float = 0.05) -> pd.DataFrame:
    """
    Apply Bonferroni correction to a list of p-values.
    Returns a DataFrame showing which tests remain significant after correction.
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    return pd.DataFrame({
        "p_value":        p_values,
        "corrected_alpha": [corrected_alpha] * n,
        "significant_after_correction": [p < corrected_alpha for p in p_values],
    })


# ---------------------------------------------------------------------------
# Simulation: generate synthetic A/B experiment data
# ---------------------------------------------------------------------------

def simulate_ab_experiment(
    n_control: int = 50_000,
    n_variant: int = 50_000,
    baseline_ctr: float = 0.023,
    true_lift: float = 0.004,
    seed: int = 42,
) -> tuple:
    """
    Simulate click data for an A/B test on two ad creatives.

    Returns
    -------
    control_clicks, control_impressions, variant_clicks, variant_impressions
    as integers (mimicking a production ad serving log).
    """
    rng = np.random.default_rng(seed)
    control_clicks = int(rng.binomial(n_control, baseline_ctr))
    variant_clicks = int(rng.binomial(n_variant, baseline_ctr + true_lift))
    return control_clicks, n_control, variant_clicks, n_variant


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Scenario 1: significant uplift ---
    c_clicks, c_impr, v_clicks, v_impr = simulate_ab_experiment(
        n_control=100_000, n_variant=100_000, baseline_ctr=0.023, true_lift=0.004
    )
    result = two_proportion_ztest(c_clicks, c_impr, v_clicks, v_impr)
    print(result)

    # --- Scenario 2: no real uplift ---
    c_clicks2, c_impr2, v_clicks2, v_impr2 = simulate_ab_experiment(
        n_control=5_000, n_variant=5_000, baseline_ctr=0.023, true_lift=0.0005
    )
    result2 = two_proportion_ztest(c_clicks2, c_impr2, v_clicks2, v_impr2)
    print(result2)

    # --- Sample size calculator ---
    n_needed = minimum_sample_size(baseline_rate=0.023, min_detectable_effect=0.003)
    print(f"Sample size needed per group: {n_needed:,}")

    # --- Power of scenario 1 ---
    pwr = compute_power(n=100_000, baseline_rate=0.023, lift=0.004)
    print(f"Power of scenario 1: {pwr:.2%}")
