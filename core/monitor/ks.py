import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any


def calculate_ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov test between two distributions.
    
    The KS test compares two distributions and returns a statistic and p-value.
    The statistic measures the maximum difference between the empirical
    distribution functions of the two samples.
    
    Args:
        expected: Expected/baseline distribution
        actual: Actual/current distribution
    
    Returns:
        Tuple of (KS statistic, p-value)
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan, np.nan
    
    # Perform KS test
    statistic, p_value = stats.ks_2samp(expected, actual)
    
    return statistic, p_value


def calculate_ks_for_series(expected_series: pd.Series, actual_series: pd.Series) -> Tuple[float, float]:
    """Calculate KS test for pandas Series."""
    return calculate_ks_test(expected_series.values, actual_series.values)


def calculate_ks_for_dataframe(expected_df: pd.DataFrame, actual_df: pd.DataFrame,
                             columns: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate KS test for multiple columns in DataFrames.
    
    Args:
        expected_df: Expected/baseline DataFrame
        actual_df: Actual/current DataFrame
        columns: List of columns to calculate KS for (if None, use all numeric columns)
    
    Returns:
        Dictionary mapping column names to KS results
    """
    if columns is None:
        # Use all numeric columns
        columns = expected_df.select_dtypes(include=[np.number]).columns.tolist()
    
    ks_results = {}
    
    for col in columns:
        if col in expected_df.columns and col in actual_df.columns:
            try:
                statistic, p_value = calculate_ks_for_series(expected_df[col], actual_df[col])
                ks_results[col] = {
                    'statistic': statistic,
                    'p_value': p_value
                }
            except Exception as e:
                print(f"Error calculating KS test for column {col}: {e}")
                ks_results[col] = {
                    'statistic': np.nan,
                    'p_value': np.nan
                }
        else:
            print(f"Column {col} not found in both DataFrames")
            ks_results[col] = {
                'statistic': np.nan,
                'p_value': np.nan
            }
    
    return ks_results


def interpret_ks(statistic: float, p_value: float, alpha: float = 0.05) -> str:
    """
    Interpret KS test results.
    
    Args:
        statistic: KS statistic
        p_value: p-value
        alpha: Significance level
    
    Returns:
        Interpretation string
    """
    if np.isnan(statistic) or np.isnan(p_value):
        return "Unable to calculate"
    
    if p_value < alpha:
        return f"Distributions are significantly different (p={p_value:.4f})"
    else:
        return f"Distributions are not significantly different (p={p_value:.4f})"


def calculate_ks_distance(expected: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate KS distance (maximum difference between CDFs).
    
    This is a simpler metric than the full KS test that just returns
    the maximum difference between the empirical distribution functions.
    
    Args:
        expected: Expected/baseline distribution
        actual: Actual/current distribution
    
    Returns:
        KS distance
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Sort both arrays
    expected_sorted = np.sort(expected)
    actual_sorted = np.sort(actual)
    
    # Create combined sorted array
    combined = np.concatenate([expected_sorted, actual_sorted])
    combined = np.sort(combined)
    
    # Calculate empirical CDFs
    expected_cdf = np.searchsorted(expected_sorted, combined, side='right') / len(expected)
    actual_cdf = np.searchsorted(actual_sorted, combined, side='right') / len(actual)
    
    # Calculate maximum difference
    ks_distance = np.max(np.abs(expected_cdf - actual_cdf))
    
    return ks_distance


def main():
    """Example usage of KS test calculation."""
    # Generate sample data
    np.random.seed(42)
    
    # Baseline distribution (normal)
    expected = np.random.normal(0, 1, 1000)
    
    # Current distribution (slightly shifted)
    actual = np.random.normal(0.2, 1.1, 1000)
    
    # Calculate KS test
    statistic, p_value = calculate_ks_test(expected, actual)
    interpretation = interpret_ks(statistic, p_value)
    
    print(f"KS Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Interpretation: {interpretation}")
    
    # Calculate KS distance
    ks_distance = calculate_ks_distance(expected, actual)
    print(f"KS Distance: {ks_distance:.4f}")


if __name__ == '__main__':
    main()

