import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    PSI measures the stability of a population over time by comparing
    the distribution of a variable between two time periods.
    
    Args:
        expected: Expected/baseline distribution
        actual: Actual/current distribution
        bins: Number of bins for discretization
    
    Returns:
        PSI value (0-0.1: no change, 0.1-0.2: moderate change, >0.2: significant change)
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Create bins based on expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    
    # Handle edge case where all values are the same
    if min_val == max_val:
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calculate histograms
    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    expected_hist = expected_hist + epsilon
    actual_hist = actual_hist + epsilon
    
    # Normalize to probabilities
    expected_prob = expected_hist / expected_hist.sum()
    actual_prob = actual_hist / actual_hist.sum()
    
    # Calculate PSI
    psi = np.sum((actual_prob - expected_prob) * np.log(actual_prob / expected_prob))
    
    return psi


def calculate_psi_for_series(expected_series: pd.Series, actual_series: pd.Series, 
                           bins: int = 10) -> float:
    """Calculate PSI for pandas Series."""
    return calculate_psi(expected_series.values, actual_series.values, bins)


def calculate_psi_for_dataframe(expected_df: pd.DataFrame, actual_df: pd.DataFrame,
                              columns: List[str] = None, bins: int = 10) -> Dict[str, float]:
    """
    Calculate PSI for multiple columns in DataFrames.
    
    Args:
        expected_df: Expected/baseline DataFrame
        actual_df: Actual/current DataFrame
        columns: List of columns to calculate PSI for (if None, use all numeric columns)
        bins: Number of bins for discretization
    
    Returns:
        Dictionary mapping column names to PSI values
    """
    if columns is None:
        # Use all numeric columns
        columns = expected_df.select_dtypes(include=[np.number]).columns.tolist()
    
    psi_results = {}
    
    for col in columns:
        if col in expected_df.columns and col in actual_df.columns:
            try:
                psi_val = calculate_psi_for_series(expected_df[col], actual_df[col], bins)
                psi_results[col] = psi_val
            except Exception as e:
                print(f"Error calculating PSI for column {col}: {e}")
                psi_results[col] = np.nan
        else:
            print(f"Column {col} not found in both DataFrames")
            psi_results[col] = np.nan
    
    return psi_results


def interpret_psi(psi_value: float) -> str:
    """
    Interpret PSI value.
    
    Args:
        psi_value: PSI value
    
    Returns:
        Interpretation string
    """
    if np.isnan(psi_value):
        return "Unable to calculate"
    elif psi_value < 0.1:
        return "No significant change"
    elif psi_value < 0.2:
        return "Moderate change"
    else:
        return "Significant change"


def main():
    """Example usage of PSI calculation."""
    # Generate sample data
    np.random.seed(42)
    
    # Baseline distribution (normal)
    expected = np.random.normal(0, 1, 1000)
    
    # Current distribution (slightly shifted)
    actual = np.random.normal(0.2, 1.1, 1000)
    
    # Calculate PSI
    psi = calculate_psi(expected, actual)
    interpretation = interpret_psi(psi)
    
    print(f"PSI: {psi:.4f}")
    print(f"Interpretation: {interpretation}")


if __name__ == '__main__':
    main()

