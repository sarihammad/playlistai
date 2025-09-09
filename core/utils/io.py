import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_parquet(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """Save DataFrame to parquet file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)


def load_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from parquet file."""
    return pd.read_parquet(filepath)


def save_numpy(array: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save numpy array to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, array)


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """Load numpy array from file."""
    return np.load(filepath)


def ensure_dir(filepath: Union[str, Path]) -> Path:
    """Ensure directory exists for filepath."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath

