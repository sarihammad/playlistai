import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, List
from sklearn.model_selection import train_test_split


def temporal_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally by user.
    
    For each user, split their sessions chronologically:
    - Train: earliest sessions
    - Val: middle sessions  
    - Test: latest sessions
    """
    print("Performing temporal split...")
    
    train_sessions = []
    val_sessions = []
    test_sessions = []
    
    for user_id, user_data in df.groupby('user_id'):
        user_data = user_data.sort_values('session_id').reset_index(drop=True)
        n_sessions = len(user_data)
        
        if n_sessions < 3:
            # If user has too few sessions, put all in train
            train_sessions.append(user_data)
            continue
        
        # Calculate split points
        n_test = max(1, int(n_sessions * test_size))
        n_val = max(1, int(n_sessions * val_size))
        n_train = n_sessions - n_val - n_test
        
        # Split chronologically
        train_sessions.append(user_data.iloc[:n_train])
        val_sessions.append(user_data.iloc[n_train:n_train + n_val])
        test_sessions.append(user_data.iloc[n_train + n_val:])
    
    train_df = pd.concat(train_sessions, ignore_index=True)
    val_df = pd.concat(val_sessions, ignore_index=True)
    test_df = pd.concat(test_sessions, ignore_index=True)
    
    return train_df, val_df, test_df


def random_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, 
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data randomly by sessions."""
    print("Performing random split...")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), random_state=random_state
    )
    
    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df, test_size=(test_size / (val_size + test_size)), random_state=random_state
    )
    
    return train_df, val_df, test_df


def user_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, 
              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by users (cold start scenario)."""
    print("Performing user-based split...")
    
    # Get unique users
    users = df['user_id'].unique()
    
    # Split users
    train_users, temp_users = train_test_split(
        users, test_size=(val_size + test_size), random_state=random_state
    )
    
    val_users, test_users = train_test_split(
        temp_users, test_size=(test_size / (val_size + test_size)), random_state=random_state
    )
    
    # Split data by user groups
    train_df = df[df['user_id'].isin(train_users)]
    val_df = df[df['user_id'].isin(val_users)]
    test_df = df[df['user_id'].isin(test_users)]
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Split sequences into train/val/test')
    parser.add_argument('--in', dest='input_dir', type=str, default='data/processed', help='Input directory')
    parser.add_argument('--out', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--method', type=str, default='temporal', choices=['temporal', 'random', 'user'], 
                       help='Split method')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_dir)
    df = pd.read_parquet(input_path / 'sequences.parquet')
    
    print(f"Loaded {len(df)} sequences")
    print(f"Users: {df['user_id'].nunique()}")
    
    # Split data
    if args.method == 'temporal':
        train_df, val_df, test_df = temporal_split(df, args.test_size, args.val_size)
    elif args.method == 'random':
        train_df, val_df, test_df = random_split(df, args.test_size, args.val_size, args.random_state)
    elif args.method == 'user':
        train_df, val_df, test_df = user_split(df, args.test_size, args.val_size, args.random_state)
    
    # Save splits
    output_path = Path(args.out)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(output_path / 'train.parquet', index=False)
    val_df.to_parquet(output_path / 'val.parquet', index=False)
    test_df.to_parquet(output_path / 'test.parquet', index=False)
    
    print(f"Split results:")
    print(f"  Train: {len(train_df)} sequences ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} sequences ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} sequences ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Train users: {train_df['user_id'].nunique()}")
    print(f"  Val users:   {val_df['user_id'].nunique()}")
    print(f"  Test users:  {test_df['user_id'].nunique()}")
    
    print(f"Saved splits to: {output_path}")


if __name__ == '__main__':
    main()

