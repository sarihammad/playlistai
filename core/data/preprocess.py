import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def sessionize_interactions(df: pd.DataFrame, gap_minutes: int = 45) -> pd.DataFrame:
    """Convert interactions to sessions based on time gaps."""
    print(f"Sessionizing interactions with {gap_minutes} minute gap...")
    
    sessions = []
    
    for user_id, user_data in df.groupby('user_id'):
        user_data = user_data.sort_values('timestamp').reset_index(drop=True)
        
        session_id = 0
        last_timestamp = None
        
        for _, row in user_data.iterrows():
            current_timestamp = row['timestamp']
            
            # Start new session if gap is too large or first interaction
            if (last_timestamp is None or 
                (current_timestamp - last_timestamp).total_seconds() > gap_minutes * 60):
                session_id += 1
            
            sessions.append({
                'user_id': user_id,
                'session_id': f"{user_id}_session_{session_id}",
                'track_id': row['track_id'],
                'timestamp': current_timestamp,
                'artist': row['artist'],
                'genre': row['genre']
            })
            
            last_timestamp = current_timestamp
    
    return pd.DataFrame(sessions)


def dedupe_immediate_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Remove immediate consecutive repeats within sessions."""
    print("Removing immediate consecutive repeats...")
    
    deduped = []
    
    for session_id, session_data in df.groupby('session_id'):
        session_data = session_data.sort_values('timestamp').reset_index(drop=True)
        
        # Keep first occurrence of each track in sequence
        last_track = None
        for _, row in session_data.iterrows():
            if row['track_id'] != last_track:
                deduped.append(row.to_dict())
                last_track = row['track_id']
    
    return pd.DataFrame(deduped)


def truncate_sequences(df: pd.DataFrame, max_len: int = 50) -> pd.DataFrame:
    """Truncate sequences to maximum length, keeping most recent items."""
    print(f"Truncating sequences to max length {max_len}...")
    
    truncated = []
    
    for session_id, session_data in df.groupby('session_id'):
        session_data = session_data.sort_values('timestamp').reset_index(drop=True)
        
        # Keep last max_len items
        if len(session_data) > max_len:
            session_data = session_data.tail(max_len)
        
        truncated.append(session_data)
    
    return pd.concat(truncated, ignore_index=True)


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add context features like hour of day and day of week."""
    print("Adding context features...")
    
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    return df


def create_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Create sequence representations for each session."""
    print("Creating sequence representations...")
    
    sequences = []
    
    for session_id, session_data in df.groupby('session_id'):
        session_data = session_data.sort_values('timestamp').reset_index(drop=True)
        
        sequences.append({
            'user_id': session_data['user_id'].iloc[0],
            'session_id': session_id,
            'track_seq': session_data['track_id'].tolist(),
            'ts_seq': session_data['timestamp'].tolist(),
            'ctx_seq': list(zip(session_data['hour'], session_data['dow'])),
            'artist_seq': session_data['artist'].tolist(),
            'genre_seq': session_data['genre'].tolist(),
            'seq_len': len(session_data)
        })
    
    return pd.DataFrame(sequences)


def filter_short_sessions(df: pd.DataFrame, min_len: int = 2) -> pd.DataFrame:
    """Filter out sessions that are too short."""
    print(f"Filtering sessions with length < {min_len}...")
    
    return df[df['seq_len'] >= min_len].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Preprocess interaction data')
    parser.add_argument('--in', dest='input_dir', type=str, default='data/raw', help='Input directory')
    parser.add_argument('--out', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--gap', type=int, default=45, help='Session gap in minutes')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--min_len', type=int, default=2, help='Minimum sequence length')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_dir)
    df = pd.read_parquet(input_path / 'interactions.parquet')
    
    print(f"Loaded {len(df)} interactions")
    print(f"Users: {df['user_id'].nunique()}")
    print(f"Tracks: {df['track_id'].nunique()}")
    
    # Process data
    df = sessionize_interactions(df, args.gap)
    df = dedupe_immediate_repeats(df)
    df = add_context_features(df)
    df = truncate_sequences(df, args.max_len)
    df = create_sequences(df)
    df = filter_short_sessions(df, args.min_len)
    
    # Save processed data
    output_path = Path(args.out)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path / 'sequences.parquet', index=False)
    
    print(f"Processed {len(df)} sessions")
    print(f"Average sequence length: {df['seq_len'].mean():.2f}")
    print(f"Sequence length range: {df['seq_len'].min()} - {df['seq_len'].max()}")
    print(f"Saved to: {output_path / 'sequences.parquet'}")


if __name__ == '__main__':
    main()

