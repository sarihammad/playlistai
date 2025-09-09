import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any
import random


def generate_synthetic_data(n_users: int = 10000, n_tracks: int = 50000, 
                          n_interactions: int = 500000) -> pd.DataFrame:
    """Generate synthetic Last.fm-like interaction data."""
    print(f"Generating synthetic data: {n_users} users, {n_tracks} tracks, {n_interactions} interactions")
    
    # Generate user IDs
    user_ids = [f"user_{i:06d}" for i in range(n_users)]
    
    # Generate track IDs with some popularity bias
    track_ids = [f"track_{i:06d}" for i in range(n_tracks)]
    
    # Generate artists and genres
    artists = [f"artist_{i:04d}" for i in range(1000)]
    genres = ["pop", "rock", "hip_hop", "electronic", "jazz", "classical", "country", "blues"]
    
    # Create popularity distribution (power law)
    track_popularity = np.random.pareto(1.2, n_tracks)
    track_popularity = track_popularity / track_popularity.sum()
    
    # Generate interactions
    interactions = []
    
    for _ in range(n_interactions):
        user_id = random.choice(user_ids)
        track_id = np.random.choice(track_ids, p=track_popularity)
        artist = random.choice(artists)
        genre = random.choice(genres)
        
        # Generate timestamp (last 2 years)
        timestamp = pd.Timestamp.now() - pd.Timedelta(days=random.randint(0, 730))
        
        interactions.append({
            'user_id': user_id,
            'track_id': track_id,
            'timestamp': timestamp,
            'artist': artist,
            'genre': genre
        })
    
    df = pd.DataFrame(interactions)
    
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic interaction data')
    parser.add_argument('--out', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--n_users', type=int, default=10000, help='Number of users')
    parser.add_argument('--n_tracks', type=int, default=50000, help='Number of tracks')
    parser.add_argument('--n_interactions', type=int, default=500000, help='Number of interactions')
    
    args = parser.parse_args()
    
    # Generate data
    df = generate_synthetic_data(args.n_users, args.n_tracks, args.n_interactions)
    
    # Save to parquet
    output_path = Path(args.out)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path / 'interactions.parquet', index=False)
    
    print(f"Generated {len(df)} interactions")
    print(f"Users: {df['user_id'].nunique()}")
    print(f"Tracks: {df['track_id'].nunique()}")
    print(f"Artists: {df['artist'].nunique()}")
    print(f"Genres: {df['genre'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Saved to: {output_path / 'interactions.parquet'}")


if __name__ == '__main__':
    main()

