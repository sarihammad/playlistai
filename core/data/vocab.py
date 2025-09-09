import pandas as pd
import json
from pathlib import Path
import argparse
from collections import Counter
from typing import Dict, List, Set


def build_vocab(df: pd.DataFrame, min_freq: int = 5) -> Dict[str, Any]:
    """Build vocabulary from track sequences."""
    print(f"Building vocabulary with min_freq={min_freq}...")
    
    # Count track frequencies
    track_counts = Counter()
    
    for track_seq in df['track_seq']:
        track_counts.update(track_seq)
    
    print(f"Total unique tracks: {len(track_counts)}")
    print(f"Tracks with freq >= {min_freq}: {sum(1 for count in track_counts.values() if count >= min_freq)}")
    
    # Create vocabulary
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2
    }
    
    # Add tracks that meet frequency threshold
    track_to_id = vocab.copy()
    id_to_track = {v: k for k, v in vocab.items()}
    
    for track, count in track_counts.items():
        if count >= min_freq:
            track_id = len(track_to_id)
            track_to_id[track] = track_id
            id_to_track[track_id] = track
    
    # Create frequency mapping
    track_freqs = {track: count for track, count in track_counts.items() if count >= min_freq}
    
    vocab_data = {
        'track_to_id': track_to_id,
        'id_to_track': id_to_track,
        'track_freqs': track_freqs,
        'vocab_size': len(track_to_id),
        'min_freq': min_freq,
        'total_tracks': len(track_counts),
        'filtered_tracks': len(track_freqs)
    }
    
    return vocab_data


def create_artist_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Create track to artist mapping."""
    print("Creating artist mapping...")
    
    track_to_artist = {}
    
    for _, row in df.iterrows():
        for track, artist in zip(row['track_seq'], row['artist_seq']):
            if track not in track_to_artist:
                track_to_artist[track] = artist
    
    return track_to_artist


def create_genre_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Create track to genre mapping."""
    print("Creating genre mapping...")
    
    track_to_genre = {}
    
    for _, row in df.iterrows():
        for track, genre in zip(row['track_seq'], row['genre_seq']):
            if track not in track_to_genre:
                track_to_genre[track] = genre
    
    return track_to_genre


def main():
    parser = argparse.ArgumentParser(description='Build vocabulary from sequences')
    parser.add_argument('--in', dest='input_dir', type=str, default='data/processed', help='Input directory')
    parser.add_argument('--out', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--min_freq', type=int, default=5, help='Minimum frequency for tracks')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_dir)
    df = pd.read_parquet(input_path / 'sequences.parquet')
    
    print(f"Loaded {len(df)} sequences")
    
    # Build vocabulary
    vocab_data = build_vocab(df, args.min_freq)
    
    # Create mappings
    artist_mapping = create_artist_mapping(df)
    genre_mapping = create_genre_mapping(df)
    
    # Add mappings to vocab data
    vocab_data['track_to_artist'] = artist_mapping
    vocab_data['track_to_genre'] = genre_mapping
    
    # Save vocabulary
    output_path = Path(args.out)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main vocabulary
    with open(output_path / 'vocab.json', 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    # Save frequencies separately
    with open(output_path / 'freqs.json', 'w') as f:
        json.dump(vocab_data['track_freqs'], f, indent=2)
    
    print(f"Vocabulary size: {vocab_data['vocab_size']}")
    print(f"Artist mappings: {len(artist_mapping)}")
    print(f"Genre mappings: {len(genre_mapping)}")
    print(f"Saved vocab to: {output_path / 'vocab.json'}")
    print(f"Saved freqs to: {output_path / 'freqs.json'}")


if __name__ == '__main__':
    main()

