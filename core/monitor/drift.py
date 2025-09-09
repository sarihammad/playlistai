import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from .psi import calculate_psi_for_dataframe, interpret_psi
from .ks import calculate_ks_for_dataframe, interpret_ks
from core.utils.io import save_json, save_parquet


class DriftDetector:
    """Drift detection for playlist data."""
    
    def __init__(self, baseline_path: str, current_path: str):
        """
        Initialize drift detector.
        
        Args:
            baseline_path: Path to baseline data (parquet or json)
            current_path: Path to current data (parquet or json)
        """
        self.baseline_path = baseline_path
        self.current_path = current_path
        self.baseline_data = None
        self.current_data = None
        
    def load_data(self):
        """Load baseline and current data."""
        # Load baseline data
        if self.baseline_path.endswith('.parquet'):
            self.baseline_data = pd.read_parquet(self.baseline_path)
        elif self.baseline_path.endswith('.json'):
            with open(self.baseline_path, 'r') as f:
                baseline_dict = json.load(f)
            self.baseline_data = pd.DataFrame(baseline_dict)
        else:
            raise ValueError(f"Unsupported file format: {self.baseline_path}")
        
        # Load current data
        if self.current_path.endswith('.parquet'):
            self.current_data = pd.read_parquet(self.current_path)
        elif self.current_path.endswith('.json'):
            with open(self.current_path, 'r') as f:
                current_dict = json.load(f)
            self.current_data = pd.DataFrame(current_dict)
        else:
            raise ValueError(f"Unsupported file format: {self.current_path}")
        
        print(f"Loaded baseline data: {len(self.baseline_data)} rows")
        print(f"Loaded current data: {len(self.current_data)} rows")
    
    def detect_drift(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect drift between baseline and current data.
        
        Args:
            columns: List of columns to analyze (if None, use all numeric columns)
        
        Returns:
            Dictionary containing drift analysis results
        """
        if self.baseline_data is None or self.current_data is None:
            self.load_data()
        
        # Determine columns to analyze
        if columns is None:
            numeric_cols = self.baseline_data.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in numeric_cols if col in self.current_data.columns]
        
        print(f"Analyzing drift for columns: {columns}")
        
        # Calculate PSI
        psi_results = calculate_psi_for_dataframe(
            self.baseline_data, self.current_data, columns
        )
        
        # Calculate KS test
        ks_results = calculate_ks_for_dataframe(
            self.baseline_data, self.current_data, columns
        )
        
        # Compile results
        drift_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'baseline_path': self.baseline_path,
            'current_path': self.current_path,
            'baseline_size': len(self.baseline_data),
            'current_size': len(self.current_data),
            'columns_analyzed': columns,
            'psi_results': psi_results,
            'ks_results': ks_results,
            'drift_summary': {}
        }
        
        # Create summary
        for col in columns:
            psi_val = psi_results.get(col, np.nan)
            ks_stat = ks_results.get(col, {}).get('statistic', np.nan)
            ks_pval = ks_results.get(col, {}).get('p_value', np.nan)
            
            drift_results['drift_summary'][col] = {
                'psi': psi_val,
                'psi_interpretation': interpret_psi(psi_val),
                'ks_statistic': ks_stat,
                'ks_p_value': ks_pval,
                'ks_interpretation': interpret_ks(ks_stat, ks_pval),
                'drift_detected': (
                    (not np.isnan(psi_val) and psi_val > 0.2) or
                    (not np.isnan(ks_pval) and ks_pval < 0.05)
                )
            }
        
        return drift_results
    
    def generate_report(self, drift_results: Dict[str, Any], output_path: str):
        """Generate drift detection report."""
        # Save detailed results
        save_json(drift_results, output_path)
        
        # Create summary report
        summary = []
        summary.append("=== DRIFT DETECTION REPORT ===")
        summary.append(f"Timestamp: {drift_results['timestamp']}")
        summary.append(f"Baseline: {drift_results['baseline_size']} samples")
        summary.append(f"Current: {drift_results['current_size']} samples")
        summary.append("")
        
        summary.append("DRIFT ANALYSIS:")
        for col, results in drift_results['drift_summary'].items():
            summary.append(f"\n{col}:")
            summary.append(f"  PSI: {results['psi']:.4f} ({results['psi_interpretation']})")
            summary.append(f"  KS: {results['ks_statistic']:.4f} (p={results['ks_p_value']:.4f})")
            summary.append(f"  Drift Detected: {results['drift_detected']}")
        
        # Print summary
        print("\n".join(summary))
        
        # Save summary to file
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary))
        
        print(f"\nDetailed results saved to: {output_path}")
        print(f"Summary saved to: {summary_path}")


def create_baseline_stats(sequences_path: str, output_path: str):
    """Create baseline statistics from training data."""
    print(f"Creating baseline statistics from {sequences_path}")
    
    # Load sequences
    df = pd.read_parquet(sequences_path)
    
    # Calculate statistics
    stats = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_sequences': len(df),
        'unique_users': df['user_id'].nunique(),
        'avg_seq_len': df['seq_len'].mean(),
        'std_seq_len': df['seq_len'].std(),
        'min_seq_len': df['seq_len'].min(),
        'max_seq_len': df['seq_len'].max(),
    }
    
    # Track popularity distribution
    all_tracks = []
    for track_seq in df['track_seq']:
        all_tracks.extend(track_seq)
    
    track_counts = pd.Series(all_tracks).value_counts()
    stats['track_popularity'] = {
        'mean': track_counts.mean(),
        'std': track_counts.std(),
        'min': track_counts.min(),
        'max': track_counts.max(),
        'unique_tracks': len(track_counts)
    }
    
    # Context distribution (if available)
    if 'ctx_seq' in df.columns:
        all_hours = []
        all_dows = []
        for ctx_seq in df['ctx_seq']:
            for hour, dow in ctx_seq:
                all_hours.append(hour)
                all_dows.append(dow)
        
        stats['context_distribution'] = {
            'hour_mean': np.mean(all_hours),
            'hour_std': np.std(all_hours),
            'dow_mean': np.mean(all_dows),
            'dow_std': np.std(all_dows)
        }
    
    # Save baseline stats
    save_json(stats, output_path)
    print(f"Baseline statistics saved to: {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Detect drift in playlist data')
    parser.add_argument('--baseline', type=str, required=True, help='Baseline data path')
    parser.add_argument('--current', type=str, required=True, help='Current data path')
    parser.add_argument('--output', type=str, default='data/reports/drift_report.json', help='Output path')
    parser.add_argument('--columns', nargs='+', help='Columns to analyze (default: all numeric)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Create drift detector
    detector = DriftDetector(args.baseline, args.current)
    
    # Detect drift
    drift_results = detector.detect_drift(args.columns)
    
    # Generate report
    detector.generate_report(drift_results, args.output)


if __name__ == '__main__':
    main()

