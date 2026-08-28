"""
Main experiment runner for Vendi topic reduction evaluation.

This script orchestrates all experimental protocols using YAML configuration files. All parameters are specified in a YAML config file for simplicity and reproducibility.

Usage:
    # Run with custom config
    python experiments/vendi_experiments.py --config experiments/configs/quick_test.yaml
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils.data_loaders import prepare_dataset, print_dataset_stats
from experiments.protocols import run_protocol_1, run_protocol_2, run_protocol_3, run_protocol_4, run_protocol_5, run_protocol_6

AVAILABLE_PROTOCOLS = {
    "p1": {
        "name": "Protocol 1: Fixed-k Comparison (2×2 Matrix)",
        "function": run_protocol_1,
        "description": "Compare Vendi vs Agglomerative across c-TF-IDF and SBERT embeddings",
    },
    "p2": {
        "name": "Protocol 2: VTR vs. LLM-Assisted Reduction",
        "function": run_protocol_2,
        "description": "Compare VTR against iterative LLM-based merging (cost vs quality analysis)",
    },
    "p3": {
        "name": "Protocol 3: VTR vs. Direct HDBSCAN Tuning",
        "function": run_protocol_3,
        "description": "Compare outlier assignment rates between VTR and direct HDBSCAN parameter tuning",
    },
    "p4": {
        "name": "Protocol 4: Robustness & Sensitivity Analysis",
        "function": run_protocol_4,
        "description": "Test robustness across multiple initial HDBSCAN configurations",
    },
    "p5": {
        "name": "Protocol 5: Reduction Trajectory",
        "function": run_protocol_5,
        "description": "Merge one topic at a time from initial k to 1, recording VS2, NPMI, C_v and TD at every step",
    },
    "p6": {
        "name": "Protocol 6: q-Value Sensitivity Ablation",
        "function": run_protocol_6,
        "description": "Ablation study: how does the Vendi Score order q affect topic reduction quality?",
    },
}

def load_config(config_path: str) -> dict:
    """
    Load experiment configuration from YAML file.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def validate_config(config: dict) -> None:
    """
    Validate configuration dictionary.
    """
    # Check required top-level keys
    required_keys = ['dataset', 'protocol', 'output']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: '{key}'")
    
    # Check protocol is valid
    protocol = config['protocol']
    if protocol not in AVAILABLE_PROTOCOLS:
        available = ', '.join(AVAILABLE_PROTOCOLS.keys())
        raise ValueError(f"Unknown protocol '{protocol}'. Available: {available}")
    
    # Check protocol-specific parameters exist
    protocol_key = f"protocol_{protocol.replace('p', '')}"
    if protocol_key not in config:
        raise ValueError(f"Missing protocol parameters: '{protocol_key}'")

def print_config(config: dict):
    """Pretty print configuration."""
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Protocol: {config['protocol']}")
    print(f"Dataset: {config['dataset']['name']}")
    if config['dataset'].get('n_samples'):
        print(f"Samples: {config['dataset']['n_samples']}")
    else:
        print("Samples: all")
    print(f"Output dir: {config['output']['dir']}")
    
    # Print protocol-specific parameters
    protocol_key = f"protocol_{config['protocol'].replace('p', '')}"
    if protocol_key in config:
        print(f"\n{config['protocol'].upper()} Parameters:")
        for key, value in config[protocol_key].items():
            print(f"  {key}: {value}")
    
    print("="*70 + "\n")


def run_experiment_from_config(config: dict):
    """
    Run experiment using configuration dictionary.
    
    Args:
        config: Configuration dictionary loaded from YAML
        
    Returns:
        pandas.DataFrame with results
    """
    # Validate configuration
    validate_config(config)
    
    # Print configuration
    print("\n" + "="*70)
    print("VENDI TOPIC REDUCTION EXPERIMENTS")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_config(config)
    
    # Extract parameters
    protocol = config['protocol']
    dataset_config = config['dataset']
    output_config = config['output']
    output_dir = output_config['dir']
    output_filename = output_config.get('filename')
    
    # Get protocol-specific parameters
    protocol_key = f"protocol_{protocol.replace('p', '')}"
    protocol_params = config[protocol_key].copy()
    
    # Add output configuration to protocol params
    protocol_params['save_dir'] = output_dir
    if output_filename:
        protocol_params['output_filename'] = output_filename
    
    # Load dataset
    print("Loading dataset...")
    dataset = prepare_dataset(
        name=dataset_config['name'],
        n_samples=dataset_config.get('n_samples')
    )
    print_dataset_stats(dataset)
    
    # Get protocol function
    protocol_info = AVAILABLE_PROTOCOLS[protocol]
    print(f"\nRunning: {protocol_info['name']}")
    print(f"Description: {protocol_info['description']}\n")
    
    protocol_function = protocol_info['function']
    
    # Run protocol
    try:
        results = protocol_function(dataset, **protocol_params)
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Results shape: {results.shape}")
        print(f"Output directory: {output_dir}")
        
        # Save metadata
        if config['output'].get('save_metadata', True):
            save_experiment_metadata(config, results, output_dir)
        
        return results
        
    except Exception as e:
        print("\n" + "="*70)
        print("EXPERIMENT FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_experiment_metadata(config: dict, results, output_dir: str):
    """Save experiment metadata."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results_shape": list(results.shape) if results is not None else None,
        "status": "completed" if results is not None else "failed"
    }
    
    protocol = config['protocol']
    output_path = Path(output_dir) / f"{protocol}_metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {output_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Vendi topic reduction experiments using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python experiments/vendi_experiments.py
  
  # Run with custom config
  python experiments/vendi_experiments.py --config my_experiment.yaml
    """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/experiment_config.yaml",
        help="Path to YAML configuration file (default: experiments/configs/experiment_config.yaml)"
    )    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()

    # Run experiment with config file
    try:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        run_experiment_from_config(config)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure the config file exists at the specified path.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
