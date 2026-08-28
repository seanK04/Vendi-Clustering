"""
Specialized runner for Protocol 2 (LLM-based reduction).
This script handles the LLM client initialization that can't be serialized in YAML.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from experiments.utils.data_loaders import prepare_dataset, print_dataset_stats
from experiments.utils.llm_client import VLLMClient
from experiments.protocols import run_protocol_2

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Protocol 2 (LLM-based reduction) experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/p2_20NG.yaml",
        help="Path to YAML configuration file (default: experiments/configs/p2_20NG.yaml)"
    )
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    print(f"Loading config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare dataset
    print("\nLoading dataset...")
    dataset = prepare_dataset(
        name=config['dataset']['name'],
        n_samples=config['dataset'].get('n_samples')
    )
    print_dataset_stats(dataset)
    
    # Initialize LLM client
    print("\nInitializing vLLM client...")
    llm_client = VLLMClient(
        model_name=config['protocol_2']['llm_model_name'],
        base_url="http://localhost:8000/v1"  # Adjust for your OSCAR setup
    )
    
    # Wait for vLLM server to be ready
    print("Waiting for vLLM server...")
    llm_client.wait_for_server(timeout=600)
    
    # Extract min_cluster_size from config (with default fallback)
    min_cluster_size = config['protocol_2'].get('min_cluster_size', 15)
    print(f"\n⚙️  Configuration: min_cluster_size = {min_cluster_size}")
    
    # Run Protocol 2
    print("\nRunning Protocol 2...")
    results = run_protocol_2(
        dataset=dataset,
        llm_client=llm_client,
        target_k_values=config['protocol_2']['target_k_values'],
        seeds=config['protocol_2']['seeds'],
        min_cluster_size=min_cluster_size,
        llm_model_name=config['protocol_2']['llm_model_name'],
        save_dir=config['output']['dir'],
        output_filename=config['output'].get('filename')
    )
    
    print(f"\n{'='*80}")
    print("✓ Protocol 2 Complete!")
    print(f"{'='*80}")
    print(f"Results: {results.shape[0]} runs")
    print(f"Saved to: {config['output']['dir']}")

if __name__ == "__main__":
    main()
