"""
Specialized runner for Protocol 2 (LLM-based reduction) using OpenAI's hosted API.
This script handles the LLM client initialization that can't be serialized in YAML.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from experiments.utils.data_loaders import prepare_dataset, print_dataset_stats
from experiments.utils.llm_client import OpenAIClient
from experiments.protocols import run_protocol_2

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Protocol 2 (LLM-based reduction) experiments with OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/p2_20NG_openai.yaml",
        help="Path to YAML configuration file (default: experiments/configs/p2_20NG_openai.yaml)"
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

    # Initialize OpenAI client
    print("\nInitializing OpenAI client...")
    llm_client = OpenAIClient(
        model_name=config['protocol_2']['llm_model_name']
    )

    # No-op for OpenAI; kept so the runner has the same call shape as the vLLM variant
    print("Waiting for OpenAI client...")
    llm_client.wait_for_server(timeout=600)

    # Extract min_cluster_size from config (with default fallback)
    min_cluster_size = config['protocol_2'].get('min_cluster_size', 15)
    print(f"\n⚙️  Configuration: min_cluster_size = {min_cluster_size}")

    # Use an OpenAI-specific output filename so we don't overwrite the Gemma CSVs
    output_filename = f"p2_results_{dataset.name}_openai.csv"

    # Run Protocol 2
    print("\nRunning Protocol 2...")
    results = run_protocol_2(
        dataset=dataset,
        llm_client=llm_client,
        target_k_values=config['protocol_2']['target_k_values'],
        seeds=config['protocol_2']['seeds'],
        min_cluster_size=min_cluster_size,
        llm_model_name=config['protocol_2']['llm_model_name'],
        top_keywords=config['protocol_2'].get('top_keywords', 5),
        save_dir=config['output']['dir'],
        output_filename=output_filename
    )

    print(f"\n{'='*80}")
    print("✓ Protocol 2 Complete!")
    print(f"{'='*80}")
    print(f"Results: {results.shape[0]} runs")
    print(f"Saved to: {config['output']['dir']}")

if __name__ == "__main__":
    main()
