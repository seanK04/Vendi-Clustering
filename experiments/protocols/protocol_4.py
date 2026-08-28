"""
Protocol 4: Vendi Robustness and Sensitivity Analysis
    Test Vendi's robustness across multiple initial configurations:
        1. Vary min_cluster_size (e.g., [15, 25, 50])
        2. Vary selection method ('eom', 'leaf')
        3. Vary embedding type (c-TF-IDF, SBERT)
        4. For 6 configurations (3 × 2) and 2 embeddings, fit and reduce with Vendi only
        5. Report mean ± std across all configurations
"""
import time
import pandas as pd
from pathlib import Path
from typing import List

from experiments.utils.data_loaders import DatasetConfig
from experiments.utils.reduction import apply_reduction
from experiments.utils.models import create_bertopic_model, ModelConfig
from experiments.utils.metrics import evaluate_model


def run_protocol_4(
    dataset: DatasetConfig,
    target_k_values: List[int] = [25, 50, 100],
    min_cluster_sizes: List[int] = [15, 25, 50],
    seed: int = 42,
    save_dir: str = "experiments/results/p4_robustness",
    output_filename: str = None
) -> pd.DataFrame:
    """
    Protocol 4: Vendi Robustness and Sensitivity Analysis
    Tests Vendi Topic Reduction's robustness across different initial HDBSCAN configurations.
    
    Varies:
    - min_cluster_size: [15, 25, 50] (or custom)
    - selection_method: ['eom', 'leaf']
    - embedding_type: ['c-TF-IDF', 'SBERT']
    
    For 6 configurations (3 × 2) × 2 embeddings, tests Vendi's stability.
    
    Args:
        dataset: DatasetConfig with docs, embeddings, labels
        target_k_values: List of target topic counts
        min_cluster_sizes: List of min_cluster_size values to test
        seed: Random seed for reproducibility
        save_dir: Directory to save results
        output_filename: Optional custom filename for CSV output

    Returns:
        DataFrame with all results including configuration parameters
    """
    print("="*80)
    print("PROTOCOL 4: Vendi Robustness & Sensitivity Analysis")
    print("="*80)
    print(f"Dataset: {dataset.name}")
    print(f"Target k values: {target_k_values}")
    print(f"Min cluster sizes: {min_cluster_sizes}")
    print(f"Selection methods: ['eom', 'leaf']")
    print(f"Embedding types: ['c-TF-IDF', 'SBERT']")
    print(f"Seed: {seed}")
    print(f"\nTotal configurations: {len(min_cluster_sizes)} × 2 = {len(min_cluster_sizes) * 2}")
    print(f"Testing: Vendi Topic Reduction only")
    print("="*80 + "\n")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    results = []

    # Configuration matrix - VENDI ONLY
    selection_methods = ['eom', 'leaf']
    embedding_types = [
        ("ctfidf", True),
        ("sbert", False)
    ]
    
    total_configs = len(min_cluster_sizes) * len(selection_methods)
    total_runs = total_configs * len(embedding_types) * len(target_k_values)
    current_run = 0
    config_num = 0

    for min_cs in min_cluster_sizes:
        for sel_method in selection_methods:
            config_num += 1
            print(f"\n{'='*80}")
            print(f"Configuration {config_num}/{total_configs}: min_cluster_size={min_cs}, selection={sel_method}")
            print(f"{'='*80}")
            
            # Fit model with this configuration
            config = ModelConfig(
                name=f"config_mcs{min_cs}_{sel_method}_seed{seed}",
                min_cluster_size=min_cs,
                cluster_selection_method=sel_method,  # Set selection method here
                reduction_method="agglomerative",  # Placeholder
                seed=seed
            )
            
            print(f"  Fitting BERTopic...", end='', flush=True)
            start_time = time.time()
            model = create_bertopic_model(config)
            topics, probs = model.fit_transform(dataset.docs, dataset.embeddings)
            fit_time = time.time() - start_time
            initial_k = len(set(topics) - {-1})
            print(f" ✓ {fit_time:.1f}s (k={initial_k})")
            
            # Test VENDI with both embedding types
            for emb_name, use_ctfidf in embedding_types:
                print(f"\n  [VENDI + {emb_name.upper()}]")
                
                for target_k in target_k_values:
                    current_run += 1
                    print(f"    [Run {current_run}/{total_runs}] k={target_k}", end='')
                    
                    try:
                        # REFIT model with same seed for reproducibility (avoids deepcopy issues)
                        refit_start = time.time()
                        config_refit = ModelConfig(
                            name=f"vendi_{emb_name}_mcs{min_cs}_{sel_method}_k{target_k}_seed{seed}",
                            min_cluster_size=min_cs,
                            cluster_selection_method=sel_method,
                            reduction_method="vendi",
                            seed=seed
                        )
                        model_refit = create_bertopic_model(config_refit)
                        model_refit.fit_transform(dataset.docs, dataset.embeddings)
                        refit_time = time.time() - refit_start
                        refit_k = len(set(model_refit.topics_) - {-1})
                        
                        if refit_k > target_k:
                            reduction_start = time.time()
                            apply_reduction(
                                model_refit,
                                dataset.docs,
                                nr_topics=target_k,
                                method="vendi",
                                use_ctfidf=use_ctfidf,
                            )
                            reduction_time = time.time() - reduction_start
                            final_k = len(set(model_refit.topics_) - {-1})
                            print(f" → {final_k} topics ({reduction_time:.1f}s)", end='')
                        else:
                            reduction_time = 0.0
                            final_k = refit_k
                            print(f" (skipped)", end='')
                        
                        # Evaluate
                        metrics = evaluate_model(model_refit, dataset.docs)
                        print(f" ✓")
                        
                        # Collect results with configuration info
                        result = {
                            "config_num": config_num,
                            "min_cluster_size": min_cs,
                            "selection_method": sel_method,
                            "seed": seed,
                            "method": "vendi",
                            "embedding_type": emb_name,
                            "use_ctfidf": use_ctfidf,
                            "target_k": target_k,
                            "initial_k": refit_k,
                            "final_k": final_k,
                            "fit_time": refit_time,
                            "reduction_time": reduction_time,
                            "total_time": refit_time + reduction_time,
                            **metrics
                        }
                        results.append(result)
                        
                    except Exception as e:
                        print(f" ✗ Error: {e}")
                        continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    if output_filename:
        csv_filename = output_filename if output_filename.endswith('.csv') else f"{output_filename}.csv"
    else:
        csv_filename = f"p4_results_{dataset.name}.csv"
    
    csv_path = save_path / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")

    # Print summary
    print_protocol_4_summary(df)

    return df


def print_protocol_4_summary(df: pd.DataFrame):
    """Print summary statistics for Protocol 4 robustness results."""
    print("\n" + "="*80)
    print("PROTOCOL 4 SUMMARY: Robustness Analysis")
    print("="*80)
    
    for target_k in sorted(df['target_k'].unique()):
        print(f"\n{'='*80}")
        print(f"Target k={target_k}")
        print(f"{'='*80}")
        
        print("\nMean ± Std across all configurations:")
        print("-" * 80)
        
        for method in sorted(df['method'].unique()):
            for emb_type in sorted(df['embedding_type'].unique()):
                subset = df[
                    (df['target_k'] == target_k) & 
                    (df['method'] == method) & 
                    (df['embedding_type'] == emb_type)
                ]
                
                if len(subset) > 0:
                    print(f"\n[{method.upper():>15} + {emb_type.upper():>6}] (n={len(subset)} configs)")
                    print(f"  Coherence C_v:       {subset['coherence_cv'].mean():6.4f} ± {subset['coherence_cv'].std():.4f}")
                    print(f"  Coherence NPMI:      {subset['coherence_npmi'].mean():6.4f} ± {subset['coherence_npmi'].std():.4f}")
                    print(f"  Word Unique @10:     {subset['word_uniqueness_10'].mean():6.4f} ± {subset['word_uniqueness_10'].std():.4f}")
                    print(f"  Reduction Time:      {subset['reduction_time'].mean():6.2f}s ± {subset['reduction_time'].std():.2f}s")
    
    print("\n" + "="*80)
    print("ROBUSTNESS COMPARISON TABLE")
    print("="*80)
    
    for target_k in sorted(df['target_k'].unique()):
        print(f"\nk={target_k}")
        print("-" * 80)
        print(f"{'Method + Embedding':<25} {'C_v':>12} {'NPMI':>12} {'WU@10':>12}")
        print("-" * 80)
        
        for method in sorted(df['method'].unique()):
            for emb_type in sorted(df['embedding_type'].unique()):
                subset = df[
                    (df['target_k'] == target_k) & 
                    (df['method'] == method) & 
                    (df['embedding_type'] == emb_type)
                ]
                
                if len(subset) > 0:
                    label = f"{method.capitalize():<12} + {emb_type.upper():<6}"
                    cv_str = f"{subset['coherence_cv'].mean():.4f}±{subset['coherence_cv'].std():.3f}"
                    npmi_str = f"{subset['coherence_npmi'].mean():.4f}±{subset['coherence_npmi'].std():.3f}"
                    wu10_str = f"{subset['word_uniqueness_10'].mean():.4f}±{subset['word_uniqueness_10'].std():.3f}"
                    print(f"{label:<25} {cv_str:>12} {npmi_str:>12} {wu10_str:>12}")
    
    print("\n" + "="*80)
