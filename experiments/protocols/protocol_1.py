"""
Protocol 1: Fixed-k comparison between Vendi and Agglomerative reduction.
    For each target k and seed:
        1. Fit initial BERTopic model
        2. Reduce to k topics using both methods
        3. Evaluate metrics
        4. Save results
"""
import time
import pandas as pd
from pathlib import Path
from typing import List

from experiments.utils.data_loaders import DatasetConfig
from experiments.utils.reduction import apply_reduction
from experiments.utils.models import create_bertopic_model, ModelConfig
from experiments.utils.metrics import evaluate_model


def run_protocol_1(
    dataset: DatasetConfig,
    target_k_values: List[int] = [25, 50, 100],
    seeds: List[int] = [42, 43, 44],
    min_cluster_size: int = 15,
    save_dir: str = "experiments/results/p1_fixed_k",
    output_filename: str = None
) -> pd.DataFrame:
    """
    Protocol 1: 2x2 Experimental Matrix
    Tests Reduction Method (Agglomerative vs Vendi) × Embedding Type (c-TF-IDF vs SBERT)
    
    Args:
        dataset: DatasetConfig with docs, embeddings, labels
        target_k_values: List of target topic counts
        seeds: List of random seeds for reproducibility
        min_cluster_size: HDBSCAN min_cluster_size parameter for initial clustering
        save_dir: Directory to save results
        output_filename: Optional custom filename for CSV output. If None, auto-generates as "p1_results_{dataset.name}.csv"

    Returns:
        DataFrame with all results
    """
    print("="*80)
    print("PROTOCOL 1: 2×2 Experimental Matrix")
    print("="*80)
    print(f"Dataset: {dataset.name}")
    print(f"Target k values: {target_k_values}")
    print(f"Seeds: {seeds}")
    print(f"\nExperimental Matrix:")
    print("  Reduction Methods: Agglomerative vs Vendi")
    print("  Embedding Types:   c-TF-IDF (Lexical) vs SBERT (Semantic)")
    print("="*80 + "\n")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)    
    results = []

    # 2x2 matrix: method × embedding_type
    methods = ["agglomerative", "vendi"]
    embedding_types = [
        ("ctfidf", True),   # (name, use_ctfidf flag)
        ("sbert", False)
    ]
    
    total_runs = len(seeds) * len(methods) * len(embedding_types) * len(target_k_values)
    current_run = 0

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Seed: {seed}")
        print(f"{'='*80}")
        
        # FIT THE MODEL ONCE per seed (fixed initial configuration)
        print(f"\n[INITIAL FIT - Seed {seed}]")
        
        # Use a neutral config for initial fit (method doesn't matter yet)
        config = ModelConfig(
            name=f"base_seed{seed}",
            reduction_method="agglomerative",  # Placeholder, not used in fit
            min_cluster_size=min_cluster_size,
            seed=seed
        )
        
        print(f"  Fitting BERTopic...", end='', flush=True)
        start_time = time.time()
        model = create_bertopic_model(config)
        topics, probs = model.fit_transform(dataset.docs, dataset.embeddings)
        fit_time = time.time() - start_time
        initial_k = len(set(topics) - {-1})
        print(f" ✓ {fit_time:.1f}s (k={initial_k})")
        
        # Now test all 2×2 combinations on this SAME fitted model
        for method in methods:
            for emb_name, use_ctfidf in embedding_types:
                print(f"\n[{method.upper()} + {emb_name.upper()}]")
                
                for target_k in target_k_values:
                    current_run += 1
                    print(f"\n[Run {current_run}/{total_runs}] Target k={target_k}")
                    
                    try:
                        # REFIT model with same seed for reproducibility 
                        print(f"  Refitting model...", end='', flush=True)
                        refit_start = time.time()
                        config_refit = ModelConfig(
                            name=f"{method}_{emb_name}_k{target_k}_seed{seed}",
                            reduction_method=method,
                            min_cluster_size=min_cluster_size,
                            seed=seed
                        )
                        model_refit = create_bertopic_model(config_refit)
                        model_refit.fit_transform(dataset.docs, dataset.embeddings)
                        refit_time = time.time() - refit_start
                        refit_k = len(set(model_refit.topics_) - {-1})
                        print(f" ✓ {refit_time:.1f}s (k={refit_k})")
                        
                        if refit_k > target_k:
                            print(f"  Reducing with {method} + {emb_name}...", end='', flush=True)
                            reduction_start = time.time()
                            apply_reduction(
                                model_refit,
                                dataset.docs,
                                nr_topics=target_k,
                                method=method,
                                use_ctfidf=use_ctfidf,
                            )
                            reduction_time = time.time() - reduction_start
                            final_k = len(set(model_refit.topics_) - {-1})
                            print(f" ✓ {reduction_time:.1f}s (k={final_k})")
                        else:
                            reduction_time = 0.0
                            final_k = refit_k
                            print(f"  Skipped (k={refit_k} ≤ {target_k})")
                        
                        print(f"  Evaluating...", end='', flush=True)
                        eval_start = time.time()
                        metrics = evaluate_model(model_refit, dataset.docs)
                        eval_time = time.time() - eval_start
                        print(f" ✓ {eval_time:.1f}s")
                        
                        # Collect results
                        result = {
                            "seed": seed,
                            "method": method,
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
                        
                        # Print all metrics
                        print(f"  Coherence: C_v={metrics.get('coherence_cv', 0):.4f}, NPMI={metrics.get('coherence_npmi', 0):.4f}")
                        print(f"  Diversity: WU@10={metrics.get('word_uniqueness_10', 0):.4f}")
                        
                    except Exception as e:
                        print(f"  Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results with custom or auto-generated filename
    if output_filename:
        csv_filename = output_filename if output_filename.endswith('.csv') else f"{output_filename}.csv"
    else:
        csv_filename = f"p1_results_{dataset.name}.csv"
    
    csv_path = save_path / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"\n Results saved to {csv_path}")

    # Print summary statistics
    print_protocol_1_summary(df)

    return df


def print_protocol_1_summary(df: pd.DataFrame):
    """Print summary statistics for Protocol 1 results showing 2x2 matrix."""
    print("\n" + "="*80)
    print("PROTOCOL 1 SUMMARY: 2×2 Matrix (Method × Embedding)")
    print("="*80)
    
    for target_k in sorted(df['target_k'].unique()):
        print(f"\n{'='*80}")
        print(f"Target k={target_k}")
        print(f"{'='*80}")
        
        # Create 2x2 matrix display
        for method in sorted(df['method'].unique()):
            for emb_type in sorted(df['embedding_type'].unique()):
                subset = df[
                    (df['target_k'] == target_k) & 
                    (df['method'] == method) & 
                    (df['embedding_type'] == emb_type)
                ]
                
                if len(subset) > 0:
                    print(f"\n[{method.upper():>15} + {emb_type.upper():>6}]")
                    print(f"  Coherence C_v:       {subset['coherence_cv'].mean():6.4f} ± {subset['coherence_cv'].std():.4f}")
                    print(f"  Coherence NPMI:      {subset['coherence_npmi'].mean():6.4f} ± {subset['coherence_npmi'].std():.4f}")
                    print(f"  Word Unique @10:     {subset['word_uniqueness_10'].mean():6.4f} ± {subset['word_uniqueness_10'].std():.4f}")
                    print(f"  Reduction Time:      {subset['reduction_time'].mean():6.2f}s ± {subset['reduction_time'].std():.2f}s")
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    # Create a comparison table for each target_k
    for target_k in sorted(df['target_k'].unique()):
        print(f"\nk={target_k}")
        print("-" * 80)
        print(f"{'Method + Embedding':<25} {'C_v':>8} {'NPMI':>8} {'WU@10':>8} {'Time(s)':>8}")
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
                    cv = subset['coherence_cv'].mean()
                    npmi = subset['coherence_npmi'].mean()
                    wu10 = subset['word_uniqueness_10'].mean()
                    time_val = subset['reduction_time'].mean()
                    print(f"{label:<25} {cv:8.4f} {npmi:8.4f} {wu10:8.4f} {time_val:8.2f}")
    
    print("\n" + "="*80)
