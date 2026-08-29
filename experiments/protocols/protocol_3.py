"""
Protocol 3: Vendi Topic Reduction vs. HDBSCAN Direct Tuning
    This protocol compares Vendi Topic Reduction against the default "direct" method 
    of re-clustering documents via HDBSCAN parameter tuning, with a focus on outlier assignment rates.
    
    Procedure:
        1. Fit a high-resolution Base Model (low min_cluster_size, many topics)
        2. Create target states by fitting direct HDBSCAN models with varying parameters
        3. For each target state, record k_target and outlier rate
        4. Reduce the Base Model using VTR to match k_target
        5. Compare outlier rates and quality metrics between methods
"""
import time
import pandas as pd
from pathlib import Path
from typing import List

from experiments.utils.data_loaders import DatasetConfig
from experiments.utils.reduction import apply_reduction
from experiments.utils.models import create_bertopic_model, ModelConfig
from experiments.utils.metrics import evaluate_model


def run_protocol_3(
    dataset: DatasetConfig,
    base_min_cluster_size: int = 10,
    direct_configs: List[dict] = None,
    seed: int = 42,
    save_dir: str = "experiments/results/p3_hdbscan_comparison",
    output_filename: str = None
) -> pd.DataFrame:
    """
    Protocol 3: VTR vs Direct HDBSCAN Tuning
    
    Tests hypothesis: VTR can reach low topic counts while preserving document assignments,
    whereas direct HDBSCAN tuning often forces many documents into the outlier category.
    
    Args:
        dataset: DatasetConfig with docs, embeddings, labels
        base_min_cluster_size: min_cluster_size for high-resolution base model
        direct_configs: List of dicts with 'min_cluster_size' and 'selection_method' for direct HDBSCAN
        seed: Random seed for reproducibility
        save_dir: Directory to save results
        output_filename: Optional custom filename for CSV output

    Returns:
        DataFrame comparing VTR vs direct HDBSCAN across different target states
    """    
    print("="*80)
    print("PROTOCOL 3: Vendi Topic Reduction vs. Direct HDBSCAN Tuning")
    print("="*80)
    print(f"Dataset: {dataset.name}")
    print(f"Base model min_cluster_size: {base_min_cluster_size}")
    print(f"Direct HDBSCAN configurations: {len(direct_configs)}")
    print(f"Seed: {seed}")
    print("="*80 + "\n")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    results = []

    # Step 1: Fit high-resolution base model
    print(f"\n{'='*80}")
    print(f"STEP 1: Fitting High-Resolution Base Model")
    print(f"{'='*80}")
    
    base_config = ModelConfig(
        name=f"base_mcs{base_min_cluster_size}_seed{seed}",
        min_cluster_size=base_min_cluster_size,
        cluster_selection_method="eom",
        reduction_method="vendi",  # Will use VTR for reductions
        seed=seed
    )
    
    print(f"  Fitting base model (min_cluster_size={base_min_cluster_size})...", end='', flush=True)
    start_time = time.time()
    base_model = create_bertopic_model(base_config)
    base_topics, base_probs = base_model.fit_transform(dataset.docs, dataset.embeddings)
    base_fit_time = time.time() - start_time
    base_k = len(set(base_topics) - {-1})
    base_outliers = sum(1 for t in base_topics if t == -1)
    base_outlier_ratio = base_outliers / len(base_topics)
    print(f" ✓ {base_fit_time:.1f}s")
    print(f"  Base model: k={base_k}, outliers={base_outliers} ({base_outlier_ratio:.2%})")
    
    # Step 2: Test each direct HDBSCAN configuration
    print(f"\n{'='*80}")
    print(f"STEP 2: Direct HDBSCAN Configurations")
    print(f"{'='*80}")
    
    for i, config_dict in enumerate(direct_configs, 1):
        min_cs = config_dict["min_cluster_size"]
        sel_method = config_dict["selection_method"]
        
        print(f"\n[Configuration {i}/{len(direct_configs)}]: min_cluster_size={min_cs}, selection={sel_method}")
        
        # Fit direct HDBSCAN model
        print(f"  Fitting direct HDBSCAN model...", end='', flush=True)
        direct_config = ModelConfig(
            name=f"direct_mcs{min_cs}_{sel_method}_seed{seed}",
            min_cluster_size=min_cs,
            cluster_selection_method=sel_method,
            reduction_method="agglomerative",  # Placeholder
            seed=seed
        )
        
        direct_start = time.time()
        direct_model = create_bertopic_model(direct_config)
        direct_topics, direct_probs = direct_model.fit_transform(dataset.docs, dataset.embeddings)
        direct_fit_time = time.time() - direct_start
        direct_k = len(set(direct_topics) - {-1})
        direct_outliers = sum(1 for t in direct_topics if t == -1)
        direct_outlier_ratio = direct_outliers / len(direct_topics)
        print(f" ✓ {direct_fit_time:.1f}s")
        print(f"    Direct HDBSCAN: k={direct_k}, outliers={direct_outliers} ({direct_outlier_ratio:.2%})")
        
        # Evaluate direct model
        print(f"  Evaluating direct model...", end='', flush=True)
        direct_metrics = evaluate_model(direct_model, dataset.docs)
        print(f" ✓")
        
        # Record direct HDBSCAN results
        direct_result = {
            "config_num": i,
            "method": "direct_hdbscan",
            "min_cluster_size": min_cs,
            "selection_method": sel_method,
            "seed": seed,
            "base_k": base_k,
            "target_k": direct_k,
            "final_k": direct_k,
            "fit_time": direct_fit_time,
            "reduction_time": 0.0,
            "total_time": direct_fit_time,
            "n_outliers": direct_outliers,
            "outlier_ratio": direct_outlier_ratio,
            **direct_metrics
        }
        results.append(direct_result)
        
        # Step 3: Reduce base model using VTR to match target_k
        print(f"  Reducing base model with VTR to k={direct_k}...", end='', flush=True)
        
        # Refit base model for VTR
        vtr_config = ModelConfig(
            name=f"vtr_target{direct_k}_seed{seed}",
            min_cluster_size=base_min_cluster_size,
            cluster_selection_method="eom",
            reduction_method="vendi",
            seed=seed
        )
        
        vtr_refit_start = time.time()
        vtr_model = create_bertopic_model(vtr_config)
        vtr_model.fit_transform(dataset.docs, dataset.embeddings)
        vtr_refit_time = time.time() - vtr_refit_start
        
        # Reduce to target_k
        if len(set(vtr_model.topics_) - {-1}) > direct_k:
            vtr_reduction_start = time.time()
            apply_reduction(
                vtr_model, dataset.docs, nr_topics=direct_k, method="vendi", use_ctfidf=True
            )
            vtr_reduction_time = time.time() - vtr_reduction_start
            vtr_k = len(set(vtr_model.topics_) - {-1})
            vtr_outliers = sum(1 for t in vtr_model.topics_ if t == -1)
            vtr_outlier_ratio = vtr_outliers / len(vtr_model.topics_)
            print(f" ✓ {vtr_reduction_time:.1f}s")
            print(f"    VTR: k={vtr_k}, outliers={vtr_outliers} ({vtr_outlier_ratio:.2%})")
        else:
            vtr_reduction_time = 0.0
            vtr_k = len(set(vtr_model.topics_) - {-1})
            vtr_outliers = sum(1 for t in vtr_model.topics_ if t == -1)
            vtr_outlier_ratio = vtr_outliers / len(vtr_model.topics_)
            print(f" (skipped, already at k={vtr_k})")
        
        # Evaluate VTR model
        print(f"  Evaluating VTR model...", end='', flush=True)
        vtr_metrics = evaluate_model(vtr_model, dataset.docs)
        print(f" ✓")
        
        # Record VTR results
        vtr_result = {
            "config_num": i,
            "method": "vtr",
            "min_cluster_size": base_min_cluster_size,
            "selection_method": "eom",
            "seed": seed,
            "base_k": base_k,
            "target_k": direct_k,
            "final_k": vtr_k,
            "fit_time": vtr_refit_time,
            "reduction_time": vtr_reduction_time,
            "total_time": vtr_refit_time + vtr_reduction_time,
            "n_outliers": vtr_outliers,
            "outlier_ratio": vtr_outlier_ratio,
            **vtr_metrics
        }
        results.append(vtr_result)
        
        # Print comparison
        outlier_diff = direct_outlier_ratio - vtr_outlier_ratio
        print(f"\n  COMPARISON:")
        print(f"    Outlier Ratio Δ: {outlier_diff:+.2%} (Direct - VTR)")
        print(f"    C_v Δ: {vtr_metrics['coherence_cv'] - direct_metrics['coherence_cv']:+.4f}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if output_filename:
        csv_filename = output_filename if output_filename.endswith('.csv') else f"{output_filename}.csv"
    else:
        csv_filename = f"p3_results_{dataset.name}.csv"
    
    csv_path = save_path / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")
    
    # Print summary
    print_protocol_3_summary(df)
    
    return df


def print_protocol_3_summary(df: pd.DataFrame):
    """Print summary statistics for Protocol 3 results."""
    print("\n" + "="*80)
    print("PROTOCOL 3 SUMMARY: VTR vs. Direct HDBSCAN")
    print("="*80)
    
    # Group by method
    direct_df = df[df['method'] == 'direct_hdbscan']
    vtr_df = df[df['method'] == 'vtr']
    
    print("\n[DIRECT HDBSCAN] - Across all configurations:")
    print(f"  Mean Outlier Ratio:      {direct_df['outlier_ratio'].mean():6.2%} ± {direct_df['outlier_ratio'].std():.2%}")
    print(f"  Mean k:                  {direct_df['final_k'].mean():6.1f} ± {direct_df['final_k'].std():.1f}")
    print(f"  Coherence C_v:           {direct_df['coherence_cv'].mean():6.4f} ± {direct_df['coherence_cv'].std():.4f}")
    print(f"  Coherence NPMI:          {direct_df['coherence_npmi'].mean():6.4f} ± {direct_df['coherence_npmi'].std():.4f}")
    print(f"  Word Uniqueness @10:     {direct_df['word_uniqueness_10'].mean():6.4f} ± {direct_df['word_uniqueness_10'].std():.4f}")
    
    print("\n[VENDI TOPIC REDUCTION] - Across all configurations:")
    print(f"  Mean Outlier Ratio:      {vtr_df['outlier_ratio'].mean():6.2%} ± {vtr_df['outlier_ratio'].std():.2%}")
    print(f"  Mean k:                  {vtr_df['final_k'].mean():6.1f} ± {vtr_df['final_k'].std():.1f}")
    print(f"  Coherence C_v:           {vtr_df['coherence_cv'].mean():6.4f} ± {vtr_df['coherence_cv'].std():.4f}")
    print(f"  Coherence NPMI:          {vtr_df['coherence_npmi'].mean():6.4f} ± {vtr_df['coherence_npmi'].std():.4f}")
    print(f"  Word Uniqueness @10:     {vtr_df['word_uniqueness_10'].mean():6.4f} ± {vtr_df['word_uniqueness_10'].std():.4f}")
    
    print("\n" + "="*80)
    print("DETAILED COMPARISON BY TARGET STATE")
    print("="*80)
    
    for config_num in sorted(df['config_num'].unique()):
        direct_row = df[(df['config_num'] == config_num) & (df['method'] == 'direct_hdbscan')].iloc[0]
        vtr_row = df[(df['config_num'] == config_num) & (df['method'] == 'vtr')].iloc[0]
        
        print(f"\nConfig {config_num}: min_cluster_size={direct_row['min_cluster_size']}, "
              f"selection={direct_row['selection_method']}, k={direct_row['target_k']}")
        print("-" * 80)
        print(f"{'Metric':<25} {'Direct HDBSCAN':>18} {'VTR':>18} {'Δ (VTR - Direct)':>18}")
        print("-" * 80)
        
        # Outlier ratio (key metric!)
        outlier_delta = vtr_row['outlier_ratio'] - direct_row['outlier_ratio']
        print(f"{'Outlier Ratio':<25} {direct_row['outlier_ratio']:>17.2%} {vtr_row['outlier_ratio']:>17.2%} {outlier_delta:>17.2%}")
        
        # Quality metrics
        cv_delta = vtr_row['coherence_cv'] - direct_row['coherence_cv']
        print(f"{'C_v Coherence':<25} {direct_row['coherence_cv']:>18.4f} {vtr_row['coherence_cv']:>18.4f} {cv_delta:>+18.4f}")
        
        npmi_delta = vtr_row['coherence_npmi'] - direct_row['coherence_npmi']
        print(f"{'NPMI Coherence':<25} {direct_row['coherence_npmi']:>18.4f} {vtr_row['coherence_npmi']:>18.4f} {npmi_delta:>+18.4f}")
        
        wu_delta = vtr_row['word_uniqueness_10'] - direct_row['word_uniqueness_10']
        print(f"{'Word Uniqueness @10':<25} {direct_row['word_uniqueness_10']:>18.4f} {vtr_row['word_uniqueness_10']:>18.4f} {wu_delta:>+18.4f}")
        

    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    mean_outlier_diff = vtr_df['outlier_ratio'].mean() - direct_df['outlier_ratio'].mean()
    print(f"\n✓ VTR preserves {abs(mean_outlier_diff):.2%} more documents on average")
    print(f"  (Direct HDBSCAN: {direct_df['outlier_ratio'].mean():.2%} outliers)")
    print(f"  (VTR:            {vtr_df['outlier_ratio'].mean():.2%} outliers)")
    
    mean_cv_diff = vtr_df['coherence_cv'].mean() - direct_df['coherence_cv'].mean()
    if mean_cv_diff > 0:
        print(f"\n✓ VTR maintains {mean_cv_diff:+.4f} higher C_v coherence on average")
    else:
        print(f"\n⚠ VTR has {mean_cv_diff:.4f} lower C_v coherence on average")
    
    print("\n" + "="*80)
