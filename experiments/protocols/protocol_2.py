"""
Protocol 2: Vendi Topic Reduction vs. LLM-Assisted Reduction

This protocol compares VTR against iterative LLM-based merging.
Note: VTR results can be taken from Protocol 1 for comparison.
This implementation focuses on LLM reduction benchmarking.

Procedure:
    1. Fit initial BERTopic model with fixed configuration
    2. Reduce to target_k using LLM-based iterative merging
    3. Track computational cost (API calls, time, etc.)
    4. Evaluate quality metrics
    5. Save results for comparison with P1 VTR results
"""
import time
import pandas as pd
from pathlib import Path
from typing import List, Optional
import gc

from experiments.utils.data_loaders import DatasetConfig
from experiments.utils.models import create_bertopic_model, ModelConfig
from experiments.utils.llm_reduction import LLMReducer
from experiments.utils.metrics import evaluate_model


def run_protocol_2(
    dataset: DatasetConfig,
    target_k_values: List[int] = [25, 50, 100],
    seeds: List[int] = [42],
    min_cluster_size: int = 15,
    llm_client=None,
    llm_model_name: str = "gemma3:12b",
    top_keywords: int = 5,
    save_dir: str = "experiments/results/p2_llm_comparison",
    output_filename: str = None
) -> pd.DataFrame:
    """
    Protocol 2: LLM-Assisted Topic Reduction Benchmarking
    
    Tests LLM-based iterative merging strategy where an LLM identifies
    overlapping topic pairs based on their top 10 keywords.
    
    Key Metrics:
    - Quality: Coherence (C_v, NPMI), Diversity (Word Uniqueness, Vendi)
    - Cost: API calls, total time, average time per call
    
    Note: For comparison with VTR, use results from Protocol 1.
    
    Args:
        dataset: DatasetConfig with docs, embeddings, labels
        target_k_values: List of target topic counts
        seeds: List of random seeds for reproducibility
        min_cluster_size: HDBSCAN min_cluster_size parameter for initial clustering
        llm_client: LLM client with .generate(prompt) method
        llm_model_name: Name of LLM model (e.g., "gemma3:12b")
        save_dir: Directory to save results
        output_filename: Optional custom filename for CSV output

    Returns:
        DataFrame with LLM reduction results and computational costs
    """
    print("="*80)
    print("PROTOCOL 2: LLM-Assisted Topic Reduction")
    print("="*80)
    print(f"Dataset: {dataset.name}")
    print(f"Target k values: {target_k_values}")
    print(f"Seeds: {seeds}")
    print(f"LLM Model: {llm_model_name}")
    print(f"\nNote: For VTR comparison, use Protocol 1 results")
    print("="*80 + "\n")

    if llm_client is None:
        raise ValueError("llm_client must be provided. Pass a client with .generate(prompt) method.")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    results = []

    # Initialize LLM reducer
    llm_reducer = LLMReducer(llm_client, model_name=llm_model_name, top_keywords=top_keywords)

    total_runs = len(seeds) * len(target_k_values)
    current_run = 0

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Seed: {seed}")
        print(f"{'='*80}")
        
        # Fit base model once per seed
        print(f"\n[BASE MODEL - Seed {seed}]")
        base_config = ModelConfig(
            name=f"base_llm_seed{seed}",
            reduction_method="agglomerative",  # Placeholder (not used)
            min_cluster_size=min_cluster_size,
            seed=seed
        )
        
        print(f"  Fitting BERTopic...", end='', flush=True)
        start_time = time.time()
        base_model = create_bertopic_model(base_config)
        base_topics, base_probs = base_model.fit_transform(dataset.docs, dataset.embeddings)
        fit_time = time.time() - start_time
        initial_k = len(set(base_topics) - {-1})
        print(f" ✓ {fit_time:.1f}s (k={initial_k})")
        
        # Test LLM reduction for each target_k
        for target_k in target_k_values:
            current_run += 1
            print(f"\n{'='*80}")
            print(f"[Run {current_run}/{total_runs}] LLM Reduction to k={target_k}")
            print(f"{'='*80}")
            
            try:
                # Refit model for clean state
                print(f"  Refitting base model...", end='', flush=True)
                refit_start = time.time()
                llm_config = ModelConfig(
                    name=f"llm_k{target_k}_seed{seed}",
                    reduction_method="agglomerative",  # Not used for LLM
                    min_cluster_size=min_cluster_size,
                    seed=seed
                )
                llm_model = create_bertopic_model(llm_config)
                llm_model.fit_transform(dataset.docs, dataset.embeddings)
                refit_time = time.time() - refit_start
                refit_k = len(set(llm_model.topics_) - {-1})
                print(f" ✓ {refit_time:.1f}s (k={refit_k})")
                
                if refit_k <= target_k:
                    print(f"  Skipped: Base model already at k={refit_k} ≤ {target_k}")
                    # Still evaluate and save results
                    llm_stats = {
                        'total_api_calls': 0,
                        'total_llm_time': 0.0,
                        'avg_llm_time': 0.0,
                        'merge_history': []
                    }
                    reduction_time = 0.0
                    final_k = refit_k
                else:
                    # Perform LLM reduction
                    print(f"  Starting LLM reduction ({refit_k} → {target_k})...")
                    reduction_start = time.time()
                    llm_stats = llm_reducer.reduce(llm_model, dataset.docs, target_k)
                    reduction_time = time.time() - reduction_start
                    final_k = len(set(llm_model.topics_) - {-1})
                    
                    print(f"  ✓ LLM Reduction complete: {reduction_time:.1f}s")
                    print(f"    API Calls: {llm_stats['total_api_calls']}")
                    print(f"    Avg LLM Time: {llm_stats['avg_llm_time']:.2f}s/call")
                    print(f"    Final k: {final_k}")
                
                # Evaluate model
                print(f"  Evaluating...", end='', flush=True)
                eval_start = time.time()
                metrics = evaluate_model(llm_model, dataset.docs)
                eval_time = time.time() - eval_start
                print(f" ✓ {eval_time:.1f}s")
                
                # Collect results
                result = {
                    "method": "llm",
                    "seed": seed,
                    "llm_model": llm_model_name,
                    "target_k": target_k,
                    "initial_k": refit_k,
                    "final_k": final_k,
                    "fit_time": refit_time,
                    "reduction_time": reduction_time,
                    "total_time": refit_time + reduction_time,
                    "api_calls": llm_stats['total_api_calls'],
                    "total_llm_time": llm_stats['total_llm_time'],
                    "avg_llm_time": llm_stats['avg_llm_time'],
                    "n_merges": len(llm_stats['merge_history']),
                    **metrics
                }
                results.append(result)
                
                # Print key metrics
                print(f"\n  METRICS:")
                print(f"    Coherence: C_v={metrics.get('coherence_cv', 0):.4f}, NPMI={metrics.get('coherence_npmi', 0):.4f}")
                print(f"    Diversity: WU@10={metrics.get('word_uniqueness_10', 0):.4f}")
                print(f"    Cost: {llm_stats['total_api_calls']} calls, {reduction_time:.1f}s total")

                del llm_model
                if 'llm_stats' in locals():
                    del llm_stats                
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    if output_filename:
        csv_filename = output_filename if output_filename.endswith('.csv') else f"{output_filename}.csv"
    else:
        csv_filename = f"p2_results_{dataset.name}.csv"
    
    csv_path = save_path / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")

    # Print summary
    print_protocol_2_summary(df)

    return df


def print_protocol_2_summary(df: pd.DataFrame):
    """Print summary statistics for Protocol 2 LLM reduction results."""
    print("\n" + "="*80)
    print("PROTOCOL 2 SUMMARY: LLM-Assisted Topic Reduction")
    print("="*80)
    
    for target_k in sorted(df['target_k'].unique()):
        print(f"\n{'='*80}")
        print(f"Target k={target_k}")
        print(f"{'='*80}")
        
        subset = df[df['target_k'] == target_k]
        
        if len(subset) > 0:
            print(f"\n[LLM REDUCTION] (n={len(subset)} runs)")
            print(f"  Final k:             {subset['final_k'].mean():6.1f} ± {subset['final_k'].std():.1f}")
            print(f"  Coherence C_v:       {subset['coherence_cv'].mean():6.4f} ± {subset['coherence_cv'].std():.4f}")
            print(f"  Coherence NPMI:      {subset['coherence_npmi'].mean():6.4f} ± {subset['coherence_npmi'].std():.4f}")
            print(f"  Word Unique @10:     {subset['word_uniqueness_10'].mean():6.4f} ± {subset['word_uniqueness_10'].std():.4f}")
            print(f"\n  COMPUTATIONAL COST:")
            print(f"  API Calls:           {subset['api_calls'].mean():6.1f} ± {subset['api_calls'].std():.1f}")
            print(f"  Total LLM Time:      {subset['total_llm_time'].mean():6.1f}s ± {subset['total_llm_time'].std():.1f}s")
            print(f"  Avg Time/Call:       {subset['avg_llm_time'].mean():6.2f}s ± {subset['avg_llm_time'].std():.2f}s")
            print(f"  Reduction Time:      {subset['reduction_time'].mean():6.1f}s ± {subset['reduction_time'].std():.1f}s")
            print(f"  Total Time:          {subset['total_time'].mean():6.1f}s ± {subset['total_time'].std():.1f}s")
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Target k':<10} {'C_v':>10} {'NPMI':>10} {'WU@10':>10} {'Calls':>10} {'Time(s)':>10}")
    print("-" * 80)
    
    for target_k in sorted(df['target_k'].unique()):
        subset = df[df['target_k'] == target_k]
        
        if len(subset) > 0:
            cv = subset['coherence_cv'].mean()
            npmi = subset['coherence_npmi'].mean()
            wu10 = subset['word_uniqueness_10'].mean()
            calls = subset['api_calls'].mean()
            time_val = subset['reduction_time'].mean()
            
            print(f"{target_k:<10} {cv:>10.4f} {npmi:>10.4f} {wu10:>10.4f} {calls:>10.0f} {time_val:>10.1f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    total_calls = df['api_calls'].sum()
    total_time = df['reduction_time'].sum()
    avg_time_per_call = df['avg_llm_time'].mean()
    
    print(f"\n✓ Total API Calls: {total_calls:.0f}")
    print(f"✓ Total Reduction Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"✓ Average Time per LLM Call: {avg_time_per_call:.2f}s")
    print(f"\n⚠ For comparison with VTR:")
    print(f"  - Use Protocol 1 results for VTR performance")
    print(f"  - Compare quality metrics (C_v, NPMI) at same k values")
    print(f"  - VTR reduction time is typically <10s vs {df['reduction_time'].mean():.1f}s for LLM")
    
    print("\n" + "="*80)
