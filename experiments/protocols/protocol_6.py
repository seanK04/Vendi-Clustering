"""
Protocol 6: q-Value Sensitivity Ablation for Vendi Clustering.

For each seed:
    1. Fit initial BERTopic model once
    2. For each q value and target k:
        a. Refit model with same seed
        b. Reduce to k topics using Vendi reduction with order q
        c. Evaluate metrics
        d. Record reduction time
"""
import time
import copy
import pandas as pd
from pathlib import Path
from typing import List

from experiments.utils.data_loaders import DatasetConfig
from experiments.utils.reduction import apply_reduction
from experiments.utils.models import create_bertopic_model, ModelConfig
from experiments.utils.metrics import evaluate_model


def run_protocol_6(
    dataset: DatasetConfig,
    q_values: List[float] = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, float("inf")],
    target_k_values: List[int] = [25, 50, 100],
    seeds: List[int] = [42, 43, 44],
    min_cluster_size: int = 15,
    save_dir: str = "experiments/results/p6_q_ablation",
    output_filename: str = None,
) -> pd.DataFrame:
    """
    Protocol 6: q-Value Sensitivity Ablation

    Tests how the Vendi Score order q affects topic reduction quality.
    Uses c-TF-IDF embeddings (best-performing representation from P1).

    Args:
        dataset: DatasetConfig with docs, embeddings, labels
        q_values: List of q values to test
        target_k_values: List of target topic counts
        seeds: Random seeds for reproducibility
        min_cluster_size: HDBSCAN min_cluster_size for initial clustering
        save_dir: Directory to save results
        output_filename: Optional custom filename for CSV output

    Returns:
        DataFrame with all results
    """
    # Normalize q values (handle string 'inf' from YAML)
    q_values = [float("inf") if (isinstance(q, str) and q.lower() == "inf") else float(q) for q in q_values]

    print("=" * 80)
    print("PROTOCOL 6: q-Value Sensitivity Ablation")
    print("=" * 80)
    print(f"Dataset: {dataset.name}")
    print(f"q values: {[('∞' if q == float('inf') else q) for q in q_values]}")
    print(f"Target k values: {target_k_values}")
    print(f"Seeds: {seeds}")
    print("=" * 80 + "\n")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    results = []

    total_runs = len(seeds) * len(q_values) * len(target_k_values)
    current_run = 0

    for seed in seeds:
        print(f"\n{'=' * 80}")
        print(f"Seed: {seed}")
        print(f"{'=' * 80}")

        for q in q_values:
            q_label = "∞" if q == float("inf") else q
            print(f"\n[q={q_label}]")

            for target_k in target_k_values:
                current_run += 1
                print(f"\n[Run {current_run}/{total_runs}] q={q_label}, k={target_k}")

                try:
                    # Fit a fresh model with vendi reduction at this q
                    config = ModelConfig(
                        name=f"vendi_q{q_label}_k{target_k}_seed{seed}",
                        reduction_method="vendi",
                        vendi_q=q,
                        min_cluster_size=min_cluster_size,
                        seed=seed,
                    )

                    print(f"  Fitting model...", end="", flush=True)
                    fit_start = time.time()
                    model = create_bertopic_model(config)
                    model.fit_transform(dataset.docs, dataset.embeddings)
                    fit_time = time.time() - fit_start
                    initial_k = len(set(model.topics_) - {-1})
                    print(f" done {fit_time:.1f}s (k={initial_k})")

                    if initial_k > target_k:
                        print(f"  Reducing (q={q_label})...", end="", flush=True)
                        reduction_start = time.time()
                        apply_reduction(
                            model,
                            dataset.docs,
                            nr_topics=target_k,
                            method="vendi",
                            use_ctfidf=True,
                            q=q,
                        )
                        reduction_time = time.time() - reduction_start
                        final_k = len(set(model.topics_) - {-1})
                        print(f" done {reduction_time:.1f}s (k={final_k})")
                    else:
                        reduction_time = 0.0
                        final_k = initial_k
                        print(f"  Skipped reduction (k={initial_k} <= {target_k})")

                    print(f"  Evaluating...", end="", flush=True)
                    eval_start = time.time()
                    metrics = evaluate_model(model, dataset.docs)
                    eval_time = time.time() - eval_start
                    print(f" done {eval_time:.1f}s")

                    result = {
                        "seed": seed,
                        "q": q,
                        "q_label": str(q_label),
                        "target_k": target_k,
                        "initial_k": initial_k,
                        "final_k": final_k,
                        "fit_time": fit_time,
                        "reduction_time": reduction_time,
                        **metrics,
                    }
                    results.append(result)

                    print(
                        f"  C_v={metrics.get('coherence_cv', 0):.4f}, "
                        f"NPMI={metrics.get('coherence_npmi', 0):.4f}, "
                        f"WU@10={metrics.get('word_uniqueness_10', 0):.4f}"
                    )

                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    df = pd.DataFrame(results)

    if output_filename:
        csv_filename = (
            output_filename
            if output_filename.endswith(".csv")
            else f"{output_filename}.csv"
        )
    else:
        csv_filename = f"p6_results_{dataset.name}.csv"

    csv_path = save_path / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print_protocol_6_summary(df)

    return df


def print_protocol_6_summary(df: pd.DataFrame):
    """Print summary statistics for Protocol 6 q-value ablation."""
    print("\n" + "=" * 80)
    print("PROTOCOL 6 SUMMARY: q-Value Sensitivity Ablation")
    print("=" * 80)

    for target_k in sorted(df["target_k"].unique()):
        print(f"\n{'=' * 80}")
        print(f"Target k={target_k}")
        print("-" * 80)
        print(
            f"{'q':>6} {'C_v':>12} {'NPMI':>12} "
            f"{'WU@10':>12} {'Time(s)':>10}"
        )
        print("-" * 80)

        for q_label in df["q_label"].unique():
            subset = df[
                (df["target_k"] == target_k) & (df["q_label"] == q_label)
            ]

            if len(subset) > 0:
                cv_mean = subset["coherence_cv"].mean()
                cv_std = subset["coherence_cv"].std()
                npmi_mean = subset["coherence_npmi"].mean()
                npmi_std = subset["coherence_npmi"].std()
                wu_mean = subset["word_uniqueness_10"].mean()
                wu_std = subset["word_uniqueness_10"].std()
                time_mean = subset["reduction_time"].mean()

                print(
                    f"{q_label:>6} "
                    f"{cv_mean:6.4f}±{cv_std:.4f} "
                    f"{npmi_mean:6.4f}±{npmi_std:.4f} "
                    f"{wu_mean:6.4f}±{wu_std:.4f} "
                    f"{time_mean:8.1f}"
                )

    print("\n" + "=" * 80)
