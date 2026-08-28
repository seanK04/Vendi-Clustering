"""Protocol 5: Reduction Trajectory (Fig. 1).

Fits a BERTopic model, then merges one topic at a time from the initial k down to
1, recording VS2, NPMI, C_v and topic diversity at every step.

This is the one protocol that reports VS2 as a subject rather than a diagnostic.
"""

import time
from pathlib import Path
from typing import List

import pandas as pd

from vendi_clustering.metrics.adapters import bertopic_analyzer, from_bertopic
from vendi_clustering.metrics.scores import (
    coherence_cv,
    coherence_npmi,
    tokenize,
    vendi_diversity,
    word_uniqueness,
)

from experiments.utils.data_loaders import DatasetConfig
from experiments.utils.models import ModelConfig, create_bertopic_model
from experiments.utils.reduction import apply_reduction


def measure(model, tokenized_docs) -> dict:
    output = from_bertopic(model, method="vendi")
    return {
        "k": output.n_topics,
        "VS2": vendi_diversity(output, q=2.0),
        "NPMI": coherence_npmi(output, tokenized_docs),
        "C_v": coherence_cv(output, tokenized_docs),
        "TD": word_uniqueness(output, top_n=10),
    }


def run_protocol_5(
    dataset: DatasetConfig,
    min_cluster_size: int = 15,
    seed: int = 42,
    use_ctfidf: bool = True,
    stop_at_k: int = 1,
    save_dir: str = "experiments/results/p5_trajectory",
    output_filename: str = None,
) -> pd.DataFrame:
    """Record the metric trajectory as topics are merged one at a time.

    Arguments:
        dataset: Documents and precomputed embeddings.
        min_cluster_size: HDBSCAN minimum cluster size for the base fit.
        seed: Random seed for the base fit.
        use_ctfidf: Cluster on c-TF-IDF vectors rather than semantic embeddings.
        stop_at_k: Lowest topic count to record.
        save_dir: Where the trajectory CSV is written.
        output_filename: CSV name; defaults to trajectory.csv.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Protocol 5: Reduction Trajectory  |  {dataset.name}")
    print("=" * 70)

    config = ModelConfig(
        name="trajectory_base",
        reduction_method="vendi",
        min_cluster_size=min_cluster_size,
        seed=seed,
    )
    model = create_bertopic_model(config)

    start = time.time()
    topics, _ = model.fit_transform(dataset.docs, dataset.embeddings)
    initial_k = len(set(topics) - {-1})
    print(f"  fit complete in {time.time() - start:.1f}s, initial k={initial_k}")

    # Tokenize once: every step scores against the same reference corpus.
    tokenized_docs = tokenize(dataset.docs, bertopic_analyzer(model))

    records: List[dict] = []
    row = measure(model, tokenized_docs)
    records.append(row)
    print(f"  k={row['k']:>4} (initial)  VS2={row['VS2']:.3f}  NPMI={row['NPMI']:.4f}")

    current_k = initial_k
    step = 0
    while current_k > stop_at_k:
        target_k = current_k - 1
        step += 1

        step_start = time.time()
        apply_reduction(
            model,
            dataset.docs,
            nr_topics=target_k,
            method="vendi",
            use_ctfidf=use_ctfidf,
        )
        step_time = time.time() - step_start

        previous_k, current_k = current_k, len(set(model.topics_) - {-1})
        if current_k >= previous_k:
            print(f"  reduction stalled at k={current_k}, stopping")
            break

        row = measure(model, tokenized_docs)
        row["step_time"] = step_time
        records.append(row)
        print(
            f"  k={row['k']:>4} (step {step:>4}) [{step_time:.1f}s]  "
            f"VS2={row['VS2']:.3f}  NPMI={row['NPMI']:.4f}  "
            f"C_v={row['C_v']:.4f}  TD={row['TD']:.4f}"
        )

    results = pd.DataFrame(records)
    csv_path = save_path / (output_filename or "trajectory.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nsaved {len(results)} steps to {csv_path}")

    return results


def print_protocol_5_summary(results: pd.DataFrame) -> None:
    if results.empty:
        print("No trajectory recorded.")
        return

    print("\n" + "=" * 70)
    print("PROTOCOL 5 SUMMARY: Reduction Trajectory")
    print("=" * 70)
    print(f"  steps recorded:  {len(results)}")
    print(f"  k range:         {results['k'].max()} -> {results['k'].min()}")

    peak = results.loc[results["VS2"].idxmax()]
    print(f"  peak VS2:        {peak['VS2']:.3f} at k={int(peak['k'])}")

    best_npmi = results.loc[results["NPMI"].idxmax()]
    print(f"  peak NPMI:       {best_npmi['NPMI']:.4f} at k={int(best_npmi['k'])}")
