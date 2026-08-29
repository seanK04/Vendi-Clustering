"""Fit Vendi-reduced BERTopic and write a TopicModelOutput JSON.

Runs in the `vendi-clustering` env. Mirrors protocol 1's fit path exactly, so
the resulting output scores identically to the committed p1 CSV — which is the
check that the baseline comparison is reading the same model the paper did.

    PYTHONPATH=src:. ~/anaconda3/envs/vendi-clustering/bin/python \
        experiments/baselines/run_vendi.py --k 50 --seed 42
"""

import argparse
import json
import time
from pathlib import Path

from experiments.utils.data_loaders import prepare_dataset
from experiments.utils.models import ModelConfig, create_bertopic_model
from experiments.utils.reduction import apply_reduction
from vendi_clustering.metrics.adapters import from_bertopic

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="20newsgroups")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--embedding-type", default="ctfidf", choices=["ctfidf", "sbert"])
    parser.add_argument("--q", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    use_ctfidf = args.embedding_type == "ctfidf"
    dataset = prepare_dataset(name=args.dataset, n_samples=args.n_samples)

    config = ModelConfig(
        name=f"vendi_{args.embedding_type}_k{args.k}_seed{args.seed}",
        reduction_method="vendi",
        min_cluster_size=args.min_cluster_size,
        seed=args.seed,
    )
    print(f"vendi/{args.embedding_type} k={args.k} seed={args.seed} on {len(dataset.docs)} docs")

    start = time.time()
    model = create_bertopic_model(config)
    model.fit_transform(dataset.docs, dataset.embeddings)
    initial_k = len(set(model.topics_) - {-1})
    print(f"  fitted: k={initial_k}")

    if initial_k > args.k:
        apply_reduction(
            model, dataset.docs, nr_topics=args.k,
            method="vendi", use_ctfidf=use_ctfidf, q=args.q,
        )
    final_k = len(set(model.topics_) - {-1})
    fit_time = time.time() - start

    output = from_bertopic(
        model,
        method=f"vendi_{args.embedding_type}",
        metadata={
            "dataset": args.dataset,
            "seed": args.seed,
            "target_k": args.k,
            "fit_time": fit_time,
            "n_docs": len(dataset.docs),
            "initial_k": initial_k,
            "final_k": final_k,
            "min_cluster_size": args.min_cluster_size,
            "embedding_type": args.embedding_type,
            "use_ctfidf": use_ctfidf,
            "q": args.q,
        },
    )

    out = args.out or (
        REPO / "experiments" / "results" / "baselines"
        / f"vendi_{args.embedding_type}_{args.dataset}_k{args.k}_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    output.save(out)

    print(f"fit in {fit_time:.1f}s; {output.n_topics} topics, {output.n_outliers} outlier docs")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
