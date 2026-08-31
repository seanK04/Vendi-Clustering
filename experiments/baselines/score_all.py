"""Score a whole (k, seed) grid in one process.

Same path as score.py -- same analyzer, same docs, same evaluate() -- but the
corpus, its tokenization and the COH encoder are built once rather than per
invocation, which is most of the cost when scoring many runs. Skips any combo
whose CSV already lists every method, so an interrupted run resumes for free.

    PYTHONPATH=src:. ~/anaconda3/envs/vendi-clustering/bin/python \
        experiments/baselines/score_all.py
"""

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "results" / "baselines"
COLUMNS = ["method", "n_topics", "coherence_cv", "coherence_npmi", "word_uniqueness_10", "coh"]
COH_MODEL = "paraphrase-MiniLM-L6-v2"
PREFIXES = ["vendi_ctfidf", "lda", "ctm", "etm", "fastopic"]


def out_path(results, k, seed, base_seed):
    """Seed 42 keeps the unsuffixed name the earlier runs wrote."""
    return results / (f"scores_k{k}.csv" if seed == base_seed else f"scores_k{k}_seed{seed}.csv")


def is_done(path, n_methods):
    return path.exists() and len(list(csv.DictReader(open(path)))) == n_methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--methods", nargs="+", default=PREFIXES)
    parser.add_argument("--dataset", default="20newsgroups")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--force", action="store_true", help="rescore completed combos")
    args = parser.parse_args()

    base_seed = args.seeds[0]
    todo = [
        (k, s)
        for s in args.seeds
        for k in args.k
        if args.force or not is_done(out_path(args.results, k, s, base_seed), len(args.methods))
    ]
    print(f"{len(todo)} combos to score: {todo}", flush=True)
    if not todo:
        return

    from sklearn.feature_extraction.text import CountVectorizer
    from sentence_transformers import SentenceTransformer

    from experiments.utils.data_loaders import load_20newsgroups
    from vendi_clustering.metrics.scores import evaluate
    from vendi_clustering.metrics.types import TopicModelOutput

    docs, _ = load_20newsgroups()
    analyzer = CountVectorizer().build_analyzer()
    encoder = SentenceTransformer(COH_MODEL)
    print(f"corpus {len(docs)} docs; {COH_MODEL} ready", flush=True)

    for k, seed in todo:
        rows = []
        for prefix in args.methods:
            output = TopicModelOutput.load(
                args.results / f"{prefix}_{args.dataset}_k{k}_seed{seed}.json"
            )
            metrics = evaluate(output, docs, analyzer=analyzer, embed_words=encoder.encode)
            rows.append({"method": output.method, **metrics})
        path = out_path(args.results, k, seed, base_seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"k={k} seed={seed}: "
            + " ".join(f"{r['method']}={r['coherence_cv']:.4f}" for r in rows),
            flush=True,
        )


if __name__ == "__main__":
    main()
