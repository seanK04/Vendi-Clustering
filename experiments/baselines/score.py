"""Score TopicModelOutput JSONs — every method through one code path.

Runs in the `vendi-clustering` env. The comparison is only honest if the
analyzer and the COH encoder are identical across methods, so both are built
once here and passed to every evaluate() call.

    PYTHONPATH=src:. ~/anaconda3/envs/vendi-clustering/bin/python \
        experiments/baselines/score.py experiments/results/baselines/*.json
"""

import argparse
import csv
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer

from experiments.utils.data_loaders import load_20newsgroups
from vendi_clustering.metrics.scores import evaluate
from vendi_clustering.metrics.types import TopicModelOutput

COH_MODEL = "paraphrase-MiniLM-L6-v2"
COLUMNS = ["method", "n_topics", "coherence_cv", "coherence_npmi", "word_uniqueness_10", "coh"]


def make_embed_words():
    """COH's encoder. Not all-MiniLM-L6-v2 — that is the CEDC extraction model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(COH_MODEL)
    return lambda words: model.encode(words)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--dataset", default="20newsgroups")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--no-coh", action="store_true", help="skip COH (no encoder download)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.dataset != "20newsgroups":
        raise ValueError(f"unsupported dataset: {args.dataset}")
    docs, _ = load_20newsgroups(n_samples=args.n_samples)

    analyzer = CountVectorizer().build_analyzer()
    embed_words = None if args.no_coh else make_embed_words()

    rows = []
    for path in args.paths:
        output = TopicModelOutput.load(path)
        n_docs = output.metadata.get("n_docs")
        if n_docs is not None and n_docs != len(docs):
            raise ValueError(
                f"{path.name} was fitted on {n_docs} docs but scoring against {len(docs)}"
            )
        metrics = evaluate(output, docs, analyzer=analyzer, embed_words=embed_words)
        rows.append({"method": output.method, **metrics})
        print(f"scored {path.name}")

    print()
    header = f"{'method':<22}" + "".join(f"{c:>20}" for c in COLUMNS[1:])
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['method']:<22}"
        for col in COLUMNS[1:]:
            value = row.get(col)
            line += f"{value:>20.6f}" if isinstance(value, float) else f"{str(value):>20}"
        print(line)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
