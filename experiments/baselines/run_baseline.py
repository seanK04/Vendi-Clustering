"""Fit a native topic-model baseline and write a TopicModelOutput JSON.

Runs in the `topmost` conda env, which does not have vendi_clustering
installed. The JSON is the only thing that crosses into the scoring env, so
the contract is written out by hand here rather than imported.

    ~/anaconda3/envs/topmost/bin/python experiments/baselines/run_baseline.py \
        --method fastopic --k 50 --seed 42
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

TOP_WORDS = 25
EMBED_MODEL = "all-MiniLM-L6-v2"
REPO = Path(__file__).resolve().parents[2]
EMBED_DIR = REPO / "experiments" / "results" / "embeddings"


def load_docs(dataset, n_samples=None):
    """The document list, in the order the Vendi experiments see it."""
    if dataset != "20newsgroups":
        raise ValueError(f"unsupported dataset: {dataset}")
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    docs = data["data"]
    if n_samples:
        docs = docs[:n_samples]
    return docs


def load_embeddings(n_docs):
    """The same cached MiniLM vectors BERTopic was given."""
    path = EMBED_DIR / f"{EMBED_MODEL.replace('/', '_')}_{n_docs}_docs.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"no cached embeddings at {path}; the cache keys on document count"
        )
    embeddings = pickle.load(open(path, "rb"))
    if embeddings.shape[0] != n_docs:
        raise ValueError(f"{path} has {embeddings.shape[0]} rows, expected {n_docs}")
    return np.asarray(embeddings, dtype=np.float32)


def make_preprocess(vocab_size):
    """One preprocessing convention for every baseline.

    stopwords="English", min_length=3, keep_num=False are Preprocess defaults.
    min_term=0 keeps every document, so doc_topics stays aligned with docs.
    """
    from topmost import Preprocess

    return Preprocess(vocab_size=vocab_size, min_term=0, verbose=False)


def seed_everything(seed):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_fastopic(docs, k, seed, embeddings, preprocess, args):
    """FASTopic via the raw class, not topmost's FASTopicTrainer.

    The trainer passes already-preprocessed text to fit_transform, which feeds
    it to both the bag-of-words step and the sentence encoder — so the encoder
    would see de-stopworded token soup instead of documents. It also has no
    parameter for preset_doc_embeddings. Both are avoided by calling FASTopic
    directly with the raw docs.
    """
    from fastopic import FASTopic

    model = FASTopic(
        num_topics=k,
        preprocess=preprocess,
        num_top_words=TOP_WORDS,
        device="cpu",
        doc_embed_model=EMBED_MODEL,
        verbose=False,
    )
    seed_everything(seed)
    top_words, theta = model.fit_transform(
        docs,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        preset_doc_embeddings=embeddings,
    )
    config = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "preset_doc_embeddings": True,
        "doc_embed_model": EMBED_MODEL,
    }
    return top_words, theta, config


class _Dataset:
    """The three attributes topmost's LDA trainer reads off a dataset.

    RawDataset is not usable here: it never sets `train_bow`, which
    LDAGensimTrainer indexes directly. Dense2Corpus also needs a dense array,
    so the sparse counts are materialised once and reused.
    """

    def __init__(self, bow, vocab):
        self.train_bow = bow
        self.vocab = vocab
        self.vocab_size = len(vocab)


def fit_lda(docs, k, seed, embeddings, preprocess, args):
    """LDA via gensim. Ignores embeddings; LDA is bag-of-words only.

    max_iter maps to gensim `passes`, whose topmost default of 1 is ~9 online
    updates over 20NG. Seeding happens after preprocess(), which resets numpy's
    global seed to 42 mid-call; LdaModel takes no random_state and falls back to
    that global RNG.
    """
    from topmost import LDAGensimTrainer

    rst = preprocess.preprocess(docs)
    dataset = _Dataset(np.asarray(rst["train_bow"].todense(), dtype=np.int32), rst["vocab"])
    seed_everything(seed)

    trainer = LDAGensimTrainer(
        dataset,
        num_topics=k,
        num_top_words=TOP_WORDS,
        max_iter=args.passes,
        verbose=False,
    )
    top_words, theta = trainer.train()

    unassigned = int((np.asarray(theta).sum(axis=1) == 0).sum())
    config = {
        "passes": args.passes,
        "alpha": "symmetric",
        "eta": None,
        "minimum_probability": 0.01,
        "docs_below_minimum_probability": unassigned,
    }
    return top_words, theta, config


class _PresetEmbedder:
    """Stands in for RawDataset's sentence encoder, returning the cached vectors.

    RawDataset takes any non-str `doc_embed_model` as its embedder verbatim, and
    encodes the raw docs (basic_dataset.py:75) rather than the preprocessed ones,
    so the cached vectors line up index for index.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def encode(self, docs):
        if len(docs) != len(self.embeddings):
            raise ValueError(f"{len(docs)} docs but {len(self.embeddings)} cached embeddings")
        return self.embeddings


def fit_ctm(docs, k, seed, embeddings, preprocess, args):
    """CombinedTM via topmost's BasicTrainer.

    The model consumes [bow || contextual_embed] concatenated, which is what
    RawDataset(contextual_embed=True) builds. Seeding waits until after the
    dataset is constructed: preprocessing resets numpy's global seed, and the
    model's parameters are not allocated until CombinedTM() is called.
    """
    from topmost import BasicTrainer, CombinedTM, RawDataset

    dataset = RawDataset(
        docs,
        preprocess,
        batch_size=args.batch_size,
        device="cpu",
        contextual_embed=True,
        doc_embed_model=_PresetEmbedder(embeddings),
        verbose=False,
    )
    seed_everything(seed)

    model = CombinedTM(
        vocab_size=dataset.vocab_size,
        contextual_embed_size=dataset.contextual_embed_size,
        num_topics=k,
    )
    trainer = BasicTrainer(
        model,
        dataset,
        num_top_words=TOP_WORDS,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        verbose=False,
    )
    top_words, theta = trainer.train()

    config = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "preset_doc_embeddings": True,
        "doc_embed_model": EMBED_MODEL,
        "en_units": 200,
        "dropout": 0.4,
    }
    return top_words, theta, config


FITTERS = {"fastopic": fit_fastopic, "lda": fit_lda, "ctm": fit_ctm}


def to_output(method, top_words, theta, n_docs, metadata):
    """Build the TopicModelOutput dict, enforcing the contract before writing."""
    topic_words = [w.split() if isinstance(w, str) else list(w) for w in top_words]
    topic_ids = list(range(len(topic_words)))
    doc_topics = np.asarray(theta).argmax(axis=1).astype(int).tolist()

    if -1 in topic_ids:
        raise ValueError("topic_ids must exclude the outlier topic")
    if len(topic_ids) != len(topic_words):
        raise ValueError(f"{len(topic_ids)} topic_ids but {len(topic_words)} word lists")
    if len(doc_topics) != n_docs:
        raise ValueError(f"{len(doc_topics)} doc_topics but {n_docs} documents")
    short = [i for i, w in enumerate(topic_words) if len(w) != TOP_WORDS]
    if short:
        raise ValueError(f"topics {short[:5]} do not have {TOP_WORDS} words")

    return {
        "method": method,
        "topic_ids": topic_ids,
        "topic_words": topic_words,
        "doc_topics": doc_topics,
        "topic_embeddings": None,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(FITTERS))
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="20newsgroups")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--passes", type=int, default=20, help="LDA corpus sweeps")
    parser.add_argument("--batch-size", type=int, default=200, help="CTM minibatch")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    docs = load_docs(args.dataset, args.n_samples)
    embeddings = load_embeddings(len(docs))
    preprocess = make_preprocess(args.vocab_size)

    print(f"{args.method} k={args.k} seed={args.seed} on {len(docs)} docs")
    start = time.time()
    top_words, theta, config = FITTERS[args.method](
        docs, args.k, args.seed, embeddings, preprocess, args
    )
    fit_time = time.time() - start

    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "target_k": args.k,
        "fit_time": fit_time,
        "n_docs": len(docs),
        "vocab_size": args.vocab_size,
        **config,
    }
    output = to_output(args.method, top_words, theta, len(docs), metadata)

    out = args.out or (
        REPO / "experiments" / "results" / "baselines"
        / f"{args.method}_{args.dataset}_k{args.k}_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output))

    assigned = len(set(output["doc_topics"]))
    print(f"fit in {fit_time:.1f}s; {len(output['topic_ids'])} topics, {assigned} assigned")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
