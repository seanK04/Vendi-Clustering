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
GLOVE_MODEL = "glove-wiki-gigaword-200"
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


def _argmax_topics(theta):
    """Hard document assignment; not used by any reported metric."""
    return np.asarray(theta).argmax(axis=1).astype(int).tolist()


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
    return top_words, _argmax_topics(theta), config


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
    return top_words, _argmax_topics(theta), config


def _combined_tm_full(vocab_size, contextual_embed_size, num_topics, en_units=200, dropout=0.4):
    """CombinedTM with the bag-of-words concatenation restored.

    topmost ships the concatenation commented out (CombinedTM.py:23 and :54), so
    its inference network reads only the contextual embedding — which is
    ZeroShotTM's defining property, not CombinedTM's. Restoring both lines
    reproduces the reference architecture of Bianchi et al. (2021): the embedding
    is projected to |V| (fc_contextual, matching the reference's adapt_bert) and
    concatenated with the BoW, giving a 2|V|-wide encoder input. The BoW remains
    the decoder's reconstruction target either way.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from topmost import CombinedTM

    class CombinedTMFull(CombinedTM):
        def __init__(self):
            super().__init__(vocab_size, contextual_embed_size, num_topics, en_units, dropout)
            self.fc11 = nn.Linear(vocab_size + vocab_size, en_units)

        def get_theta(self, x):
            contextual = self.fc_contextual(x[:, self.vocab_size:])
            combined = torch.cat((x[:, : self.vocab_size], contextual), dim=1)
            mu, logvar = self.encode(combined)
            z = self.reparameterize(mu, logvar)
            theta = self.theta_drop(F.softmax(z, dim=1))
            return (theta, mu, logvar) if self.training else theta

    return CombinedTMFull()


def fit_ctm(docs, k, seed, embeddings, preprocess, args):
    """CombinedTM via topmost's BasicTrainer.

    The model consumes [bow || contextual_embed] concatenated, which is what
    RawDataset(contextual_embed=True) builds. Seeding waits until after the
    dataset is constructed: preprocessing resets numpy's global seed, and the
    model's parameters are not allocated until the model is constructed.
    """
    import torch
    from torch.utils.data import DataLoader
    from topmost import BasicTrainer

    # RawDataset would build this same tensor, but via an int64 dense array and a
    # float64 concatenation -- ~3.9GB of simultaneous intermediates at this scale.
    # Filling a float32 array in chunks is bit-identical (counts are exact in
    # float32, the embeddings already are) and peaks under 1GB.
    rst = preprocess.preprocess(docs)
    bow, vocab = rst["train_bow"], rst["vocab"]
    n_docs, vocab_size, dim = bow.shape[0], len(vocab), embeddings.shape[1]
    train = np.empty((n_docs, vocab_size + dim), dtype=np.float32)
    for i in range(0, n_docs, 2000):
        train[i : i + 2000, :vocab_size] = bow[i : i + 2000].toarray()
    train[:, vocab_size:] = embeddings

    dataset = _Dataset(None, vocab)
    dataset.train_data = torch.from_numpy(train)
    dataset.contextual_embed_size = dim
    dataset.train_dataloader = DataLoader(
        dataset.train_data, batch_size=args.batch_size, shuffle=True
    )
    seed_everything(seed)

    model = _combined_tm_full(
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
        "bow_concatenation_restored": True,
    }
    return top_words, _argmax_topics(theta), config


def _glove_embeddings(vocab, model=GLOVE_MODEL):
    """Pretrained word vectors for `vocab`, zero rows where a word is absent.

    Reimplements topmost's make_word_embeddings. Its own version is unusable
    here: RawDataset discards the result, it returns a sparse matrix that
    ETM's torch.from_numpy rejects, and it tests membership against a
    400k-entry list rather than the index.
    """
    import gensim.downloader

    vectors = gensim.downloader.load(model)
    matrix = np.zeros((len(vocab), vectors.vectors.shape[1]), dtype=np.float32)
    found = 0
    for i, word in enumerate(vocab):
        if word in vectors.key_to_index:
            matrix[i] = vectors[word]
            found += 1
    return matrix, found


def fit_etm(docs, k, seed, embeddings, preprocess, args):
    """ETM with pre-fitted, frozen GloVe word embeddings.

    contextual_embed=False is required: ETM's encoder is Linear(vocab_size, ...)
    and get_theta row-normalizes the whole input, so concatenated document
    embeddings would break both the shape and the normalizer. Its defaults
    (pretrained_WE=None, train_WE=False) are randomly initialized weights that
    never train, so both are set explicitly.
    """
    from topmost import ETM, BasicTrainer, RawDataset

    import torch
    from torch.utils.data import DataLoader

    dataset = RawDataset(
        docs,
        preprocess,
        batch_size=args.batch_size,
        device="cpu",
        contextual_embed=False,
        verbose=False,
    )

    # ETM row-normalizes its input (ETM.py:52), so a document with no
    # in-vocabulary tokens gives 0/0 and NaN spreads to every parameter. Train
    # on the non-empty rows and mark the rest as unassigned. min_term does not
    # prevent this: it filters before the vocabulary is applied.
    keep = (dataset.train_data.sum(dim=1) > 0).numpy()
    dataset.train_data = dataset.train_data[torch.from_numpy(keep)]
    dataset.train_dataloader = DataLoader(
        dataset.train_data, batch_size=args.batch_size, shuffle=True
    )

    word_embeddings, found = _glove_embeddings(dataset.vocab)
    seed_everything(seed)

    model = ETM(
        vocab_size=dataset.vocab_size,
        embed_size=word_embeddings.shape[1],
        num_topics=k,
        en_units=args.en_units,
        dropout=args.etm_dropout,
        pretrained_WE=word_embeddings,
        train_WE=False,
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
        "en_units": args.en_units,
        "dropout": args.etm_dropout,
        "embed_size": int(word_embeddings.shape[1]),
        "pretrained_WE": GLOVE_MODEL,
        "train_WE": False,
        "glove_coverage": found / len(dataset.vocab),
        "docs_dropped_empty_bow": int((~keep).sum()),
    }
    doc_topics = np.full(len(docs), -1, dtype=int)
    doc_topics[keep] = np.asarray(theta).argmax(axis=1)
    return top_words, doc_topics.tolist(), config


FITTERS = {"fastopic": fit_fastopic, "lda": fit_lda, "ctm": fit_ctm, "etm": fit_etm}


def to_output(method, top_words, doc_topics, n_docs, metadata):
    """Build the TopicModelOutput dict, enforcing the contract before writing."""
    topic_words = [w.split() if isinstance(w, str) else list(w) for w in top_words]
    topic_ids = list(range(len(topic_words)))
    doc_topics = list(doc_topics)

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
    parser.add_argument("--batch-size", type=int, default=200, help="CTM/ETM minibatch")
    parser.add_argument("--en-units", type=int, default=800, help="ETM encoder width")
    parser.add_argument("--etm-dropout", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    docs = load_docs(args.dataset, args.n_samples)
    embeddings = load_embeddings(len(docs))
    preprocess = make_preprocess(args.vocab_size)

    print(f"{args.method} k={args.k} seed={args.seed} on {len(docs)} docs")
    start = time.time()
    top_words, doc_topics, config = FITTERS[args.method](
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
    output = to_output(args.method, top_words, doc_topics, len(docs), metadata)

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
