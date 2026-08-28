from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset as hf_load_dataset
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import os
import pickle
import re

def load_20newsgroups(subset='all', n_samples=None):
    """Load 20 Newsgroups dataset"""
    data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
    docs = data['data']
    labels = data['target']
    if n_samples:
        docs = docs[:n_samples]
        labels = labels[:n_samples]
    return docs, labels

def load_agnews(n_samples=None):
    """Load ag news dataset"""
    dataset = hf_load_dataset("ag_news", split="train")
    docs = [x["text"] for x in dataset]
    labels = np.array([x["label"] for x in dataset])
    if n_samples:
        docs = docs[:n_samples]
        labels = labels[:n_samples]
    return docs, labels

def load_covid_twitter(n_samples=None):
    """
    Load COVID-19 Twitter dataset from Hugging Face.
    Dataset: istat-ai/covid-tweets-100k
    """
    dataset = hf_load_dataset("istat-ai/covid-tweets-100k", split="train")
    raw_docs = [x["text"] for x in dataset]
    clean_docs = []
    for text in raw_docs:
        # 1. Remove Retweet markers (to avoid artificial density peaks)
        text = re.sub(r'^RT @\w+: ', '', text)
        # 2. Remove URLs (no semantic value for SBERT)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # 3. Basic whitespace normalization
        text = " ".join(text.split())

        if len(text.split()) > 5:
            clean_docs.append(text)
    
    if n_samples:
        clean_docs = clean_docs[:n_samples]
    
    labels = np.zeros(len(clean_docs))

    return clean_docs, labels

def get_embeddings(docs, model_name="all-MiniLM-L6-v2", cache_dir="experiments/results/embeddings"):
    """
    Compute or load cached embeddings for documents.

    Args:
        docs: List of documents to embed
        model_name: SentenceTransformer model name
        cache_dir: Directory to store cached embeddings
    
    Returns:
        np.ndarray: Document embeddings of shape (n_docs, embedding_dim)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Create unique cache filename based on model and # of docs
    safe_model_name = model_name.replace('/', '_')
    cache_key = f"{safe_model_name}_{len(docs)}_docs.pkl"
    cache_path = os.path.join(cache_dir, cache_key)

    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f" Shape: {embeddings.shape}")
        return embeddings

    # Cache miss: need to compute embeddings
    print(f"Computing embeddings with {model_name}...")
    print(f"Documents: {len(docs)}")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        batch_size=32, # Adjust based on GPU/CPU
        convert_to_numpy=True
    )

    print(f"Computed shape: {embeddings.shape}")

    # Save to cache for future use
    print(f"Caching to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings

@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    docs: List[str]
    labels: Optional[np.ndarray]
    embeddings: np.ndarray
    embedding_model: str = "all-MiniLM-L6-v2"

def prepare_dataset(name="20newsgroups", n_samples=None, embedding_model="all-MiniLM-L6-v2"):
    """
    Load dataset with pre-computed embeddings.

    Returns:
        DatasetConfig object with docs, labels, and embeddings
    """
    if name == "20newsgroups":
        docs, labels = load_20newsgroups(n_samples=n_samples)
    elif name == "agnews":
        docs, labels = load_agnews(n_samples=n_samples)
    elif name == "covid_twitter":
        docs, labels = load_covid_twitter(n_samples=n_samples)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Compute/load embeddings
    embeddings = get_embeddings(docs, model_name=embedding_model)

    return DatasetConfig(
        name=name,
        docs=docs,
        labels=labels,
        embeddings=embeddings,
        embedding_model=embedding_model
    )

def print_dataset_stats(dataset: DatasetConfig):
    """Print useful statistics about the dataset"""
    print(f"\nDataset: {dataset.name}")
    print(f"Documents: {len(dataset.docs)}")
    print(f"Embedding dim: {dataset.embeddings.shape[1]}")
    if dataset.labels is not None:
        print(f"Classes: {len(np.unique(dataset.labels))}")
