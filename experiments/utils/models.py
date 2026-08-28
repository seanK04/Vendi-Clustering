"""
This module defines different BERTopic configurations for comparing
reduction methods and testing robustness across hyperparameters.
"""
from dataclasses import dataclass
from typing import Optional, Literal
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from experiments.utils.progress_wrappers import ProgressUMAP, ProgressHDBSCAN

@dataclass
class ModelConfig:
    """Configuration for a BERTopic model."""
    name: str
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "cosine"
    min_cluster_size: int = 15
    cluster_metric: str = "euclidean"
    cluster_selection_method: Literal["eom", "leaf"] = "eom"
    # Topic reduction method (applied after clustering)
    reduction_method: Literal["agglomerative", "vendi"] = "agglomerative"
    vendi_q: float = 2.0
    seed: int = 42

    def __str__(self):
        return (
            f"{self.name} (nn={self.n_neighbors}, "
            f"mcs={self.min_cluster_size}, "
            f"reduction={self.reduction_method})"
        )

def create_bertopic_model(
        config: ModelConfig,
        embedding_model: Optional[SentenceTransformer] = None
) -> BERTopic:
    """
    Create a BERTopic model from a configuration.

    Args:
        config: ModelConfig specifying hyperparameters
        embedding_model: Optional pre-loaded SentenceTransformer

    Returns:
        Configured BERTopic model
    """
    umap_model = ProgressUMAP(
        n_neighbors=config.n_neighbors,
        n_components=config.n_components,
        min_dist=config.min_dist,
        metric=config.metric,
        random_state=config.seed
    )

    cluster_model = ProgressHDBSCAN(
        min_cluster_size=config.min_cluster_size,
        metric=config.cluster_metric,
        cluster_selection_method=config.cluster_selection_method,
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        verbose=False,
        calculate_probabilities=True,
        top_n_words=25,
    )

    return topic_model
    

def get_default_config(name: str = "default", seed: int = 42) -> ModelConfig:
    """Get the default BERTopic configuration (from paper)."""
    return ModelConfig(
        name=name,
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        min_cluster_size=15,
        cluster_metric="euclidean",
        seed=seed
    )

def get_baseline_configs(seeds=[42, 43, 44]):
    """
    Get baseline configurations for comparison.

    Returns different reduction methods with multiple seeds for P1 experiment
    """
    configs = []
    for seed in seeds:
        # Agglomerative (baseline)
        configs.append(ModelConfig(
            name=f"agglomerative_seed{seed}",
            reduction_method="agglomerative",
            seed=seed
        ))
        # Vendi
        configs.append(ModelConfig(
            name=f"vendi_seed{seed}",
            reduction_method="vendi",
            seed=seed
        ))
    return configs

def print_config(config: ModelConfig):
    """Print a model configuration in a readable format."""
    print(f"\n{'='*60}")
    print(f"Model Configuration: {config.name}")
    print(f"{'='*60}")
    print(f"UMAP:")
    print(f"  - n_neighbors: {config.n_neighbors}")
    print(f"  - n_components: {config.n_components}")
    print(f"  - min_dist: {config.min_dist}")
    print(f"  - metric: {config.metric}")
    print(f"Clustering (HDBSCAN):")
    print(f"  - min_cluster_size: {config.min_cluster_size}")
    print(f"  - metric: {config.cluster_metric}")
    print(f"  - selection_method: {config.cluster_selection_method}")
    print(f"Topic Reduction:")
    print(f"  - method: {config.reduction_method}")
    if config.reduction_method == "vendi":
        print(f"  - q: {config.vendi_q}")
    print(f"Other:")
    print(f"  - seed: {config.seed}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test default config
    config = get_default_config()
    print_config(config)
    
    # Create a model
    model = create_bertopic_model(config)
    print(f"Created BERTopic model: {model}")
    
    # Show all baseline configs
    print("\nBaseline Configurations:")
    for cfg in get_baseline_configs(seeds=[42]):
        print(f"  - {cfg}")
