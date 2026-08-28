"""Vendi Clustering: diversity-preserving topic merging.

Reference implementation for "Vendi Clustering for Topic Modeling".
"""

from .clustering import VendiClustering
from .clustering_general import GeneralVendiClustering

__all__ = ["VendiClustering", "GeneralVendiClustering"]

__version__ = "0.1.0.dev0"
