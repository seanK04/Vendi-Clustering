"""
Progress bar wrappers for UMAP and HDBSCAN to show internal fitting progress.
"""
from tqdm import tqdm
import numpy as np
from typing import Optional


class ProgressUMAP:
    """Wrapper around UMAP that displays progress during fitting."""
    
    def __init__(self, n_neighbors=15, n_components=5, min_dist=0.0, 
                 metric='cosine', random_state=None, **kwargs):
        from umap import UMAP
        self.umap = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=False,
            **kwargs
        )
        self.embedding_ = None
        
    def fit(self, X, y=None):
        n_samples = X.shape[0]
        with tqdm(total=100, desc="    UMAP", ncols=80, 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
            self.umap.fit(X, y)
            pbar.update(100)
        self.embedding_ = self.umap.embedding_
        return self
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.embedding_
        
    def transform(self, X):
        return self.umap.transform(X)
        
    def __getattr__(self, name):
        # Avoid infinite recursion during deepcopy/pickle
        if name in ('__getstate__', '__setstate__', '__getnewargs__', '__getnewargs_ex__'):
            raise AttributeError(name)
        return getattr(self.__dict__['umap'], name)
    
    def __getstate__(self):
        # Return state for pickling
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        # Restore state from pickling
        self.__dict__.update(state)


class ProgressHDBSCAN:
    """Wrapper around HDBSCAN that displays progress during fitting."""
    
    def __init__(self, min_cluster_size=15, metric='euclidean', 
                 cluster_selection_method='eom', prediction_data=True, **kwargs):
        from hdbscan import HDBSCAN
        self.hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=prediction_data,
            **kwargs
        )
        self.labels_ = None
        self.probabilities_ = None
        
    def fit(self, X, y=None):
        n_samples = X.shape[0]
        with tqdm(total=100, desc="    HDBSCAN", ncols=80,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
            self.hdbscan.fit(X, y)
            pbar.update(100)
        self.labels_ = self.hdbscan.labels_
        self.probabilities_ = self.hdbscan.probabilities_
        return self
        
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
        
    def __getattr__(self, name):
        # Avoid infinite recursion during deepcopy/pickle
        if name in ('__getstate__', '__setstate__', '__getnewargs__', '__getnewargs_ex__'):
            raise AttributeError(name)
        return getattr(self.__dict__['hdbscan'], name)
    
    def __getstate__(self):
        # Return state for pickling
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        # Restore state from pickling
        self.__dict__.update(state)
