
import faiss
import logging
import numpy as np
import torch
import torchvision.models
from typing import Tuple
from torch.utils.data import Dataset

import visualizations


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


def compute_recalls(eval_ds: Dataset, queries_descriptors : np.ndarray, database_descriptors : np.ndarray,
                    output_folder : str = None, num_preds_to_save : int = 0,
                    save_only_wrong_preds : bool = True) -> Tuple[np.ndarray, str]:
    """Compute the recalls given the queries and database descriptors. The dataset is needed to know the ground truth
    positives for each query."""

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(queries_descriptors.shape[1])
    faiss_index.add(database_descriptors)
    del database_descriptors

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])

    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, output_folder, save_only_wrong_preds)
    
    return recalls, recalls_str

  
class GeMPooling(nn.Module):
    def __init__(self, feature_size, pool_size=7, init_norm=3.0, eps=1e-6, normalize=False, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.feature_size = feature_size  # Final layer channel size, the pow calc at -1 axis
        self.pool_size = pool_size
        self.init_norm = init_norm
        #Use a learnable parameter p to compute not the standard average but a generalized where data is ^p and the sum is rooted by p
        self.p = torch.nn.Parameter(torch.ones(self.feature_size) * self.init_norm, requires_grad=True)
        self.p.data.fill_(init_norm)
        self.normalize = normalize
        self.avg_pooling = nn.AvgPool2d((self.pool_size, self.pool_size))
        self.eps = eps

    def forward(self, features):
        # filter invalid value: set minimum to 1e-6
        # features-> (B, H, W, C)
        
        #^p
        features = features.clamp(min=self.eps).pow(self.p)
        features = features.permute((0, 3, 1, 2))
        #standard avg pooling operation
        features = self.avg_pooling(features)
        features = torch.squeeze(features)
        features = features.permute((0, 2, 3, 1))
        #^1/p
        features = torch.pow(features, (1.0 / self.p))
        
        #if you want to normalize output featurs to a unit vector 
        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)
        return features
  
