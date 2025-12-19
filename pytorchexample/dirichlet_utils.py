import numpy as np
from collections import defaultdict

def dirichlet_partition(labels, num_clients, alpha=0.5, seed=42):
    """
    Partition labels into non-IID splits using Dirichlet distribution.
    Returns a list of lists, each containing indices for a client.
    """
    np.random.seed(seed)
    labels = np.array(labels)
    n_classes = np.max(labels) + 1
    idxs = np.arange(len(labels))
    client_indices = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        class_idxs = idxs[labels == c]
        np.random.shuffle(class_idxs)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]
        split = np.split(class_idxs, proportions)
        for i, idx in enumerate(split):
            client_indices[i].extend(idx.tolist())
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices