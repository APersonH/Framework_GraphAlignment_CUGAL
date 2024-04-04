from cugal.pred import cugal
from cugal.config import Config, SinkhornMethod
from cugal.profile import Profile, append_phases_to_csv
import numpy as np
import networkx as nx
import torch

def main(data, iter, simple, mu, path=None, sparse=False):
    config = Config(device="cuda", 
        sinkhorn_method=SinkhornMethod.MIX, 
        dtype=torch.float32,
        sinkhorn_threshold=1e-3,
        use_sparse_adjacency=sparse,
    )
    Src = data['Src']
    Tar = data['Tar']
    for i in range(Src.shape[0]):
        row_sum1 = np.sum(Src[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum1 == 0:
            Src[i, i] = 1
    for i in range(Tar.shape[0]):
        row_sum = np.sum(Tar[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum == 0:
            Tar[i, i] = 1
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)
    Src1 = nx.from_numpy_array(Src)
    Tar1 = nx.from_numpy_array(Tar)
    
    profile = Profile()

    P, mapping = cugal(Src1, Tar1, config, profile)

    if not path == None: 
        append_phases_to_csv(profile, path)

    return P
