import torch
from datasets import load_dataset
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


def chunks_difference(chunk1: torch.Tensor, chunk2:torch.Tensor) -> float:
    assert chunk1.ndim == chunk2.ndim == 2, \
    "Chunk1 and Chunk2 must have 2 dimension, (Batch size, Action Chunk size, Action dim)"
    
    return torch.linalg.norm(chunk1 - chunk2, dim=-1).mean()

def align_chunks(batched_chunks: torch.Tensor) -> list[torch.Tensor]:
    assert batched_chunks.ndim == 3, \
    "Batched chunks must have 3 dimension, (Batch size, Action Chunk size, Action dim)"
    
    n_chunks = batched_chunks.shape[0]
    
    aligned_chunks = []
    for i, chunk in enumerate(batched_chunks):
        start = (n_chunks - 1) - i 
        end = None if not i else -i

        overlap = chunk[start:end]
        aligned_chunks.append(overlap)

    return aligned_chunks


def reduce_chunks_dim(dataset):
    chunks = dataset["action_chunks"].reshape(-1, 6).numpy()
    np.random.shuffle(chunks)

    pca = PCA(n_components=1)
    pca.fit(chunks)

    print("1D projection of action chunk explains variance: ", pca.explained_variance_ratio_)

    return pca

def get_chunk_pairs_differences(dataset, n_chunk_in_pairs: int = 2):

    differences = torch.zeros(
        (len(dataset)-(n_chunk_in_pairs-1), n_chunk_in_pairs-1)
    )

    for idx in tqdm(range(len(dataset)-(n_chunk_in_pairs-1))):
        rows = dataset[idx:idx+n_chunk_in_pairs]
        aligned_chunks = align_chunks(rows["action_chunks"])
        
        for col in range(1, n_chunk_in_pairs):
            chunks_diff = chunks_difference(aligned_chunks[0], aligned_chunks[col])
            
            if differences.shape[1] != 1:
                differences[idx, col] = chunks_diff
            else:
                differences[idx, 0] = chunks_diff

    return differences