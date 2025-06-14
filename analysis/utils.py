import torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import lpips

lpips_alexnet = lpips.LPIPS(net='alex')

@torch.no_grad()
def perceptual_difference(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    assert image1.ndim == image2.ndim == 3, "Images are supposed to be (RGB)!"
    return lpips_alexnet(image1, image2).item()

perceptual_keys = ["obs_observation_images_top", "obs_observation_images_wrist"]
vector_keys = ["obs_observation_state"]


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


def compute_frames_chunks_differences(dataset, n_chunk_in_pairs: int = 2):
    assert n_chunk_in_pairs == 2, "Perceptual difference is not implemented for more than 2 pairs!"
    
    chunks_differences = torch.zeros(
        len(dataset)-(n_chunk_in_pairs-1)
    )
    frames_differences = []

    for idx in tqdm(range(len(dataset)-(n_chunk_in_pairs-1))):
        rows = dataset[idx:idx+n_chunk_in_pairs]
        aligned_chunks = align_chunks(rows["action_chunks"])

        frames_diff = {}
        for p_key in perceptual_keys:
            images = rows[p_key]
            frames_diff[p_key] = perceptual_difference(images[0, :], images[1, :])
        
        for v_key in vector_keys:
            vectors = rows[v_key]
            frames_diff[v_key] = torch.linalg.norm(vectors[0, :] - vectors[1, :]).item()
        
        frames_diff["avg"] = torch.tensor([frames_diff[k] for k in [*vector_keys, *perceptual_keys]]).mean().item()
        frames_differences.append(frames_diff)
        
        for col in range(1, n_chunk_in_pairs):
            chunks_diff = chunks_difference(aligned_chunks[0], aligned_chunks[col])
            
            chunks_differences[idx] = chunks_diff

    return chunks_differences, frames_differences

