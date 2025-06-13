# NOTE: This script can run on LeRobot main branch
from tqdm import tqdm
from act_pipeline import ACTInferencePipeline
from constants import DEFAULT_ACT_MODEL
import torch
import numpy as np
from datasets import Dataset, Features, Value, Sequence
from torch.utils.data import DataLoader


# Prepare data for Hugging Face Dataset
def prepare_hf_dataset_data(all_data):
    """Convert tensor data to numpy arrays for HF Dataset compatibility."""
    import numpy as np
    
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:  # scalar
                return obj.item()  # Convert to Python scalar
            else:
                return obj  # Keep as numpy array
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    hf_data = []
    
    for sample in all_data:
        # Convert tensors to numpy arrays
        hf_sample = {
            'action_chunks': sample['action_chunks'].numpy(),
            'batch_idx': int(sample['batch_idx']),
            'sample_idx': int(sample['sample_idx']),
            'episode_idx': int(sample['episode_idx']) if sample['episode_idx'] is not None else -1,
            'frame_idx': int(sample['frame_idx']) if sample['frame_idx'] is not None else -1,
        }
        
        # Handle observations - convert tensors to numpy and handle scalar cases
        for key, value in sample['observations'].items():
            feature_key = f'obs_{key.replace(".", "_")}'
            if isinstance(value, torch.Tensor):
                numpy_value = value.numpy()
                hf_sample[feature_key] = convert_numpy_types(numpy_value)
            elif isinstance(value, (str, int, float)):
                hf_sample[feature_key] = value
            else:
                # Convert other types to string for storage
                hf_sample[feature_key] = str(value)
        
        hf_data.append(hf_sample)
    
    return hf_data

def main():
    ###########################
    # Compute the action queues
    ###########################

    HF_USERNAME = "fracapuano"
    DATASET_NAME = "act_stacking_action_chunks"
    stacking_dataset = "lerobot/svla_so100_stacking"

    pipeline = ACTInferencePipeline(
        policy_repo_id=DEFAULT_ACT_MODEL,
        dataset_repo_id=stacking_dataset,
        device="mps"
    )

    all_data = []
    n_batches = 100

    dataloader = DataLoader(pipeline.dataset, batch_size=16, shuffle=False)

    print("\n--- Generating and collecting action queues ---")
    max_batches = min(n_batches, len(dataloader))

    for i, batch in tqdm(enumerate(dataloader), total=max_batches):
        # Get a batch of action chunks, shape: (batch_size, n_actions, action_dim)
        action_chunk_batch = pipeline.get_action_chunk(batch)
        batch_size = action_chunk_batch.shape[0]

        # Store each sample in the batch as a dictionary with tensors preserved
        for sample_idx in range(batch_size):
            sample_data = {
                # Store observation data (preserve tensors)
                'observations': {
                    key: (batch[key][sample_idx].detach().cpu() if isinstance(batch[key], torch.Tensor) 
                        else batch[key][sample_idx]) 
                    for key in batch
                },
                # Store action chunks as tensor
                'action_chunks': action_chunk_batch[sample_idx].detach().cpu(),
                # Store metadata
                'batch_idx': i,
                'sample_idx': sample_idx,
                'episode_idx': batch.get('episode_index', [None])[sample_idx] if 'episode_index' in batch else None,
                'frame_idx': batch.get('frame_index', [None])[sample_idx] if 'frame_index' in batch else None,
            }
            all_data.append(sample_data)
        
        if i == max_batches-1:
            break

    ###########################
    # Push to Hugging Face Hub
    ###########################

    # Convert data for HF Dataset
    print("\n--- Preparing data for Hugging Face Dataset ---")
    hf_data = prepare_hf_dataset_data(all_data) # TODO: this is taking a lot of time

    # Inspect first sample to understand data structure
    sample_obs = all_data[0]['observations']
    action_shape = all_data[0]['action_chunks'].shape
    
    print(f"Action chunks shape: {action_shape}")
    print(f"Sample observation keys: {list(sample_obs.keys())}")
    
    # Debug: inspect the converted data
    print(f"\nFirst HF sample keys: {list(hf_data[0].keys())}")
    print(f"Action chunks type in HF data: {type(hf_data[0]['action_chunks'])}")
    print(f"Action chunks shape in HF data: {hf_data[0]['action_chunks'].shape}")
    
    # Check for any problematic values
    for key, value in hf_data[0].items():
        if key.startswith('obs_'):
            print(f"{key}: type={type(value)}, shape={getattr(value, 'shape', 'no shape')}")
    
    # Let's just create the dataset without explicit features first and let HF infer them
    # This is often more robust for mixed data types
    print("\nCreating Hugging Face Dataset (auto-inferring features)...")
    
    try:
        # Try without explicit features first - let HF infer
        dataset = Dataset.from_list(hf_data)
        print("✅ Successfully created dataset with auto-inferred features")
        
    except Exception as e:
        print(f"Auto-inference failed: {e}")
        print("Trying with simplified explicit features...")
        
        # Fallback to very simple feature definitions
        features_dict = {
            'action_chunks': Sequence(Sequence(Value('float32'))),  # Nested sequence for 2D
            'batch_idx': Value('int32'),
            'sample_idx': Value('int32'), 
            'episode_idx': Value('int32'),
            'frame_idx': Value('int32'),
        }

        # Add observation features with simple types
        for key, value in sample_obs.items():
            feature_key = f'obs_{key.replace(".", "_")}'
            if isinstance(value, torch.Tensor):
                if len(value.shape) == 0:  # scalar tensor
                    features_dict[feature_key] = Value('float32')
                elif len(value.shape) == 1:
                    features_dict[feature_key] = Sequence(Value('float32'))
                elif len(value.shape) == 2:
                    features_dict[feature_key] = Sequence(Sequence(Value('float32')))
                elif len(value.shape) == 3:
                    features_dict[feature_key] = Sequence(Sequence(Sequence(Value('float32'))))
                else:
                    # Flatten to 1D sequence
                    print(f"Warning: {key} has shape {value.shape}, flattening to 1D")
                    features_dict[feature_key] = Sequence(Value('float32'))
            else:
                features_dict[feature_key] = Value('string')
        
        print(f"Defined features: {list(features_dict.keys())}")
        features = Features(features_dict)
        dataset = Dataset.from_list(hf_data, features=features)

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Dataset features: {list(dataset.features.keys())}")

    # Try to get action chunks info
    try:
        first_sample = dataset[0]
        if hasattr(first_sample['action_chunks'], 'shape'):
            print(f"Action chunks shape: {first_sample['action_chunks'].shape}")
        else:
            print(f"Action chunks type: {type(first_sample['action_chunks'])}")
            if isinstance(first_sample['action_chunks'], list):
                print(f"Action chunks length: {len(first_sample['action_chunks'])}")
    except Exception as e:
        print(f"Could not inspect action chunks: {e}")

    # Push to Hugging Face Hub
    print(f"\n--- Pushing to Hugging Face Hub: {HF_USERNAME}/{DATASET_NAME} ---")
    try:
        dataset.push_to_hub(
            f"{HF_USERNAME}/{DATASET_NAME}",
            private=False,  # Set to True if you want a private dataset
            commit_message="Add SmolVLA action queues for stacking task"
        )
        print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{HF_USERNAME}/{DATASET_NAME}")
        
    except Exception as e:
        print(f"❌ Error pushing to hub: {e}")
        print("Make sure you're logged in with `huggingface-cli login` and have set the correct username")
        
        # Fallback: save locally as well
        print("\n--- Saving locally as fallback ---")
        dataset.save_to_disk("smolvla_stacking_dataset")
        print("Saved dataset locally to 'smolvla_stacking_dataset' directory")


if __name__ == "__main__":
    main()