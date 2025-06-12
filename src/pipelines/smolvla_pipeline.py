from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
import random
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.constants import ACTION
from lerobot.common.policies.utils import populate_queues

import line_profiler
from memory_profiler import profile


class SmolVLAInferencePipeline:
    """A pipeline for running inference with a SmolVLA policy on a dataset."""

    def __init__(self, policy_repo_id: str, dataset_repo_id: str, device: str | None = None):
        """
        Initializes the inference pipeline.

        Args:
            policy_repo_id: The Hugging Face Hub repository ID of the pretrained policy.
            dataset_repo_id: The Hugging Face Hub repository ID of the dataset.
            device: The device to run inference on. If None, it will be auto-detected.
            n_episodes: The number of random episodes to subsample from the dataset. If None, all episodes are used.
        """
        self.policy_name = policy_repo_id
        self.dataset_name = dataset_repo_id

        self.device = device or ("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
        print(f"Using device: {self.device}")

        # Load dataset to get stats and metadata
        self.dataset = LeRobotDataset(dataset_repo_id)
        print("Loaded dataset: ", self.dataset)

        # Load policy
        model = SmolVLAPolicy.from_pretrained(policy_repo_id)
        model.to(self.device)
        
        self.policy = model
        
        # Inject normalization stats from the dataset into the policy's state dictionary.
        self._inject_normalization_stats()
        self.policy = torch.compile(self.policy)
        
        print("Inference pipeline is ready.")
    
    @property
    def name(self):
        # removing HF's user name
        model_name = self.policy_name.split("/")[-1]
        dataset_name = self.dataset_name.split("/")[-1]

        return f"{model_name}-{dataset_name}"

    def _inject_normalization_stats(self):
        """Manually loads normalization stats from the dataset into the policy's state dictionary."""
        stats = self.dataset.meta.stats
        pol_state_dict = self.policy.state_dict()

        keys_to_update = {
            "normalize_inputs.buffer_observation_state.mean": ("observation.state", "mean"),
            "normalize_inputs.buffer_observation_state.std": ("observation.state", "std"),
            "normalize_targets.buffer_action.mean": ("action", "mean"),
            "normalize_targets.buffer_action.std": ("action", "std"),
            "unnormalize_outputs.buffer_action.mean": ("action", "mean"),
            "unnormalize_outputs.buffer_action.std": ("action", "std"),
        }

        for pol_key, (stat_key, stat_type) in keys_to_update.items():
            pol_state_dict[pol_key] = torch.from_numpy(stats[stat_key][stat_type])

        self.policy.load_state_dict(pol_state_dict)
        print("Normalization stats injected into the policy.")

    def _prepare_observation(self, batch: dict) -> dict:
        """
        Prepares a batch of samples from the dataset for inference.
        This involves moving tensors to the correct device,
        and remapping image keys to match the policy's expectations.
        """
        observation = {
            "observation.state": batch["observation.state"].to(self.device),
            "observation.image": batch["observation.images.top"].to(self.device),
            "observation.image2": batch["observation.images.wrist"].to(self.device),
            "task": batch["task"],
        }
        return observation
    
    # Uncomment the following two decorators to profile both time and memory.
    #@line_profiler.profile
    #@profile
    def get_action_chunk(self, batch: dict, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Maps a batch of observations to a batch of action chunks, one for each observation in the batch.
        NOTE: Actions are not unnormalized, so they are in the range N(0,1)"""
        with torch.no_grad():
            self.policy.eval()
            
            batch = self._prepare_observation(batch)
            normalized_batch = self.policy.normalize_inputs(batch)
            self.policy._queues = populate_queues(self.policy._queues, normalized_batch, exclude_keys=[ACTION])

            for k in batch:
                if k in self.policy._queues:
                    batch[k] = torch.stack(list(self.policy._queues[k]), dim=1)

            images, img_masks = self.policy.prepare_images(batch)
            state = self.policy.prepare_state(batch)
            lang_tokens, lang_masks = self.policy.prepare_language(batch)

            actions = self.policy.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )

            # Unpad actions
            original_action_dim = self.policy.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            return actions


if __name__ == "__main__":
    import cProfile
    import pstats
    import time
    from torch.utils.data import DataLoader

    # Initialize the pipeline
    inference_pipeline = SmolVLAInferencePipeline(
        policy_repo_id="lerobot/smolvla_base",
        dataset_repo_id="lerobot/svla_so100_stacking",
    )

    loader = DataLoader(inference_pipeline.dataset, batch_size=16)
    sample = next(iter(loader))

    print("\n--- Running inference on a single sample with cProfile ---")

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    first_action = inference_pipeline.get_action_chunk(sample)
    end = time.perf_counter()
    print(f"Inference time: {end - start:.3e} seconds")

    profiler.disable()

    print("Computed action for the sample:")
    
    ps = pstats.Stats(profiler).sort_stats("cumtime")