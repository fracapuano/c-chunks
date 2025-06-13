from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from constants import DEFAULT_ACT_MODEL

import line_profiler
from memory_profiler import profile


class ACTInferencePipeline:
    """A pipeline for running inference with a ACT policy on a dataset."""

    def __init__(self, policy_repo_id: str, dataset_repo_id: str, device: str | None = None, actions_per_chunk: int = 50):
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
        self.actions_per_chunk = actions_per_chunk

        self.device = device or ("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
        print(f"Using device: {self.device}")

        # Load dataset to get stats and metadata
        self.dataset = LeRobotDataset(dataset_repo_id)
        print("Loaded dataset: ", self.dataset)

        # Load policy
        self.policy = ACTPolicy.from_pretrained(policy_repo_id)
        self.policy.to(self.device)

        self.policy.eval()
        print("Inference pipeline is ready.")
    
    @property
    def name(self):
        # removing HF's user name
        model_name = self.policy_name.split("/")[-1]
        dataset_name = self.dataset_name.split("/")[-1]

        return f"{model_name}-{dataset_name}"

    def _prepare_observation(self, batch: dict) -> dict:
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
        """Maps a batch of observations to a batch of action chunks, one for each observation in the batch."""
        with torch.no_grad():
            # prepare observation for policy forward pass
            batch = self._prepare_observation(batch)
            batch = self.policy.normalize_inputs(batch)

            if self.policy.config.image_features:
                batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
                batch["observation.images"] = [batch[key] for key in self.policy.config.image_features]

            # forward pass outputs up to policy.config.n_action_steps != actions_per_chunk
            actions = self.policy.model(batch)[0][:, :self.actions_per_chunk]

            return actions


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # Initialize the pipeline
    inference_pipeline = ACTInferencePipeline(
        policy_repo_id=DEFAULT_ACT_MODEL,
        dataset_repo_id="lerobot/svla_so100_stacking",
    )

    loader = DataLoader(inference_pipeline.dataset, batch_size=16)
    sample = next(iter(loader))

    print("\n--- Running inference on a single sample ---")
    
    first_action = inference_pipeline.get_action_chunk(sample)
    print("Computed action for the sample:")