from pathlib import Path
from unittest import mock

import numpy as np
import torch
from transformer_lens import HookedTransformer  # type: ignore[import-untyped]

from pattern_lens.activations import compute_activations, get_activations
from pattern_lens.load_activations import ActivationsMissingError

TEMP_DIR: Path = Path("tests/_temp")


class MockHookedTransformer:
	"""Mock of HookedTransformer for testing compute_activations and get_activations."""

	def __init__(self, model_name="test-model", n_layers=2, n_heads=2):
		self.model_name = model_name
		self.cfg = mock.MagicMock()
		self.cfg.n_layers = n_layers
		self.cfg.n_heads = n_heads
		self.tokenizer = mock.MagicMock()
		self.tokenizer.tokenize.return_value = ["test", "tokens"]

	def eval(self):
		return self

	def run_with_cache(self, prompt_str, names_filter=None, return_type=None):  # noqa: ARG002
		"""Mock run_with_cache to return fake attention patterns."""
		# Create a mock activation cache with appropriately shaped attention patterns
		cache: dict[str, torch.Tensor] = {}
		for i in range(self.cfg.n_layers):
			# [1, n_heads, n_ctx, n_ctx] tensor, where n_ctx is len(prompt_str)
			n_ctx: int = len(prompt_str)
			attn_pattern: torch.Tensor = torch.rand(
				1,
				self.cfg.n_heads,
				n_ctx,
				n_ctx,
			).float()
			cache[f"blocks.{i}.attn.hook_pattern"] = attn_pattern

		return None, cache


def test_compute_activations_stack_heads():
	"""Test compute_activations with stack_heads=True."""
	# Setup
	temp_dir: Path = TEMP_DIR / "test_compute_activations_stack_heads"
	model: HookedTransformer = HookedTransformer.from_pretrained("pythia-14m")
	prompt: dict[str, str] = {"text": "test prompt", "hash": "testhash123"}

	# Test with return_cache=None
	path, result = compute_activations(
		prompt=prompt,
		model=model,
		save_path=temp_dir,
		return_cache=None,
		stack_heads=True,
	)

	# Check return values
	assert (
		path
		== temp_dir
		/ model.cfg.model_name
		/ "prompts"
		/ prompt["hash"]
		/ "activations-blocks.-.attn.hook_pattern.npy"
	)
	assert result is None

	# Check the file was created and has correct shape
	assert path.exists()
	loaded = np.load(path)
	assert loaded.shape[:3] == (
		1,
		model.cfg.n_layers,
		model.cfg.n_heads,
	)
	assert loaded.shape[3] == loaded.shape[4]

	# Test with return_cache="numpy"
	path, result = compute_activations(
		prompt=prompt,
		model=model,
		save_path=temp_dir,
		return_cache="numpy",
		stack_heads=True,
	)

	# Check return values
	assert isinstance(result, np.ndarray)
	assert result.shape[:3] == (
		1,
		model.cfg.n_layers,
		model.cfg.n_heads,
	)
	assert result.shape[3] == result.shape[4]


def test_compute_activations_no_stack():
	"""Test compute_activations with stack_heads=False."""
	# Setup
	temp_dir = TEMP_DIR / "test_compute_activations_no_stack"
	model = HookedTransformer.from_pretrained("pythia-14m")
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# Test with return_cache="numpy"
	path, result = compute_activations(
		prompt=prompt,
		model=model,
		save_path=temp_dir,
		return_cache="numpy",
		stack_heads=False,
	)

	# Check return values
	assert (
		path
		== temp_dir
		/ model.cfg.model_name
		/ "prompts"
		/ prompt["hash"]
		/ "activations.npz"
	)
	assert isinstance(result, dict)

	# Check that the keys have the expected form and values have the right shape
	for i in range(model.cfg.n_layers):
		key = f"blocks.{i}.attn.hook_pattern"
		assert key in result
		assert result[key].shape[:2] == (
			1,
			model.cfg.n_heads,
		)
		assert result[key].shape[2] == result[key].shape[3]


def test_get_activations_missing():
	"""Test get_activations when activations don't exist."""
	temp_dir = TEMP_DIR / "test_get_activations_missing"
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# Patch the load_activations and compute_activations functions
	with (
		mock.patch("pattern_lens.activations.load_activations") as mock_load,
		mock.patch("pattern_lens.activations.compute_activations") as mock_compute,
	):
		# Set up load_activations to fail
		mock_load.side_effect = ActivationsMissingError("Not found")

		# Set up mock model and compute_activations
		model = HookedTransformer.from_pretrained("pythia-14m")
		mock_compute.return_value = (Path("mock/path"), {"mock": "cache"})

		# Call get_activations
		path, cache = get_activations(
			prompt=prompt,
			model=model,
			save_path=temp_dir,
			return_cache="numpy",
		)

		# Check that compute_activations was called with the right arguments
		mock_compute.assert_called_once()
		args, kwargs = mock_compute.call_args
		assert kwargs["prompt"] == prompt
		assert kwargs["save_path"] == temp_dir
		assert kwargs["return_cache"] == "numpy"
