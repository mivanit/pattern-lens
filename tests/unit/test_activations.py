import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from pattern_lens.activations import compute_activations, get_activations
from pattern_lens.load_activations import ActivationsMissingError


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

	def run_with_cache(self, prompt_str, names_filter=None, return_type=None):
		"""Mock run_with_cache to return fake attention patterns."""
		# Create a mock activation cache with appropriately shaped attention patterns
		cache = {}
		for i in range(self.cfg.n_layers):
			# [1, n_heads, n_ctx, n_ctx] tensor, where n_ctx is len(prompt_str)
			n_ctx = len(prompt_str)
			attn_pattern = np.random.rand(1, self.cfg.n_heads, n_ctx, n_ctx).astype(
				np.float32,
			)
			cache[f"blocks.{i}.attn.hook_pattern"] = attn_pattern

		return None, cache


def test_compute_activations_stack_heads():
	"""Test compute_activations with stack_heads=True."""
	# Setup
	temp_dir = Path(tempfile.mkdtemp())
	model = MockHookedTransformer(n_layers=3, n_heads=4)
	prompt = {"text": "test prompt", "hash": "testhash123"}

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
		/ model.model_name
		/ "prompts"
		/ prompt["hash"]
		/ "activations-blocks.-.attn.hook_pattern.npy"
	)
	assert result is None

	# Check the file was created and has correct shape
	assert path.exists()
	loaded = np.load(path)
	assert loaded.shape == (
		model.cfg.n_layers,
		model.cfg.n_heads,
		len(prompt["text"]),
		len(prompt["text"]),
	)

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
	assert result.shape == (
		model.cfg.n_layers,
		model.cfg.n_heads,
		len(prompt["text"]),
		len(prompt["text"]),
	)


def test_compute_activations_no_stack():
	"""Test compute_activations with stack_heads=False."""
	# Setup
	temp_dir = Path(tempfile.mkdtemp())
	model = MockHookedTransformer(n_layers=2, n_heads=2)
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
		== temp_dir / model.model_name / "prompts" / prompt["hash"] / "activations.npz"
	)
	assert isinstance(result, dict)

	# Check that the keys have the expected form and values have the right shape
	for i in range(model.cfg.n_layers):
		key = f"blocks.{i}.attn.hook_pattern"
		assert key in result
		assert result[key].shape == (
			1,
			model.cfg.n_heads,
			len(prompt["text"]),
			len(prompt["text"]),
		)


def test_get_activations_missing():
	"""Test get_activations when activations don't exist."""
	temp_dir = Path(tempfile.mkdtemp())
	model = "test-model"  # String model name, should trigger model loading if activations are missing
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# Patch the load_activations and compute_activations functions
	with (
		mock.patch("pattern_lens.activations.load_activations") as mock_load,
		mock.patch("pattern_lens.activations.compute_activations") as mock_compute,
		mock.patch(
			"pattern_lens.activations.HookedTransformer.from_pretrained",
		) as mock_from_pretrained,
	):
		# Set up load_activations to fail
		mock_load.side_effect = ActivationsMissingError("Not found")

		# Set up mock model and compute_activations
		mock_model = MockHookedTransformer(model_name=model)
		mock_from_pretrained.return_value = mock_model
		mock_compute.return_value = (Path("mock/path"), {"mock": "cache"})

		# Call get_activations
		path, cache = get_activations(
			prompt=prompt, model=model, save_path=temp_dir, return_cache="numpy",
		)

		# Check that model was loaded
		mock_from_pretrained.assert_called_once_with(model)

		# Check that compute_activations was called with the right arguments
		mock_compute.assert_called_once()
		args, kwargs = mock_compute.call_args
		assert kwargs["model"] == mock_model
		assert kwargs["prompt"] == prompt
		assert kwargs["save_path"] == temp_dir
		assert kwargs["return_cache"] == "numpy"
