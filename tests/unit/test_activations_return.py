# tests/unit/test_activations_return.py
from pathlib import Path
from unittest import mock

import pytest
import torch

from pattern_lens.activations import compute_activations, get_activations

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
		cache = {}
		for i in range(self.cfg.n_layers):
			# [1, n_heads, n_ctx, n_ctx] tensor, where n_ctx is len(prompt_str)
			n_ctx = len(prompt_str)
			attn_pattern = torch.rand(
				1,
				self.cfg.n_heads,
				n_ctx,
				n_ctx,
			).float()
			cache[f"blocks.{i}.attn.hook_pattern"] = attn_pattern

		return None, cache


def test_compute_activations_torch_return():
	"""Test compute_activations with return_cache="torch"."""
	# Setup
	temp_dir = TEMP_DIR / "test_compute_activations_torch_return"
	model = MockHookedTransformer(n_layers=3, n_heads=4)
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# Test with stack_heads=True
	path, result = compute_activations(
		prompt=prompt,
		model=model,
		save_path=temp_dir,
		return_cache="torch",
		stack_heads=True,
	)

	# Check return values
	assert isinstance(result, torch.Tensor)
	assert result.shape == (
		1,
		model.cfg.n_layers,
		model.cfg.n_heads,
		len(prompt["text"]),
		len(prompt["text"]),
	)

	# Test with stack_heads=False
	path, result = compute_activations(
		prompt=prompt,
		model=model,
		save_path=temp_dir,
		return_cache="torch",
		stack_heads=False,
	)

	# Check return values
	assert isinstance(result, dict)
	for i in range(model.cfg.n_layers):
		key = f"blocks.{i}.attn.hook_pattern"
		assert key in result
		assert isinstance(result[key], torch.Tensor)


def test_compute_activations_invalid_return():
	"""Test compute_activations with an invalid return_cache value."""
	# Setup
	temp_dir = TEMP_DIR / "test_compute_activations_invalid_return"
	model = MockHookedTransformer()
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# Test with an invalid return_cache value
	with pytest.raises(ValueError, match="invalid return_cache"):
		compute_activations(
			prompt=prompt,
			model=model,
			save_path=temp_dir,
			return_cache="invalid",  # Invalid value
			stack_heads=True,
		)


def test_get_activations_torch_return():
	"""Test get_activations with return_cache="torch" and mocked load_activations."""
	temp_dir = TEMP_DIR / "test_get_activations_torch_return"
	prompt = {"text": "test prompt", "hash": "testhash123"}
	model = MockHookedTransformer(model_name="test-model")

	# Create a mock for load_activations that returns torch tensors
	with mock.patch("pattern_lens.activations.load_activations") as mock_load:
		mock_cache = {
			"blocks.0.attn.hook_pattern": torch.rand(
				1,
				2,
				len(prompt["text"]),
				len(prompt["text"]),
			),
			"blocks.1.attn.hook_pattern": torch.rand(
				1,
				2,
				len(prompt["text"]),
				len(prompt["text"]),
			),
		}
		mock_load.return_value = (Path("mock/path"), mock_cache)

		# Call get_activations with torch return format
		path, cache = get_activations(
			prompt=prompt,
			model=model,
			save_path=temp_dir,
			return_cache="torch",
		)

		# Check that we got torch tensors back
		assert isinstance(cache, dict)
		for key, value in cache.items():
			assert isinstance(key, str)
			assert isinstance(value, torch.Tensor)


def test_get_activations_none_return():
	"""Test get_activations with return_cache=None."""
	temp_dir = TEMP_DIR / "test_get_activations_none_return"
	prompt = {"text": "test prompt", "hash": "testhash123"}
	model = MockHookedTransformer(model_name="test-model")

	# Create a mock for load_activations that returns a path but no cache
	with mock.patch("pattern_lens.activations.load_activations") as mock_load:
		mock_path = Path("mock/path")
		mock_load.return_value = (mock_path, {})  # Cache will be ignored

		# Call get_activations with None return format
		path, cache = get_activations(
			prompt=prompt,
			model=model,
			save_path=temp_dir,
			return_cache=None,
		)

		# Check that we got the path but no cache
		assert path == mock_path
		assert cache is None


def test_get_activations_compute_device():
	"""Test that get_activations correctly handles device for computing activations."""
	temp_dir = TEMP_DIR / "test_get_activations_compute_device"
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# Use a mock HookedTransformer.from_pretrained
	with mock.patch("pattern_lens.activations.HookedTransformer") as mock_ht:
		# Configure the mock to simulate compute_activations
		mock_model = MockHookedTransformer(model_name="gpt2")
		mock_ht.from_pretrained.return_value = mock_model

		# Also mock load_activations to raise ActivationsMissingError
		with mock.patch("pattern_lens.activations.load_activations") as mock_load:
			from pattern_lens.load_activations import ActivationsMissingError

			mock_load.side_effect = ActivationsMissingError("Not found")

			# Mock compute_activations to return a simple result
			with mock.patch(
				"pattern_lens.activations.compute_activations",
			) as mock_compute:
				mock_compute.return_value = (Path("mock/path"), {})

				# Call get_activations with a string model name
				get_activations(
					prompt=prompt,
					model="gpt2",
					save_path=temp_dir,
					allow_disk_cache=True,
				)

				# Check that from_pretrained was called with the right model name
				mock_ht.from_pretrained.assert_called_once_with("gpt2")

				# Check that compute_activations was called with the model instance
				mock_compute.assert_called_once()
				args, kwargs = mock_compute.call_args
				assert kwargs["model"] == mock_model
