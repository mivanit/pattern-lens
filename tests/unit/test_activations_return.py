# tests/unit/test_activations_return.py
from pathlib import Path
from unittest import mock

import pytest
import torch
from transformer_lens import HookedTransformer  # type: ignore[import-untyped]

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
	model = HookedTransformer.from_pretrained("pythia-14m")
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# Test with an invalid return_cache value
	with pytest.raises(ValueError, match="invalid return_cache"):
		compute_activations(
			prompt=prompt,
			model=model,
			save_path=temp_dir,
			# intentionally invalid
			return_cache="invalid",  # type: ignore[call-overload]
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
