# tests/unit/test_activations_return.py
from pathlib import Path
from unittest import mock

import pytest
import torch
from jaxtyping import TypeCheckError
from transformer_lens import (  # type: ignore[import-untyped]
	HookedTransformer,
	HookedTransformerConfig,
)

from pattern_lens.activations import compute_activations, get_activations

TEMP_DIR: Path = Path("tests/.temp")


class MockHookedTransformer(HookedTransformer):
	"""Mock of HookedTransformer for testing compute_activations and get_activations.

	Inherits from HookedTransformer so beartype/jaxtyping runtime checks pass.
	"""

	def __init__(self, model_name="test-model", n_layers=2, n_heads=2):
		cfg = HookedTransformerConfig(
			n_layers=n_layers,
			d_model=n_heads * 8,
			n_ctx=256,
			d_head=8,
			d_vocab=100,
			attn_only=True,
			init_weights=False,
			model_name=model_name,
			device="cpu",
		)
		super().__init__(cfg, move_to_device=False)
		self.tokenizer = mock.MagicMock()
		self.tokenizer.tokenize.return_value = ["test", "tokens"]

	def run_with_cache(
		self,
		input,  # noqa: A002
		names_filter=None,  # noqa: ARG002
		return_type=None,  # noqa: ARG002
		**kwargs: object,  # noqa: ARG002
	):
		"""Mock run_with_cache to return fake attention patterns."""
		prompt_str = input
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
	_path, result = compute_activations(  # type: ignore[call-overload]
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
	_path, result = compute_activations(  # type: ignore[call-overload]
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
	"""Test compute_activations with an invalid return_cache value.

	With jaxtyping+beartype enabled, the invalid literal is rejected at the
	type-checking layer before the function body runs, raising TypeCheckError.
	"""
	temp_dir = TEMP_DIR / "test_compute_activations_invalid_return"
	model = MockHookedTransformer(n_layers=3, n_heads=4)
	prompt = {"text": "test prompt", "hash": "testhash123"}

	with pytest.raises(TypeCheckError):
		compute_activations(  # type: ignore[call-overload]
			prompt=prompt,
			model=model,
			save_path=temp_dir,
			# intentionally invalid
			return_cache="invalid",
			stack_heads=True,
		)


def test_get_activations_torch_return():
	"""Test get_activations with return_cache="torch" and mocked load_activations.

	load_activations always returns numpy arrays now; get_activations
	should convert them to torch tensors when return_cache="torch".
	"""
	import numpy as np  # noqa: PLC0415

	temp_dir = TEMP_DIR / "test_get_activations_torch_return"
	prompt = {"text": "test prompt", "hash": "testhash123"}
	model = MockHookedTransformer(model_name="test-model")

	# load_activations now always returns numpy arrays
	with mock.patch("pattern_lens.activations.load_activations") as mock_load:
		mock_cache = {
			"blocks.0.attn.hook_pattern": np.random.rand(
				1,
				2,
				len(prompt["text"]),
				len(prompt["text"]),
			).astype(np.float32),
			"blocks.1.attn.hook_pattern": np.random.rand(
				1,
				2,
				len(prompt["text"]),
				len(prompt["text"]),
			).astype(np.float32),
		}
		mock_load.return_value = (Path("mock/path"), mock_cache)

		# Call get_activations with torch return format
		_path, cache = get_activations(  # type: ignore[call-overload]
			prompt=prompt,
			model=model,
			save_path=temp_dir,
			return_cache="torch",
		)

		# Check that get_activations converted numpy to torch tensors
		assert isinstance(cache, dict)
		for key, value in cache.items():
			assert isinstance(key, str)
			assert isinstance(value, torch.Tensor)


def test_get_activations_none_return():
	"""Test get_activations with return_cache=None uses the fast existence-check path.

	When return_cache=None and the activations already exist on disk,
	get_activations should use activations_exist (cheap file-existence check)
	instead of load_activations (which would decompress the full .npz).
	"""
	temp_dir = TEMP_DIR / "test_get_activations_none_return"
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# pass model as a string -- get_activations accepts HookedTransformer | str,
	# and the fast path (return_cache=None) only needs the model name for path construction
	with mock.patch("pattern_lens.activations.activations_exist", return_value=True):
		path, cache = get_activations(  # type: ignore[call-overload]
			prompt=prompt,
			model="test-model",
			save_path=temp_dir,
			return_cache=None,
		)

		# Check that we got the reconstructed path but no cache
		expected_path = (
			temp_dir / "test-model" / "prompts" / prompt["hash"] / "activations.npz"
		)
		assert path == expected_path
		assert cache is None
