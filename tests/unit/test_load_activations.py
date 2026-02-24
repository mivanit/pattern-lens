# tests/unit/test_load_activations.py
import json
import typing
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import pattern_lens.load_activations as load_activations_mod
from pattern_lens.load_activations import (
	ActivationsMismatchError,
	ActivationsMissingError,
	InvalidPromptError,
	augment_prompt_with_hash,
	load_activations,
)

TEMP_DIR: Path = Path("tests/.temp")


def test_augment_prompt_with_hash():
	"""Test adding hash to prompt."""
	# Test with a prompt that doesn't have a hash
	prompt_no_hash = {"text": "test prompt"}
	result = augment_prompt_with_hash(prompt_no_hash)

	# Check that the hash was added and is deterministic
	assert "hash" in result
	assert isinstance(result["hash"], str)

	# Save the hash for comparison
	first_hash = result["hash"]

	# Test that calling it again doesn't change the hash
	result = augment_prompt_with_hash(prompt_no_hash)
	assert result["hash"] == first_hash

	# Test with a prompt that already has a hash
	prompt_with_hash = {"text": "test prompt", "hash": "existing-hash"}
	result = augment_prompt_with_hash(prompt_with_hash)

	# Check that the hash wasn't changed
	assert result["hash"] == "existing-hash"

	# Test with an invalid prompt (no text or hash)
	with pytest.raises(InvalidPromptError):
		augment_prompt_with_hash({"other_field": "value"})


def test_load_activations_success():
	"""Test successful loading of activations."""
	# Setup
	temp_dir = TEMP_DIR / "test_load_activations_success"
	model_name = "test-model"
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# Create the necessary directory structure
	prompt_dir = temp_dir / model_name / "prompts" / prompt["hash"]
	prompt_dir.mkdir(parents=True, exist_ok=True)

	# Create a dummy prompt.json file
	with open(prompt_dir / "prompt.json", "w") as f:
		json.dump(prompt, f)

	# Create a dummy activations.npz file
	fake_activations = {
		"blocks.0.attn.hook_pattern": np.random.rand(1, 2, 10, 10).astype(np.float32),
	}
	np.savez(prompt_dir / "activations.npz", **fake_activations)  # type: ignore[arg-type]

	# Test loading (always returns numpy)
	with mock.patch(
		"pattern_lens.load_activations.compare_prompt_to_loaded",
	) as mock_compare:
		path, cache = load_activations(
			model_name=model_name,
			prompt=prompt,
			save_path=temp_dir,
		)

		# Check that the path is correct
		assert path == prompt_dir / "activations.npz"

		# Check that the cache has the right structure
		assert isinstance(cache, dict)
		assert "blocks.0.attn.hook_pattern" in cache
		assert cache["blocks.0.attn.hook_pattern"].shape == (1, 2, 10, 10)

		# Check that the prompt was compared
		mock_compare.assert_called_once_with(prompt, prompt)


def test_load_activations_errors():
	"""Test error handling in load_activations."""
	# Setup
	temp_dir = TEMP_DIR / "test_load_activations_errors"
	model_name = "test-model"
	prompt = {"text": "test prompt", "hash": "testhash123"}

	# Test with missing prompt file
	with pytest.raises(ActivationsMissingError):
		load_activations(
			model_name=model_name,
			prompt=prompt,
			save_path=temp_dir,
		)

	# Create the necessary directory structure
	prompt_dir = temp_dir / model_name / "prompts" / prompt["hash"]
	prompt_dir.mkdir(parents=True, exist_ok=True)

	# Create a prompt.json file with different content
	different_prompt = {"text": "different prompt", "hash": prompt["hash"]}
	with open(prompt_dir / "prompt.json", "w") as f:
		json.dump(different_prompt, f)

	# Test with mismatched prompt
	with pytest.raises(ActivationsMismatchError):
		load_activations(
			model_name=model_name,
			prompt=prompt,
			save_path=temp_dir,
		)

	# Fix the prompt file
	with open(prompt_dir / "prompt.json", "w") as f:
		json.dump(prompt, f)

	# Test with missing activations file (prompt.json exists but activations.npz does not)
	with pytest.raises(ActivationsMissingError, match="Activations file"):
		load_activations(
			model_name=model_name,
			prompt=prompt,
			save_path=temp_dir,
		)


def test_load_activations_missing_npz_is_subclass_of_file_not_found():
	"""ActivationsMissingError for missing activations.npz is also a FileNotFoundError."""
	temp_dir = TEMP_DIR / "test_load_activations_missing_npz_subclass"
	model_name = "test-model"
	prompt = {"text": "test prompt", "hash": "testhash_sub"}

	prompt_dir = temp_dir / model_name / "prompts" / prompt["hash"]
	prompt_dir.mkdir(parents=True, exist_ok=True)

	with open(prompt_dir / "prompt.json", "w") as f:
		json.dump(prompt, f)

	# Should raise ActivationsMissingError, which is also a FileNotFoundError
	with pytest.raises(ActivationsMissingError) as exc_info:
		load_activations(
			model_name=model_name,
			prompt=prompt,
			save_path=temp_dir,
		)
	assert isinstance(exc_info.value, FileNotFoundError)


def test_load_activations_return_fmt_deprecation():
	"""Test that return_fmt='numpy' emits DeprecationWarning."""
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		with pytest.raises(ActivationsMissingError):
			load_activations(
				model_name="test-model",
				prompt={"text": "test", "hash": "test"},
				save_path=Path("/nonexistent"),
				return_fmt="numpy",
			)
		deprecation_warnings = [
			x for x in w if issubclass(x.category, DeprecationWarning)
		]
		assert len(deprecation_warnings) == 1
		assert "return_fmt is deprecated" in str(deprecation_warnings[0].message)


def test_load_activations_return_fmt_invalid():
	"""Test that invalid return_fmt (including 'torch') raises ValueError."""
	with pytest.raises(ValueError, match="Invalid return_fmt"):
		load_activations(
			model_name="test-model",
			prompt={"text": "test", "hash": "test"},
			save_path=Path("/nonexistent"),
			return_fmt="torch",
		)
	with pytest.raises(ValueError, match="Invalid return_fmt"):
		load_activations(
			model_name="test-model",
			prompt={"text": "test", "hash": "test"},
			save_path=Path("/nonexistent"),
			return_fmt="invalid",
		)


def test_annotations_resolvable_by_beartype():
	"""PEP 563 annotations must be eval()-able against module globals.

	beartype (via jaxtyping's import hook) eval()s annotation strings at
	decoration time. If any referenced name is missing from module globals
	(e.g. behind TYPE_CHECKING), it raises BeartypePep563Exception.
	"""
	for name in (
		"load_activations",
		"activations_exist",
		"augment_prompt_with_hash",
		"compare_prompt_to_loaded",
	):
		func = getattr(load_activations_mod, name)
		typing.get_type_hints(func)
