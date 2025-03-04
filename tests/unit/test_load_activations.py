# tests/unit/test_load_activations.py
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from pattern_lens.load_activations import (
	ActivationsMismatchError,
	ActivationsMissingError,
	InvalidPromptError,
	augment_prompt_with_hash,
	load_activations,
)

TEMP_DIR: Path = Path("tests/_temp")


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
	prompt_dir.mkdir(parents=True)

	# Create a dummy prompt.json file
	with open(prompt_dir / "prompt.json", "w") as f:
		json.dump(prompt, f)

	# Create a dummy activations.npz file
	fake_activations = {
		"blocks.0.attn.hook_pattern": np.random.rand(1, 2, 10, 10).astype(np.float32),
	}
	np.savez(prompt_dir / "activations.npz", **fake_activations)

	# Test loading with numpy format
	with mock.patch(
		"pattern_lens.load_activations.compare_prompt_to_loaded",
	) as mock_compare:
		path, cache = load_activations(
			model_name=model_name,
			prompt=prompt,
			save_path=temp_dir,
			return_fmt="numpy",
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
			return_fmt="numpy",
		)

	# Create the necessary directory structure
	prompt_dir = temp_dir / model_name / "prompts" / prompt["hash"]
	prompt_dir.mkdir(parents=True)

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
			return_fmt="numpy",
		)

	# Fix the prompt file
	with open(prompt_dir / "prompt.json", "w") as f:
		json.dump(prompt, f)

	# Test with missing activations file
	with pytest.raises(FileNotFoundError):
		load_activations(
			model_name=model_name,
			prompt=prompt,
			save_path=temp_dir,
			return_fmt="numpy",
		)
