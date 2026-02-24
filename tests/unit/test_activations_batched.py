"""Tests for batched activation computation.

Tests verify:
- compute_activations_batched produces correct shapes per-prompt
- Trimming correctly removes padding (variable-length prompts)
- Batched results are equivalent to single-prompt results
- Prompt metadata (prompt.json) is correctly saved
- activations_exist helper function
- Cache skipping in activations_main
"""

import json
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from transformer_lens import (  # type: ignore[import-untyped]
	HookedTransformer,
	HookedTransformerConfig,
)

from pattern_lens.activations import (
	_save_activations,
	activations_main,
	compute_activations,
	compute_activations_batched,
)
from pattern_lens.load_activations import (
	InvalidPromptError,
	activations_exist,
	augment_prompt_with_hash,
)

TEMP_DIR: Path = Path("tests/.temp")


class MockHookedTransformerBatched(HookedTransformer):
	"""Mock of HookedTransformer that supports both single and batched input.

	Inherits from HookedTransformer so beartype/jaxtyping runtime checks pass.

	Tokenization rule: each character in the text becomes one token, plus a BOS token.
	So "hello" -> seq_len=6 (1 BOS + 5 chars).

	For batched input, shorter sequences are right-padded with zeros.
	Real (non-padding) positions are filled with deterministic random values
	seeded by the text content, so the same text produces the same attention
	patterns regardless of batch composition.
	"""

	def __init__(
		self,
		model_name: str = "test-model",
		n_layers: int = 2,
		n_heads: int = 2,
	):
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
		# Set up a mock tokenizer for deterministic test control
		self.tokenizer = mock.MagicMock()
		self.tokenizer.tokenize = lambda text: list(text)  # noqa: PLW0108 -- split text into individual characters

	def _seq_len(self, text: str) -> int:
		"""Actual model sequence length for a text (includes BOS)."""
		return len(text) + 1

	def to_tokens(
		self,
		text,
		prepend_bos=None,
		padding_side=None,
		move_to_device=None,
		truncate=None,
	):
		"""Return token IDs. Includes BOS, so seq_len = len(text) + 1."""
		if isinstance(text, str):
			return torch.zeros(1, self._seq_len(text), dtype=torch.long)
		else:
			seq_lens = [self._seq_len(t) for t in text]
			max_len = max(seq_lens)
			return torch.zeros(len(text), max_len, dtype=torch.long)

	def _make_deterministic_attn(self, text: str, seq_len: int) -> torch.Tensor:
		"""Generate deterministic attention values for a text.

		Returns shape [n_heads, seq_len, seq_len] with values seeded by text content.
		"""
		gen = torch.Generator()
		gen.manual_seed(hash(text) % (2**31))
		return torch.rand(self.cfg.n_heads, seq_len, seq_len, generator=gen)

	def run_with_cache(
		self,
		input,  # noqa: A002 -- matches TransformerLens API signature
		names_filter=None,  # noqa: ARG002
		return_type=None,  # noqa: ARG002
		padding_side="right",  # noqa: ARG002
		**kwargs,  # noqa: ARG002
	):
		"""Mock run_with_cache supporting both single string and list of strings.

		For batched input, pads to max length. Real positions get deterministic
		values based on text content. Padding positions are 0.
		"""
		if isinstance(input, list):
			texts = input
			batch_size = len(texts)
		else:
			texts = [input]
			batch_size = 1

		seq_lens = [self._seq_len(t) for t in texts]
		max_len = max(seq_lens)

		cache: dict[str, torch.Tensor] = {}
		for layer in range(self.cfg.n_layers):
			attn = torch.zeros(batch_size, self.cfg.n_heads, max_len, max_len)
			for b, (text, seq_len) in enumerate(zip(texts, seq_lens, strict=True)):
				attn[b, :, :seq_len, :seq_len] = self._make_deterministic_attn(
					text,
					seq_len,
				)
			cache[f"blocks.{layer}.attn.hook_pattern"] = attn

		return None, cache


def _make_prompts() -> list[dict]:
	"""Return fresh copies of test prompts (dicts get mutated during processing)."""
	return [
		{"text": "hi", "hash": "hash_short"},
		{"text": "hello world", "hash": "hash_medium"},
		{"text": "the quick brown fox jumps", "hash": "hash_long"},
	]


def _expected_seq_len(text: str) -> int:
	"""Expected model sequence length: len(text) + 1 for BOS."""
	return len(text) + 1


# ============================================================================
# Test: activations_exist
# ============================================================================


def test_activations_exist_both_present():
	"""activations_exist returns True when both prompt.json and activations.npz exist."""
	temp_dir = TEMP_DIR / "test_activations_exist_both"
	model_name = "test-model"
	prompt = {"text": "test", "hash": "exist_hash"}

	prompt_dir = temp_dir / model_name / "prompts" / prompt["hash"]
	prompt_dir.mkdir(parents=True, exist_ok=True)

	with open(prompt_dir / "prompt.json", "w") as f:
		json.dump(prompt, f)
	np.savez(prompt_dir / "activations.npz", dummy=np.zeros(1))

	assert activations_exist(model_name, prompt, temp_dir) is True


def test_activations_exist_missing_npz():
	"""activations_exist returns False when activations.npz is missing."""
	temp_dir = TEMP_DIR / "test_activations_exist_no_npz"
	model_name = "test-model"
	prompt = {"text": "test", "hash": "exist_no_npz"}

	prompt_dir = temp_dir / model_name / "prompts" / prompt["hash"]
	prompt_dir.mkdir(parents=True, exist_ok=True)

	with open(prompt_dir / "prompt.json", "w") as f:
		json.dump(prompt, f)

	assert activations_exist(model_name, prompt, temp_dir) is False


def test_activations_exist_missing_json():
	"""activations_exist returns False when prompt.json is missing."""
	temp_dir = TEMP_DIR / "test_activations_exist_no_json"
	model_name = "test-model"
	prompt = {"text": "test", "hash": "exist_no_json"}

	prompt_dir = temp_dir / model_name / "prompts" / prompt["hash"]
	prompt_dir.mkdir(parents=True, exist_ok=True)

	np.savez(prompt_dir / "activations.npz", dummy=np.zeros(1))

	assert activations_exist(model_name, prompt, temp_dir) is False


def test_activations_exist_missing_dir():
	"""activations_exist returns False when the directory doesn't exist."""
	temp_dir = TEMP_DIR / "test_activations_exist_no_dir"
	prompt = {"text": "test", "hash": "exist_no_dir"}

	assert activations_exist("test-model", prompt, temp_dir) is False


# ============================================================================
# Test: compute_activations_batched shapes
# ============================================================================


def test_compute_activations_batched_shapes():
	"""Each prompt's saved .npz has attention patterns with the correct unpadded shape."""
	temp_dir = TEMP_DIR / "test_batched_shapes"
	model = MockHookedTransformerBatched()
	prompts = _make_prompts()

	paths = compute_activations_batched(
		prompts=prompts,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	assert len(paths) == 3

	for prompt, path in zip(prompts, paths, strict=True):
		assert path.exists(), f"Missing file: {path}"

		expected_seq_len = _expected_seq_len(prompt["text"])
		with np.load(path) as data:
			for layer in range(model.cfg.n_layers):
				key = f"blocks.{layer}.attn.hook_pattern"
				assert key in data, f"Missing key {key} in {path}"
				arr = data[key]
				expected_shape = (
					1,
					model.cfg.n_heads,
					expected_seq_len,
					expected_seq_len,
				)
				assert arr.shape == expected_shape, (
					f"Wrong shape for {prompt['text']!r}: {arr.shape} != {expected_shape}"
				)


def test_compute_activations_batched_no_padding_leaks():
	"""Verify that padding values (zeros) don't appear in saved data for real positions.

	The mock fills real positions with random values > 0 (with very high probability)
	and padding positions with exactly 0. If trimming is wrong, we'd see zeros
	in places that should have random values.
	"""
	temp_dir = TEMP_DIR / "test_batched_no_padding_leaks"
	model = MockHookedTransformerBatched()
	# Use prompts with very different lengths to ensure padding exists
	prompts = [
		{"text": "ab", "hash": "hash_tiny"},  # seq_len=3
		{"text": "a" * 50, "hash": "hash_big"},  # seq_len=51
	]

	paths = compute_activations_batched(
		prompts=prompts,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	# Check the short prompt's data — if padding leaked, it would have
	# shape (1, 2, 51, 51) instead of (1, 2, 3, 3)
	with np.load(paths[0]) as data:
		arr = data["blocks.0.attn.hook_pattern"]
		assert arr.shape == (1, 2, 3, 3), (
			f"Padding leaked into short prompt: shape={arr.shape}"
		)


# ============================================================================
# Test: batched vs single equivalence
# ============================================================================


def test_batched_vs_single_equivalence():
	"""Batched results must be identical to single-prompt results.

	Process the same prompts both individually and as a batch.
	With right-padding + causal mask, real positions should have identical values.
	"""
	temp_dir_single = TEMP_DIR / "test_equivalence_single"
	temp_dir_batch = TEMP_DIR / "test_equivalence_batch"
	model = MockHookedTransformerBatched()

	prompts_single = _make_prompts()
	prompts_batch = _make_prompts()

	# Process individually
	single_paths = []
	for p in prompts_single:
		augment_prompt_with_hash(p)
		path, _ = compute_activations(  # ty: ignore[no-matching-overload]
			prompt=p,
			model=model,  # type: ignore[arg-type]
			save_path=temp_dir_single,
			return_cache=None,
		)
		single_paths.append(path)

	# Process as batch
	batch_paths = compute_activations_batched(
		prompts=prompts_batch,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir_batch,
	)

	# Compare
	assert len(single_paths) == len(batch_paths)
	for i, (single_path, batch_path) in enumerate(
		zip(single_paths, batch_paths, strict=True),
	):
		with np.load(single_path) as single_data, np.load(batch_path) as batch_data:
			single_keys = set(single_data.keys())
			batch_keys = set(batch_data.keys())
			assert single_keys == batch_keys, (
				f"Prompt {i}: key mismatch: {single_keys} != {batch_keys}"
			)

			for key in single_keys:
				single_arr = single_data[key]
				batch_arr = batch_data[key]
				assert single_arr.shape == batch_arr.shape, (
					f"Prompt {i}, key {key}: shape mismatch: "
					f"{single_arr.shape} != {batch_arr.shape}"
				)
				np.testing.assert_allclose(
					single_arr,
					batch_arr,
					rtol=0,
					atol=0,
					err_msg=f"Prompt {i}, key {key}: values differ",
				)


def test_batched_single_prompt_equivalence():
	"""Batch of size 1 must produce identical results to single-prompt compute."""
	temp_dir_single = TEMP_DIR / "test_single_equiv_single"
	temp_dir_batch = TEMP_DIR / "test_single_equiv_batch"
	model = MockHookedTransformerBatched()

	prompt_single = {"text": "hello world", "hash": "hash_1prompt"}
	prompt_batch = {"text": "hello world", "hash": "hash_1prompt"}

	augment_prompt_with_hash(prompt_single)
	single_path, _ = compute_activations(  # ty: ignore[no-matching-overload]
		prompt=prompt_single,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir_single,
		return_cache=None,
	)

	batch_paths = compute_activations_batched(
		prompts=[prompt_batch],
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir_batch,
	)

	with np.load(single_path) as single_data, np.load(batch_paths[0]) as batch_data:
		for key in single_data:
			np.testing.assert_allclose(
				single_data[key],
				batch_data[key],
				rtol=0,
				atol=0,
				err_msg=f"Key {key}: single vs batch-of-1 differ",
			)


# ============================================================================
# Test: prompt metadata
# ============================================================================


def test_compute_activations_batched_saves_prompt_metadata():
	"""Each prompt in the batch gets its own prompt.json with correct fields."""
	temp_dir = TEMP_DIR / "test_batched_metadata"
	model = MockHookedTransformerBatched()
	prompts = _make_prompts()

	compute_activations_batched(
		prompts=prompts,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	for prompt in prompts:
		prompt_dir = temp_dir / model.cfg.model_name / "prompts" / prompt["hash"]
		prompt_json_path = prompt_dir / "prompt.json"
		assert prompt_json_path.exists(), f"Missing prompt.json for {prompt['hash']}"

		with open(prompt_json_path) as f:
			saved = json.load(f)

		assert saved["text"] == prompt["text"]
		assert saved["hash"] == prompt["hash"]
		assert "tokens" in saved
		assert "n_tokens" in saved
		# n_tokens should be len(text) since tokenizer.tokenize returns list(text)
		assert saved["n_tokens"] == len(prompt["text"])
		# tokens should be the list of characters
		assert saved["tokens"] == list(prompt["text"])


# ============================================================================
# Test: file structure and path correctness
# ============================================================================


def test_compute_activations_batched_file_paths():
	"""Saved files follow the expected directory structure."""
	temp_dir = TEMP_DIR / "test_batched_paths"
	model = MockHookedTransformerBatched(model_name="my-model")
	prompts = _make_prompts()

	paths = compute_activations_batched(
		prompts=prompts,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	for prompt, path in zip(prompts, paths, strict=True):
		expected = (
			temp_dir / "my-model" / "prompts" / prompt["hash"] / "activations.npz"
		)
		assert path == expected, f"Wrong path: {path} != {expected}"


# ============================================================================
# Test: empty batch assertion
# ============================================================================


def test_compute_activations_batched_empty_raises():
	"""Empty prompt list should raise AssertionError."""
	model = MockHookedTransformerBatched()
	with pytest.raises(AssertionError, match="prompts must not be empty"):
		compute_activations_batched(
			prompts=[],
			model=model,  # type: ignore[arg-type]
			save_path=TEMP_DIR / "test_batched_empty",
		)


# ============================================================================
# Test: variable-length trimming with extreme size differences
# ============================================================================


def test_batched_extreme_length_difference():
	"""Test with prompts whose lengths differ by 10x+ to stress-test trimming."""
	temp_dir = TEMP_DIR / "test_batched_extreme_lengths"
	model = MockHookedTransformerBatched()

	prompts = [
		{"text": "x", "hash": "hash_1char"},  # seq_len=2
		{"text": "y" * 100, "hash": "hash_100char"},  # seq_len=101
	]

	paths = compute_activations_batched(
		prompts=prompts,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	# Short prompt: shape should be [1, 2, 2, 2], NOT [1, 2, 101, 101]
	with np.load(paths[0]) as data:
		arr = data["blocks.0.attn.hook_pattern"]
		assert arr.shape == (1, 2, 2, 2)

	# Long prompt: shape should be [1, 2, 101, 101]
	with np.load(paths[1]) as data:
		arr = data["blocks.0.attn.hook_pattern"]
		assert arr.shape == (1, 2, 101, 101)


def test_batched_same_length_prompts():
	"""When all prompts have the same length, no trimming needed — should still work."""
	temp_dir = TEMP_DIR / "test_batched_same_length"
	model = MockHookedTransformerBatched()

	prompts = [
		{"text": "abc", "hash": "hash_abc"},
		{"text": "def", "hash": "hash_def"},
		{"text": "ghi", "hash": "hash_ghi"},
	]

	paths = compute_activations_batched(
		prompts=prompts,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	# All should have seq_len=4 (3 chars + BOS)
	for path in paths:
		with np.load(path) as data:
			arr = data["blocks.0.attn.hook_pattern"]
			assert arr.shape == (1, 2, 4, 4)


# ============================================================================
# Test: activations_exist integration with compute_activations_batched
# ============================================================================


def test_activations_exist_after_batched_compute():
	"""activations_exist returns True for all prompts after batched computation."""
	temp_dir = TEMP_DIR / "test_exist_after_batched"
	model = MockHookedTransformerBatched()
	prompts = _make_prompts()

	compute_activations_batched(
		prompts=prompts,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	for prompt in prompts:
		assert activations_exist(model.cfg.model_name, prompt, temp_dir), (
			f"activations_exist returned False for {prompt['hash']}"
		)


# ============================================================================
# Test: saved attention values are nonzero (trimmed correctly, not padding)
# ============================================================================


def test_batched_trimmed_values_are_nonzero():
	"""Verify saved attention values are all nonzero.

	The mock fills real positions with torch.rand (uniform [0,1)) and padding
	with exactly 0. If trimming is wrong and we include padding positions,
	we'd see exact zeros in the saved data. Since rand() producing exactly 0.0
	is astronomically unlikely for the small tensors in this test, any zero
	indicates a trimming bug.
	"""
	temp_dir = TEMP_DIR / "test_batched_nonzero_values"
	model = MockHookedTransformerBatched()
	# Deliberately different lengths so padding exists in the batch
	prompts = [
		{"text": "ab", "hash": "hash_nz_short"},  # seq_len=3
		{"text": "a" * 20, "hash": "hash_nz_long"},  # seq_len=21
	]

	paths = compute_activations_batched(
		prompts=prompts,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	for prompt, path in zip(prompts, paths, strict=True):
		with np.load(path) as data:
			for key in data:
				arr = data[key]
				assert np.all(arr != 0.0), (
					f"Found zeros in {key} for prompt {prompt['text']!r} "
					f"(shape {arr.shape}). This suggests padding was not trimmed."
				)


# ============================================================================
# Test: pre-computed seq_lens parameter
# ============================================================================


def test_batched_with_precomputed_seq_lens():
	"""Passing seq_lens explicitly produces the same result as computing internally."""
	temp_dir_auto = TEMP_DIR / "test_seq_lens_auto"
	temp_dir_manual = TEMP_DIR / "test_seq_lens_manual"
	model = MockHookedTransformerBatched()

	prompts_auto = _make_prompts()
	prompts_manual = _make_prompts()

	# Auto-computed seq_lens
	paths_auto = compute_activations_batched(
		prompts=prompts_auto,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir_auto,
	)

	# Manually pre-computed seq_lens
	manual_seq_lens = [model.to_tokens(p["text"]).shape[1] for p in prompts_manual]
	paths_manual = compute_activations_batched(
		prompts=prompts_manual,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir_manual,
		seq_lens=manual_seq_lens,
	)

	for auto_path, manual_path in zip(paths_auto, paths_manual, strict=True):
		with np.load(auto_path) as auto_data, np.load(manual_path) as manual_data:
			for key in auto_data:
				np.testing.assert_array_equal(
					auto_data[key],
					manual_data[key],
					err_msg=f"Key {key}: auto vs manual seq_lens differ",
				)


def test_batched_seq_lens_length_mismatch_raises():
	"""Passing seq_lens with wrong length raises AssertionError."""
	model = MockHookedTransformerBatched()
	prompts = _make_prompts()

	with pytest.raises(AssertionError, match="seq_lens length mismatch"):
		compute_activations_batched(
			prompts=prompts,
			model=model,  # type: ignore[arg-type]
			save_path=TEMP_DIR / "test_seq_lens_mismatch",
			seq_lens=[5, 10],  # 2 lengths for 3 prompts
		)


# ============================================================================
# Test: activations_main cache-skip path (force=False)
# ============================================================================

# Patches needed to run activations_main with a mock model
_ACTIVATIONS_MAIN_PATCHES = [
	"pattern_lens.activations.HookedTransformer",
	"pattern_lens.activations.load_text_data",
	"pattern_lens.activations.write_html_index",
	"pattern_lens.activations.generate_models_jsonl",
	"pattern_lens.activations.generate_prompts_jsonl",
	"pattern_lens.activations.asdict",
	"pattern_lens.activations.json_serialize",
]


def _make_5_prompts() -> list[dict]:
	"""5 prompts of varying lengths for cache-skip tests."""
	return [
		{"text": "aa"},
		{"text": "bbbb"},
		{"text": "cccccc"},
		{"text": "dddddddd"},
		{"text": "eeeeeeeeee"},
	]


def _run_activations_main_mocked(
	save_path: Path,
	prompts: list[dict],
	force: bool,
	batch_size: int = 32,
) -> mock.MagicMock:
	"""Run activations_main with all heavy dependencies mocked.

	Returns the mock for compute_activations_batched so callers can inspect calls.
	"""
	mock_model = MockHookedTransformerBatched()

	with (
		mock.patch("pattern_lens.activations.load_model", return_value=mock_model),
		mock.patch("pattern_lens.activations.load_text_data", return_value=prompts),
		mock.patch("pattern_lens.activations.write_html_index"),
		mock.patch("pattern_lens.activations.generate_models_jsonl"),
		mock.patch("pattern_lens.activations.generate_prompts_jsonl"),
		mock.patch("pattern_lens.activations.asdict", return_value={}),
		mock.patch("pattern_lens.activations.json_serialize", return_value={}),
		mock.patch(
			"pattern_lens.activations.compute_activations_batched",
			wraps=compute_activations_batched,
		) as spy_batched,
	):
		activations_main(
			model_name="test-model",
			save_path=str(save_path),
			prompts_path="dummy.jsonl",
			raw_prompts=True,
			min_chars=0,
			max_chars=9999,
			force=force,
			n_samples=len(prompts),
			no_index_html=True,
			device=torch.device("cpu"),
			batch_size=batch_size,
		)

	return spy_batched


def test_activations_main_partial_cache_skip():
	"""With force=False, only uncached prompts are computed."""
	temp_dir = TEMP_DIR / "test_main_partial_cache"
	if temp_dir.exists():
		shutil.rmtree(temp_dir)
	model = MockHookedTransformerBatched()
	all_prompts = _make_5_prompts()

	# Pre-compute activations for the first 2 prompts
	pre_cached = [dict(p) for p in all_prompts[:2]]
	for p in pre_cached:
		augment_prompt_with_hash(p)
	compute_activations_batched(
		prompts=pre_cached,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	# Verify they're cached
	for p in pre_cached:
		assert activations_exist("test-model", p, temp_dir)

	# Run activations_main with force=False — should skip the 2 cached ones
	spy = _run_activations_main_mocked(
		save_path=temp_dir,
		prompts=[dict(p) for p in all_prompts],
		force=False,
		batch_size=32,
	)

	# compute_activations_batched should have been called with only 3 uncached prompts
	assert spy.call_count == 1
	called_prompts = spy.call_args[1]["prompts"]
	assert len(called_prompts) == 3

	# All 5 should now be cached
	for p in all_prompts:
		augment_prompt_with_hash(p)
		assert activations_exist("test-model", p, temp_dir), (
			f"prompt {p['text']!r} not cached after activations_main"
		)


def test_activations_main_full_cache_skip():
	"""With force=False and all prompts cached, compute_activations_batched is never called."""
	temp_dir = TEMP_DIR / "test_main_full_cache"
	if temp_dir.exists():
		shutil.rmtree(temp_dir)
	model = MockHookedTransformerBatched()
	all_prompts = _make_5_prompts()

	# Pre-compute activations for ALL prompts
	pre_cached = [dict(p) for p in all_prompts]
	for p in pre_cached:
		augment_prompt_with_hash(p)
	compute_activations_batched(
		prompts=pre_cached,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	# Run activations_main with force=False — should skip everything
	spy = _run_activations_main_mocked(
		save_path=temp_dir,
		prompts=[dict(p) for p in all_prompts],
		force=False,
	)

	# compute_activations_batched should never have been called
	assert spy.call_count == 0


# ============================================================================
# Test: names_filter as a callable (not regex)
# ============================================================================


def test_names_filter_callable():
	"""Passing a plain callable as names_filter exercises the non-regex branch."""
	temp_dir = TEMP_DIR / "test_names_filter_callable"
	model = MockHookedTransformerBatched()
	prompts = _make_prompts()

	def my_filter(key: str) -> bool:
		return "hook_pattern" in key

	paths = compute_activations_batched(
		prompts=prompts,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
		names_filter=my_filter,
	)

	assert len(paths) == 3
	for prompt, path in zip(prompts, paths, strict=True):
		assert path.exists()
		expected_seq_len: int = _expected_seq_len(prompt["text"])
		with np.load(path) as data:
			for layer in range(model.cfg.n_layers):
				key = f"blocks.{layer}.attn.hook_pattern"
				assert key in data
				assert data[key].shape == (
					1,
					model.cfg.n_heads,
					expected_seq_len,
					expected_seq_len,
				)


# ============================================================================
# Test: activations_exist raises when hash is missing
# ============================================================================


def test_activations_exist_requires_hash():
	"""activations_exist raises InvalidPromptError when prompt has no hash."""
	with pytest.raises(InvalidPromptError, match="must have 'hash' key"):
		activations_exist("test-model", {"text": "no hash here"}, TEMP_DIR)


# ============================================================================
# Test: compute_activations_batched input validation
# ============================================================================


def test_compute_activations_batched_missing_text_raises():
	"""Prompt without 'text' key raises AssertionError."""
	model = MockHookedTransformerBatched()
	with pytest.raises(AssertionError, match="text"):
		compute_activations_batched(
			prompts=[{"hash": "abc"}],
			model=model,  # type: ignore[arg-type]
			save_path=TEMP_DIR / "test_missing_text",
		)


def test_compute_activations_batched_missing_hash_raises():
	"""Prompt without 'hash' key raises AssertionError."""
	model = MockHookedTransformerBatched()
	with pytest.raises(AssertionError, match="hash"):
		compute_activations_batched(
			prompts=[{"text": "hello"}],
			model=model,  # type: ignore[arg-type]
			save_path=TEMP_DIR / "test_missing_hash",
		)


# ============================================================================
# Test: activations_main with batch_size=1
# ============================================================================


def test_activations_main_batch_size_1():
	"""batch_size=1 processes each prompt individually (one call per prompt)."""
	temp_dir = TEMP_DIR / "test_main_batch_size_1"
	if temp_dir.exists():
		shutil.rmtree(temp_dir)
	all_prompts = _make_5_prompts()

	spy = _run_activations_main_mocked(
		save_path=temp_dir,
		prompts=[dict(p) for p in all_prompts],
		force=True,
		batch_size=1,
	)

	# With batch_size=1 and 5 prompts, should be called 5 times (one prompt each)
	assert spy.call_count == 5
	for call in spy.call_args_list:
		assert len(call[1]["prompts"]) == 1

	# All 5 should now be cached
	for p in all_prompts:
		augment_prompt_with_hash(p)
		assert activations_exist("test-model", p, temp_dir)


# ============================================================================
# Test: activations_main with force=True recomputes cached prompts
# ============================================================================


def test_activations_main_force_recomputes():
	"""With force=True, all prompts are recomputed even if already cached."""
	temp_dir = TEMP_DIR / "test_main_force_recompute"
	if temp_dir.exists():
		shutil.rmtree(temp_dir)
	model = MockHookedTransformerBatched()
	all_prompts = _make_5_prompts()

	# Pre-compute activations for ALL prompts
	pre_cached = [dict(p) for p in all_prompts]
	for p in pre_cached:
		augment_prompt_with_hash(p)
	compute_activations_batched(
		prompts=pre_cached,
		model=model,  # type: ignore[arg-type]
		save_path=temp_dir,
	)

	# Verify they're all cached
	for p in pre_cached:
		assert activations_exist("test-model", p, temp_dir)

	# Run with force=True — should recompute all 5
	spy = _run_activations_main_mocked(
		save_path=temp_dir,
		prompts=[dict(p) for p in all_prompts],
		force=True,
	)

	# compute_activations_batched should have been called with all 5 prompts
	total_computed = sum(len(call[1]["prompts"]) for call in spy.call_args_list)
	assert total_computed == 5


# ============================================================================
# Test: activations_main sorts prompts by length (longest first)
# ============================================================================


def test_activations_main_sorts_by_length():
	"""Prompts are sorted longest-first within each batch for padding efficiency."""
	temp_dir = TEMP_DIR / "test_main_sorts_by_length"
	if temp_dir.exists():
		shutil.rmtree(temp_dir)

	# Deliberately pass prompts in SHORT-first order
	prompts = [
		{"text": "a"},  # shortest
		{"text": "bb"},
		{"text": "ccc"},
		{"text": "dddd"},
		{"text": "eeeee"},  # longest
	]

	spy = _run_activations_main_mocked(
		save_path=temp_dir,
		prompts=[dict(p) for p in prompts],
		force=True,
		batch_size=100,  # large enough to fit all in one batch
	)

	# All prompts in one call — check they're sorted longest-first
	assert spy.call_count == 1
	called_prompts = spy.call_args[1]["prompts"]
	called_texts = [p["text"] for p in called_prompts]
	called_lengths = [len(t) for t in called_texts]
	assert called_lengths == sorted(called_lengths, reverse=True), (
		f"Prompts not sorted longest-first: {called_texts}"
	)

	# seq_lens must be sorted in the same order as prompts
	called_seq_lens = spy.call_args[1]["seq_lens"]
	assert called_seq_lens == sorted(called_seq_lens, reverse=True), (
		f"seq_lens not sorted longest-first: {called_seq_lens}"
	)


# ============================================================================
# Test: activations_main splits into multiple batches correctly
# ============================================================================


def test_activations_main_multiple_batches():
	"""batch_size=2 with 5 prompts produces 3 batched calls (2+2+1)."""
	temp_dir = TEMP_DIR / "test_main_multi_batch"
	if temp_dir.exists():
		shutil.rmtree(temp_dir)
	all_prompts = _make_5_prompts()

	spy = _run_activations_main_mocked(
		save_path=temp_dir,
		prompts=[dict(p) for p in all_prompts],
		force=True,
		batch_size=2,
	)

	# 5 prompts / batch_size=2 => 3 calls: 2, 2, 1
	assert spy.call_count == 3
	batch_sizes = [len(call[1]["prompts"]) for call in spy.call_args_list]
	assert batch_sizes == [2, 2, 1], f"Unexpected batch sizes: {batch_sizes}"

	# All 5 should be cached
	for p in all_prompts:
		augment_prompt_with_hash(p)
		assert activations_exist("test-model", p, temp_dir)


# -- _save_activations tests --------------------------------------------------


@pytest.fixture
def sample_cache_np() -> dict[str, np.ndarray]:
	"""Small cache_np for testing save/load round-trips."""
	rng = np.random.default_rng(42)
	return {
		"blocks.0.attn.hook_pattern": rng.standard_normal((1, 4, 8, 8)).astype(
			np.float32
		),
		"blocks.1.attn.hook_pattern": rng.standard_normal((1, 4, 8, 8)).astype(
			np.float32
		),
	}


@pytest.mark.parametrize("compress_level", [0, 1, 6])
def test_save_activations_roundtrip(tmp_path, sample_cache_np, compress_level):
	"""Files saved by _save_activations are loadable by np.load with correct contents."""
	path = tmp_path / "activations.npz"
	_save_activations(path, sample_cache_np, compress_level=compress_level)

	loaded = dict(np.load(path))
	assert set(loaded.keys()) == set(sample_cache_np.keys())
	for key in sample_cache_np:
		np.testing.assert_array_equal(loaded[key], sample_cache_np[key])


def test_save_activations_level0_larger_than_level6(tmp_path, sample_cache_np):
	"""Level 0 (no compression) produces larger files than level 6."""
	path_0 = tmp_path / "level0.npz"
	path_6 = tmp_path / "level6.npz"
	_save_activations(path_0, sample_cache_np, compress_level=0)
	_save_activations(path_6, sample_cache_np, compress_level=6)
	assert path_0.stat().st_size > path_6.stat().st_size
