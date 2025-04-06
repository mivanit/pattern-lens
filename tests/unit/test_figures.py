# tests/unit/test_figures.py
from pathlib import Path
from unittest import mock

import numpy as np

from pattern_lens.attn_figure_funcs import ATTENTION_MATRIX_FIGURE_FUNCS
from pattern_lens.figure_util import save_matrix_wrapper
from pattern_lens.figures import compute_and_save_figures, process_single_head

TEMP_DIR: Path = Path("tests/_temp")


def test_process_single_head():
	"""Test processing a single head's attention pattern."""
	# Setup
	temp_dir = TEMP_DIR / "test_process_single_head"
	head_dir = temp_dir / "L0" / "H0"
	head_dir.mkdir(parents=True)

	# Create a simple test attention matrix
	attn_pattern = np.random.rand(10, 10).astype(np.float32)

	# Create a simple test figure function
	@save_matrix_wrapper(fmt="svg")
	def test_fig_func(matrix):
		return matrix

	# Patch to use our test function
	result = process_single_head(
		layer_idx=0,
		head_idx=0,
		attn_pattern=attn_pattern,
		save_dir=head_dir,
		figure_funcs=[*ATTENTION_MATRIX_FIGURE_FUNCS, test_fig_func],
		force_overwrite=True,
	)

	# Check that our function was called and succeeded
	assert "test_fig_func" in result
	assert result["test_fig_func"] is True

	# Check that the figure file was created
	assert (head_dir / "test_fig_func.svg").exists()


def test_process_single_head_error_handling():
	"""Test that process_single_head properly handles errors in figure functions."""
	# Setup
	temp_dir = TEMP_DIR / "test_process_single_head_error_handling"
	head_dir = temp_dir / "L0" / "H0"
	head_dir.mkdir(parents=True)

	# Create a simple test attention matrix
	attn_pattern = np.random.rand(10, 10).astype(np.float32)

	# Create a test figure function that raises an exception
	def error_fig_func(matrix, save_dir):  # noqa: ARG001
		raise ValueError("Test error")

	# Patch to use our test function
	result = process_single_head(
		layer_idx=0,
		head_idx=0,
		attn_pattern=attn_pattern,
		save_dir=head_dir,
		figure_funcs=[error_fig_func],
		force_overwrite=True,
	)

	# Check that our function was called and failed
	assert "error_fig_func" in result
	assert isinstance(result["error_fig_func"], ValueError)

	# Check that an error file was created
	assert (head_dir / "error_fig_func.error.txt").exists()


def test_compute_and_save_figures():
	"""Test compute_and_save_figures with a mock model config and cache."""
	# Setup
	temp_dir = TEMP_DIR / "test_compute_and_save_figures"
	prompt_dir = temp_dir / "test-model" / "prompts" / "test-hash"
	prompt_dir.mkdir(parents=True)

	# Create a mock model config
	class MockConfig:
		def __init__(self):
			self.n_layers = 2
			self.n_heads = 2
			self.model_name = "test-model"

	# Create a simple test attention matrices dict
	cache_dict = {
		"blocks.0.attn.hook_pattern": np.random.rand(1, 2, 5, 5).astype(np.float32),
		"blocks.1.attn.hook_pattern": np.random.rand(1, 2, 5, 5).astype(np.float32),
	}

	# Create a simple test figure function
	@save_matrix_wrapper(fmt="png")
	def test_fig_func(matrix):
		return matrix

	# Patch ATTENTION_MATRIX_FIGURE_FUNCS and process_single_head
	with (
		mock.patch(
			"pattern_lens.figures.ATTENTION_MATRIX_FIGURE_FUNCS",
			[test_fig_func],
		),
		mock.patch("pattern_lens.figures.process_single_head") as mock_process_head,
		mock.patch("pattern_lens.figures.generate_prompts_jsonl") as mock_gen_prompts,
	):
		mock_process_head.return_value = {"test_fig_func": True}

		# Call compute_and_save_figures with dict cache
		compute_and_save_figures(
			model_cfg=MockConfig(),
			activations_path=prompt_dir / "activations.npz",
			cache=cache_dict,
			figure_funcs=ATTENTION_MATRIX_FIGURE_FUNCS,
			save_path=temp_dir,
			force_overwrite=True,
		)

		# Check that process_single_head was called for each layer and head
		assert mock_process_head.call_count == 4  # 2 layers * 2 heads

		# Check that generate_prompts_jsonl was called
		mock_gen_prompts.assert_called_once_with(temp_dir / "test-model")

		# Reset mocks and test with stacked array
		mock_process_head.reset_mock()
		mock_gen_prompts.reset_mock()

		# Create a stacked array of shape [n_layers, n_heads, n_ctx, n_ctx]
		cache_array = np.random.rand(2, 2, 5, 5).astype(np.float32)

		# Call compute_and_save_figures with array cache
		compute_and_save_figures(
			model_cfg=MockConfig(),
			activations_path=prompt_dir / "activations.npy",
			cache=cache_array,
			figure_funcs=ATTENTION_MATRIX_FIGURE_FUNCS,
			save_path=temp_dir,
			force_overwrite=True,
		)

		# Check that process_single_head was called for each layer and head
		assert mock_process_head.call_count == 4  # 2 layers * 2 heads

		# Check that generate_prompts_jsonl was called
		mock_gen_prompts.assert_called_once_with(temp_dir / "test-model")
