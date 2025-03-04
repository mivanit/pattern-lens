import base64
import gzip
import io
import re
from pathlib import Path

import jaxtyping
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from pattern_lens.figure_util import (
	MATPLOTLIB_FIGURE_FMT,
	matplotlib_figure_saver,
	matrix_as_svg,
	save_matrix_wrapper,
)

TEMP_DIR: Path = Path("tests/_temp")


def test_matplotlib_figure_saver():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@matplotlib_figure_saver
	def plot_matrix(attn_matrix, ax):
		ax.matshow(attn_matrix, cmap="viridis")
		ax.axis("off")

	attn_matrix = np.random.rand(10, 10).astype(np.float32)
	plot_matrix(attn_matrix, TEMP_DIR)

	saved_file = TEMP_DIR / f"plot_matrix.{MATPLOTLIB_FIGURE_FMT}"
	assert saved_file.exists(), "Matplotlib figure file was not saved"


def test_matplotlib_figure_saver_exception():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@matplotlib_figure_saver
	def faulty_plot(attn_matrix, ax):  # noqa: ARG001
		raise ValueError("Intentional failure for testing")

	attn_matrix = np.random.rand(10, 10).astype(np.float32)
	with pytest.raises(ValueError, match="Intentional failure for testing"):
		faulty_plot(attn_matrix, TEMP_DIR)


def test_matrix_as_svg_normalization():
	matrix = np.array([[2, 4], [6, 8]], dtype=np.float32)
	svg_content = matrix_as_svg(matrix, normalize=True)
	assert "image href=" in svg_content, "SVG content is malformed"
	assert "data:image/png;base64," in svg_content, "Base64 encoding is missing"


def test_matrix_as_svg_no_normalization():
	matrix = np.array([[0.1, 0.4], [0.6, 0.9]], dtype=np.float32)
	svg_content = matrix_as_svg(matrix, normalize=False)
	assert "image href=" in svg_content, "SVG content is malformed"
	assert "data:image/png;base64," in svg_content, "Base64 encoding is missing"


def test_matrix_as_svg_invalid_range():
	matrix = np.array([[-1, 2], [3, 4]], dtype=np.float32)
	with pytest.raises(
		AssertionError,
		match="Matrix values must be in range \\[0, 1\\], or normalize must be True",
	):
		matrix_as_svg(matrix, normalize=False)


def test_matrix_as_svg_invalid_dims():
	matrix = np.random.rand(5, 5, 5).astype(np.float32)
	with pytest.raises((AssertionError, jaxtyping.TypeCheckError)):
		matrix_as_svg(matrix, normalize=True)


def test_matrix_as_svg_invalid_cmap_fixed():
	matrix = np.array([[0.1, 0.4], [0.6, 0.9]], dtype=np.float32)
	with pytest.raises(KeyError, match="'invalid_cmap' is not a known colormap name"):
		matrix_as_svg(matrix, cmap="invalid_cmap")


# Test with no arguments
def test_save_matrix_as_svgz_wrapper_no_args():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@save_matrix_wrapper(fmt="svgz")
	def no_op(matrix):
		return matrix

	test_matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
	no_op(test_matrix, TEMP_DIR)

	saved_file = TEMP_DIR / "no_op.svgz"
	assert saved_file.exists(), "SVGZ file was not saved in no-args case"


# Test with keyword-only arguments
def test_save_matrix_as_svgz_wrapper_with_args():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@save_matrix_wrapper(normalize=True, cmap="plasma")
	def scale_matrix(matrix):
		return matrix * 2

	test_matrix = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
	scale_matrix(test_matrix, TEMP_DIR)

	saved_file = TEMP_DIR / "scale_matrix.svgz"
	assert saved_file.exists(), "SVGZ file was not saved with keyword-only arguments"


# Test exception handling
def test_save_matrix_as_svgz_wrapper_exceptions():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@save_matrix_wrapper(normalize=False)
	def invalid_range(matrix):
		return matrix * 2

	test_matrix = np.array([[2, 3], [4, 5]], dtype=np.float32)
	with pytest.raises(
		AssertionError,
		match=r"Matrix values must be in range \[0, 1\], or normalize must be True.*",
	):
		invalid_range(test_matrix, TEMP_DIR)


# Test keyword-only arguments enforced
def test_save_matrix_as_svgz_wrapper_keyword_only():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@save_matrix_wrapper(normalize=True, cmap="plasma")
	def scale_matrix(matrix):
		return matrix * 2

	test_matrix = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
	scale_matrix(test_matrix, TEMP_DIR)

	saved_file = TEMP_DIR / "scale_matrix.svgz"
	assert saved_file.exists(), "SVGZ file was not saved with keyword-only arguments"


# Test multiple calls to the decorator
def test_save_matrix_as_svgz_wrapper_multiple():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@save_matrix_wrapper(normalize=True)
	def scale_by_factor(matrix):
		return matrix * 3

	matrix_1 = np.array([[0.1, 0.5], [0.7, 0.9]], dtype=np.float32)
	matrix_2 = np.array([[0.2, 0.6], [0.8, 1.0]], dtype=np.float32)

	scale_by_factor(matrix_1, TEMP_DIR)
	scale_by_factor(matrix_2, TEMP_DIR)

	# Check the saved files
	saved_file = TEMP_DIR / "scale_by_factor.svgz"
	assert saved_file.exists(), "SVGZ file was not saved for multiple calls"


# Validate behavior when normalize is False and values are in range
def test_save_matrix_as_svgz_wrapper_no_normalization():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@save_matrix_wrapper(normalize=False)
	def pass_through(matrix):
		return matrix

	test_matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
	pass_through(test_matrix, TEMP_DIR)

	saved_file = TEMP_DIR / "pass_through.svgz"
	assert saved_file.exists(), (
		"SVGZ file was not saved when normalization was not applied"
	)


# Test with a complex matrix
def test_save_matrix_as_svgz_wrapper_complex_matrix():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@save_matrix_wrapper(normalize=True, cmap="viridis")
	def complex_processing(matrix):
		return np.sin(matrix)

	test_matrix = np.linspace(0, np.pi, 16).reshape(4, 4).astype(np.float32)
	complex_processing(test_matrix, TEMP_DIR)

	saved_file = TEMP_DIR / "complex_processing.svgz"
	assert saved_file.exists(), "SVGZ file was not saved for complex matrix processing"


def test_matrix_as_svg_dimensions():
	# Test different matrix shapes
	matrices = [
		np.random.rand(5, 10),  # Non-square
		np.random.rand(3, 3),  # Small square
		np.random.rand(100, 50),  # Large non-square
	]

	for matrix in matrices:
		m, n = matrix.shape
		svg_content = matrix_as_svg(matrix, normalize=True)
		assert f'width="{m}"' in svg_content
		assert f'height="{n}"' in svg_content
		assert f'viewBox="0 0 {m} {n}"' in svg_content


def test_save_matrix_as_svgz_wrapper_content():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	@save_matrix_wrapper(normalize=True)
	def identity(matrix):
		return matrix

	test_matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
	identity(test_matrix, TEMP_DIR)

	saved_file = TEMP_DIR / "identity.svgz"
	with gzip.open(saved_file, "rt") as f:
		content = f.read()
		assert "svg" in content
		assert "image href=" in content
		assert "base64" in content


@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_matplotlib_figure_saver_formats(fmt):
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	# TYPING: error: Too few arguments  [call-arg]
	@matplotlib_figure_saver(None, fmt=fmt)  # type: ignore[call-arg]
	def plot_matrix(attn_matrix, ax):
		ax.matshow(attn_matrix)
		ax.axis("off")

	matrix = np.random.rand(5, 5)
	plot_matrix(matrix, TEMP_DIR)
	saved_file = TEMP_DIR / f"plot_matrix.{fmt}"
	assert saved_file.exists(), f"File not saved for format {fmt}"


def test_matrix_as_svg_empty():
	empty_matrix = np.array([[]], dtype=np.float32).reshape(0, 0)
	with pytest.raises(AssertionError, match="Matrix cannot be empty"):
		matrix_as_svg(empty_matrix)


def test_matplotlib_figure_saver_cleanup():
	TEMP_DIR.mkdir(parents=True, exist_ok=True)
	initial_figures = len(plt.get_fignums())

	@matplotlib_figure_saver
	def plot_matrix(attn_matrix, ax):
		ax.matshow(attn_matrix)

	matrix = np.random.rand(5, 5)
	plot_matrix(matrix, TEMP_DIR)

	# Check that no figure objects remain
	assert len(plt.get_fignums()) == initial_figures, "Figure not properly cleaned up"


def test_matrix_as_svg_non_numeric():
	matrix = np.array([["a", "b"], ["c", "d"]])
	with pytest.raises(TypeError):
		matrix_as_svg(matrix)


def test_matrix_as_svg_format():
	# create a small 2x2 matrix
	matrix = np.array([[0.0, 0.5], [1.0, 0.75]], dtype=float)

	svg_str = matrix_as_svg(matrix)

	# ensure it's got the correct SVG wrapper
	assert svg_str.startswith("<svg"), "SVG should start with <svg>"
	assert svg_str.endswith("</svg>"), "SVG should end with </svg>"

	# find the embedded base64 image data
	match = re.search(r'data:image/png;base64,([^"]+)', svg_str)
	assert match, "Expected an embedded PNG in data URI format"

	embedded_data = match.group(1)
	png_data = base64.b64decode(embedded_data)

	Image.open(io.BytesIO(png_data))
