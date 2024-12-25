import pytest
import numpy as np
from pathlib import Path
import gzip
import base64

import matplotlib
import matplotlib.pyplot as plt

from pattern_lens.consts import AttentionMatrix

from pattern_lens.figure_util import MATPLOTLIB_FIGURE_FMT, SVG_TEMPLATE, save_matrix_as_svgz_wrapper


TEMP_DIR: Path = Path("tests/_temp")

def matplotlib_figure_saver(func, fmt: str = MATPLOTLIB_FIGURE_FMT):
    """Decorator for saving matplotlib figures."""
    def wrapped(attn_matrix, save_dir: Path):
        fig_path: Path = save_dir / f"{func.__name__}.{fmt}"
        fig, ax = plt.subplots(figsize=(10, 10))
        func(attn_matrix, ax)
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)
    return wrapped

def matrix_as_svg(matrix: np.ndarray, normalize: bool = False, cmap="viridis") -> str:
    """Convert a 2D matrix to an SVG image."""
    assert matrix.ndim == 2, f"Matrix must be 2D, got {matrix.ndim = }"
    if normalize:
        max_val, min_val = matrix.max(), matrix.min()
        matrix = (matrix - min_val) / (max_val - min_val)
    else:
        assert 0 <= matrix.min() <= matrix.max() <= 1, "Matrix values must be in range [0, 1], or normalize must be True."
    cmap = matplotlib.colormaps[cmap]
    rgba_matrix = (cmap(matrix)[:, :, :3] * 255).astype(np.uint8)
    n, m, channels = rgba_matrix.shape
    assert channels == 3, f"Expected 3 channels, got {channels = }"
    image_data = f"P6 {m} {n} 255\n".encode() + rgba_matrix.tobytes()
    png_base64 = base64.b64encode(image_data).decode("utf-8")
    return SVG_TEMPLATE.format(m=m, n=n, png_base64=png_base64)

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
    def faulty_plot(attn_matrix, ax):
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
    with pytest.raises(AssertionError, match="Matrix values must be in range \\[0, 1\\], or normalize must be True"):
        matrix_as_svg(matrix, normalize=False)

def test_matrix_as_svg_invalid_dims():
    matrix = np.random.rand(5, 5, 5).astype(np.float32)
    with pytest.raises(AssertionError, match="Matrix must be 2D"):
        matrix_as_svg(matrix, normalize=True)

def test_matrix_as_svg_invalid_cmap_fixed():
    matrix = np.array([[0.1, 0.4], [0.6, 0.9]], dtype=np.float32)
    with pytest.raises(KeyError, match="'invalid_cmap' is not a known colormap name"):
        matrix_as_svg(matrix, cmap="invalid_cmap")

# Test with no arguments
def test_save_matrix_as_svgz_wrapper_no_args():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @save_matrix_as_svgz_wrapper
    def no_op(matrix):
        return matrix

    test_matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    no_op(test_matrix, TEMP_DIR)

    saved_file = TEMP_DIR / "no_op.svgz"
    assert saved_file.exists(), "SVGZ file was not saved in no-args case"

# Test with keyword-only arguments
def test_save_matrix_as_svgz_wrapper_with_args():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @save_matrix_as_svgz_wrapper(normalize=True, cmap="plasma")
    def scale_matrix(matrix):
        return matrix * 2

    test_matrix = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    scale_matrix(test_matrix, TEMP_DIR)

    saved_file = TEMP_DIR / "scale_matrix.svgz"
    assert saved_file.exists(), "SVGZ file was not saved with keyword-only arguments"

# Test exception handling
def test_save_matrix_as_svgz_wrapper_exceptions():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @save_matrix_as_svgz_wrapper(normalize=False)
    def invalid_range(matrix):
        return matrix * 2

    test_matrix = np.array([[2, 3], [4, 5]], dtype=np.float32)
    with pytest.raises(
        AssertionError,
        match=r"Matrix values must be in range \[0, 1\], or normalize must be True\. got: min: .*?, max: .*?",
    ):
        invalid_range(test_matrix, TEMP_DIR)

# Test keyword-only arguments enforced
def test_save_matrix_as_svgz_wrapper_keyword_only():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @save_matrix_as_svgz_wrapper(normalize=True, cmap="plasma")
    def scale_matrix(matrix):
        return matrix * 2

    test_matrix = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    scale_matrix(test_matrix, TEMP_DIR)

    saved_file = TEMP_DIR / "scale_matrix.svgz"
    assert saved_file.exists(), "SVGZ file was not saved with keyword-only arguments"

# Test multiple calls to the decorator
def test_save_matrix_as_svgz_wrapper_multiple():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @save_matrix_as_svgz_wrapper(normalize=True)
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

    @save_matrix_as_svgz_wrapper(normalize=False)
    def pass_through(matrix):
        return matrix

    test_matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    pass_through(test_matrix, TEMP_DIR)

    saved_file = TEMP_DIR / "pass_through.svgz"
    assert saved_file.exists(), "SVGZ file was not saved when normalization was not applied"

# Test with a complex matrix
def test_save_matrix_as_svgz_wrapper_complex_matrix():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @save_matrix_as_svgz_wrapper(normalize=True, cmap="viridis")
    def complex_processing(matrix):
        return np.sin(matrix)

    test_matrix = np.linspace(0, np.pi, 16).reshape(4, 4).astype(np.float32)
    complex_processing(test_matrix, TEMP_DIR)

    saved_file = TEMP_DIR / "complex_processing.svgz"
    assert saved_file.exists(), "SVGZ file was not saved for complex matrix processing"
