from pathlib import Path
from typing import Any, Callable, Protocol
import functools
import base64
import gzip
from io import BytesIO

import numpy as np
from jaxtyping import Float, Int, Bool
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from pattern_lens.consts import AttentionMatrix

AttentionMatrixFigureFunc = Callable[[AttentionMatrix, Path], None]
"Type alias for a function that, given an attention matrix, saves a figure"

MATPLOTLIB_FIGURE_FMT: str = "svgz"
"format for saving matplotlib figures"

SVG_TEMPLATE: str = """<svg xmlns="http://www.w3.org/2000/svg" width="{m}" height="{n}" viewBox="0 0 {m} {n}" shape-rendering="crispEdges"> <image href="data:image/png;base64,{png_base64}" width="{m}" height="{n}" /> </svg>"""
"template for saving an `n` by `m` matrix as an svg/svgz"



def matplotlib_figure_saver(
	func: Callable[[AttentionMatrix, plt.Axes], None],
	fmt: str = MATPLOTLIB_FIGURE_FMT,
) -> AttentionMatrixFigureFunc:
	"""decorator for functions which take an attention matrix and predefined `ax` object, making it save a figure
	
	# Parameters:
	 - `func : Callable[[AttentionMatrix, plt.Axes], None]`   
	   your function, which should take an attention matrix and predefined `ax` object
	 - `fmt : str`   
	   format for saving matplotlib figures
	   (defaults to `MATPLOTLIB_FIGURE_FMT`)
	
	# Returns:
	 - `AttentionMatrixFigureFunc` 
	   your function, after we wrap it to save a figure

	# Usage:
	```python	
	@register_attn_figure_func
	@matplotlib_figure_saver
	def raw(attn_matrix: AttentionMatrix, ax: plt.Axes) -> None:
		ax.matshow(attn_matrix, cmap="viridis")
		ax.set_title("Raw Attention Pattern")
		ax.axis("off")
	```

	"""	

	@functools.wraps(func)
	def wrapped(attn_matrix: AttentionMatrix, save_dir: Path) -> None:
		fig_path: Path = save_dir / f"{func.__name__}.{fmt}"

		fig, ax = plt.subplots(figsize=(10, 10))
		func(attn_matrix, ax)
		plt.tight_layout()
		plt.savefig(fig_path)
		plt.close(fig)

	return wrapped




def matrix_as_svg(
	matrix: Float[np.ndarray, "n m"],
	normalize: bool = False,
	cmap = "viridis",
) -> str:
	"""quickly convert a 2D matrix to an SVG image, without matplotlib
		
	# Parameters:
	 - `matrix : Float[np.ndarray, 'n m']`   
	   a 2D matrix to convert to an SVG image
	 - `normalize : bool`   
	   whether to normalize the matrix to range [0, 1]. if it's not in the range [0, 1], this must be `True` or it will raise an `AssertionError`
	   (defaults to `False`)
	 - `cmap : str`   
	   the colormap to use for the matrix -- will look up in `matplotlib.colormaps` if it's a string
	   (defaults to `"viridis"`)
	
	# Returns:
	 - `str` 
	   the SVG content for the matrix
	"""	
	
	# check dims
	assert matrix.ndim == 2, f"Matrix must be 2D, got {matrix.ndim = }"

	# Normalize the matrix to range [0, 1]
	normalized_matrix: Float[np.ndarray, "n m"]
	if normalize:
		max_val, min_val = matrix.max(), matrix.min()
		normalized_matrix = (matrix - min_val) / (max_val - min_val)
	else:
		assert matrix.min() >= 0 and matrix.max() <= 1, "Matrix values must be in range [0, 1], or normalize must be True. got: min: {matrix.min() = }, max: {matrix.max() = }"
		normalized_matrix = matrix

	# get the colormap
	if isinstance(cmap, str):
		cmap = matplotlib.colormaps[cmap]

	# Apply the viridis colormap
	rgba_matrix: Float[np.ndarray, "n m 3"] = (
		(cmap(normalized_matrix)[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel
	)

	# Encode the matrix as PNG-like base64
	n: int; m: int; channels: int
	n, m, channels = rgba_matrix.shape
	assert channels == 3, f"Matrix after colormap must have 3 channels, got {channels = }"
	image_data: bytes = f"P6 {m} {n} 255\n".encode() + rgba_matrix.tobytes()  # PPM binary header
	png_base64: str = base64.b64encode(image_data).decode('utf-8')

	# Generate the SVG content
	svg_content: str = SVG_TEMPLATE.format(m=m, n=n, png_base64=png_base64)

	return svg_content


def save_matrix_as_svgz_wrapper(
	func: Callable[[Float[np.ndarray, "n m"]], Float[np.ndarray, "n m"]],
	normalize: Bool = False,
	cmap = "viridis",
) -> AttentionMatrixFigureFunc:
	"""decorator for functions which take an attention matrix and return a new matrix, making it save an svgz figure using `matrix_as_svg`
	
	# Parameters:
	 - `func : Callable[[Float[np.ndarray, 'n m']], Float[np.ndarray, 'n m']]`   
	   your function, which should take an attention matrix and return a new matrix
	 - `normalize : Bool`   
	   whether to normalize the matrix to range [0, 1]. if it's not in the range [0, 1], this must be `True` or it will raise an `AssertionError`. passed to `matrix_as_svg`
	   (defaults to `False`)
	 - `cmap : str`   
	   the colormap to use for the matrix -- will look up in `matplotlib.colormaps` if it's a string. passed to `matrix_as_svg`. passed to `matrix_as_svg`
	   (defaults to `"viridis"`)
	
	# Returns:
	 - `AttentionMatrixFigureFunc` 
	   your function, after we wrap it to save an svgz figure
	"""	
	
	@functools.wraps(func)
	def wrapped(attn_matrix: AttentionMatrix, save_dir: Path) -> None:
		fig_path: Path = save_dir / f"{func.__name__}.svgz"

		# Apply the function
		new_matrix: Float[np.ndarray, "n m"] = func(attn_matrix)

		# Save the matrix as SVGZ
		svg_content: str = matrix_as_svg(new_matrix, normalize=normalize, cmap=cmap)
		with gzip.open(fig_path, "wt") as f:
			f.write(svg_content)

	return wrapped