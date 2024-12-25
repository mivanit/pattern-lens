from pathlib import Path
from typing import Any, Callable, Protocol, overload, Union
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

Matrix2D = Float[np.ndarray, "n m"]
"Type alias for a 2D matrix (plottable)"

AttentionMatrixToMatrixFunc = Callable[[AttentionMatrix], Matrix2D]
"Type alias for a function that, given an attention matrix, returns a 2D matrix"

MATPLOTLIB_FIGURE_FMT: str = "svgz"
"format for saving matplotlib figures"

SVG_TEMPLATE: str = """<svg xmlns="http://www.w3.org/2000/svg" width="{m}" height="{n}" viewBox="0 0 {m} {n}" shape-rendering="crispEdges"> <image href="data:image/png;base64,{png_base64}" width="{m}" height="{n}" /> </svg>"""
"template for saving an `n` by `m` matrix as an svg/svgz"


@overload # without keyword arguments, returns decorated function
def matplotlib_figure_saver(
	func: Callable[[AttentionMatrix, plt.Axes], None],
	*args,
	fmt: str = MATPLOTLIB_FIGURE_FMT,
) -> AttentionMatrixFigureFunc:
	...
@overload # with keyword arguments, returns decorator
def matplotlib_figure_saver(
	func: None = None,
	*args,
	fmt: str = MATPLOTLIB_FIGURE_FMT,
) -> Callable[[Callable[[AttentionMatrix, plt.Axes], None], str], AttentionMatrixFigureFunc]:
	...
def matplotlib_figure_saver(
	func: Callable[[AttentionMatrix, plt.Axes], None]|None = None,
	*args,
	fmt: str = MATPLOTLIB_FIGURE_FMT,
) -> Union[
	AttentionMatrixFigureFunc,
	Callable[[Callable[[AttentionMatrix, plt.Axes], None], str], AttentionMatrixFigureFunc]
]:
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

	assert len(args) == 0, "This decorator only supports keyword arguments"


	def decorator(
		func: Callable[[AttentionMatrix, plt.Axes], None],
		fmt: str = fmt,
	) -> AttentionMatrixFigureFunc:
		@functools.wraps(func)
		def wrapped(attn_matrix: AttentionMatrix, save_dir: Path) -> None:
			fig_path: Path = save_dir / f"{func.__name__}.{fmt}"

			fig, ax = plt.subplots(figsize=(10, 10))
			func(attn_matrix, ax)
			plt.tight_layout()
			plt.savefig(fig_path)
			plt.close(fig)

		return wrapped

	if callable(func):
		# Handle no-arguments case
		return decorator(func)
	else:
		# Handle arguments case
		return decorator




def matrix_as_svg(
	matrix: Matrix2D,
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

	# check matrix is not empty
	assert matrix.size > 0, "Matrix cannot be empty"

	# Normalize the matrix to range [0, 1]
	normalized_matrix: Matrix2D
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


@overload # with keyword arguments, returns decorator
def save_matrix_as_svgz_wrapper(
	func: None = None,
	*args,
	normalize: Bool = False,
	cmap: str = "viridis",
) -> Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]: 
	...
@overload # without keyword arguments, returns decorated function
def save_matrix_as_svgz_wrapper(
	func: AttentionMatrixToMatrixFunc,
	*args,
	normalize: Bool = False,
	cmap: str = "viridis",
) -> AttentionMatrixFigureFunc:
	...
def save_matrix_as_svgz_wrapper(
	func: AttentionMatrixToMatrixFunc|None = None,
	*args,
	normalize: bool = False,
	cmap="viridis",
) -> AttentionMatrixFigureFunc|Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]:
	"""
	Decorator for functions that process an attention matrix and save it as an SVGZ image.
	Can handle both argumentless usage and with arguments.

	# Parameters:
	 - `func : AttentionMatrixToMatrixFunc|None` 
		Either the function to decorate (in the no-arguments case) or `None` when used with arguments.
	 - `normalize : bool, keyword-only`  
		Whether to normalize the matrix to range [0, 1]. Defaults to `False`.
	 - `cmap : str, keyword-only`  
		The colormap to use for the matrix. Defaults to `"viridis"`.

	# Returns:
	
	`AttentionMatrixFigureFunc|Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]`
	
	- `AttentionMatrixFigureFunc` if `func` is `AttentionMatrixToMatrixFunc` (no arguments case)
	- `Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]` if `func` is `None` -- returns the decorator which will then be applied to the  (with arguments case)

	# Usage:
	```python
	@save_matrix_as_svgz_wrapper(normalize=True, cmap="plasma")
	def scale_matrix(matrix):
		return matrix * 2

	@save_matrix_as_svgz_wrapper
	def identity_matrix(matrix):
		return matrix
	```
	"""

	assert len(args) == 0, "This decorator only supports keyword arguments"

	def decorator(func: Callable[[AttentionMatrix], Matrix2D]) -> AttentionMatrixFigureFunc:
		@functools.wraps(func)
		def wrapped(attn_matrix: AttentionMatrix, save_dir: Path) -> None:
			fig_path: Path = save_dir / f"{func.__name__}.svgz"
			new_matrix: Matrix2D = func(attn_matrix)
			svg_content: str = matrix_as_svg(new_matrix, normalize=normalize, cmap=cmap)
			with gzip.open(fig_path, "wt") as f:
				f.write(svg_content)
		return wrapped

	if callable(func):
		# Handle no-arguments case
		return decorator(func)
	else:
		# Handle arguments case
		return decorator