"""implements a bunch of types, default values, and templates which are useful for figure functions

notably, you can use the decorators `matplotlib_figure_saver`, `save_matrix_wrapper` to make your functions save figures
"""

from pathlib import Path
from typing import Callable, Literal, Sequence, overload, Union
import functools
import base64
import gzip
import io

from PIL import Image
import numpy as np
from jaxtyping import Float, UInt8
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from pattern_lens.consts import AttentionMatrix

AttentionMatrixFigureFunc = Callable[[AttentionMatrix, Path], None]
"Type alias for a function that, given an attention matrix, saves one or more figures"

Matrix2D = Float[np.ndarray, "n m"]
"Type alias for a 2D matrix (plottable)"

Matrix2Drgb = UInt8[np.ndarray, "n m rgb=3"]
"Type alias for a 2D matrix with 3 channels (RGB)"

AttentionMatrixToMatrixFunc = Callable[[AttentionMatrix], Matrix2D]
"Type alias for a function that, given an attention matrix, returns a 2D matrix"

MATPLOTLIB_FIGURE_FMT: str = "svgz"
"format for saving matplotlib figures"

MatrixSaveFormat = Literal["png", "svg", "svgz"]
"Type alias for the format to save a matrix as when saving raw matrix, not matplotlib figure"

MATRIX_SAVE_NORMALIZE: bool = False
"default for whether to normalize the matrix to range [0, 1]"

MATRIX_SAVE_CMAP: str = "viridis"
"default colormap for saving matrices"

MATRIX_SAVE_FMT: MatrixSaveFormat = "svgz"
"default format for saving matrices"

MATRIX_SAVE_SVG_TEMPLATE: str = """<svg xmlns="http://www.w3.org/2000/svg" width="{m}" height="{n}" viewBox="0 0 {m} {n}" image-rendering="pixelated"> <image href="data:image/png;base64,{png_base64}" width="{m}" height="{n}" /> </svg>"""
"template for saving an `n` by `m` matrix as an svg/svgz"


@overload  # without keyword arguments, returns decorated function
def matplotlib_figure_saver(
	func: Callable[[AttentionMatrix, plt.Axes], None],
	*args,
	fmt: str = MATPLOTLIB_FIGURE_FMT,
) -> AttentionMatrixFigureFunc: ...
@overload  # with keyword arguments, returns decorator
def matplotlib_figure_saver(
	func: None = None,
	*args,
	fmt: str = MATPLOTLIB_FIGURE_FMT,
) -> Callable[
	[Callable[[AttentionMatrix, plt.Axes], None], str], AttentionMatrixFigureFunc
]: ...
def matplotlib_figure_saver(
	func: Callable[[AttentionMatrix, plt.Axes], None] | None = None,
	*args,
	fmt: str = MATPLOTLIB_FIGURE_FMT,
) -> Union[
	AttentionMatrixFigureFunc,
	Callable[
		[Callable[[AttentionMatrix, plt.Axes], None], str], AttentionMatrixFigureFunc
	],
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

		wrapped.figure_save_fmt = fmt  # type: ignore[attr-defined]

		return wrapped

	if callable(func):
		# Handle no-arguments case
		return decorator(func)
	else:
		# Handle arguments case
		return decorator

def matplotlib_multifigure_saver(
	names: Sequence[str],
	fmt: str = MATPLOTLIB_FIGURE_FMT,
) -> Callable[
	# decorator takes in function
	# which takes a matrix and a dictionary of axes corresponding to the names
	[Callable[[AttentionMatrix, dict[str, plt.Axes]], None]],
	# returns the decorated function
	AttentionMatrixFigureFunc,
]:
	def decorator(
		func: Callable[[AttentionMatrix, dict[str, plt.Axes]], None],
	) -> AttentionMatrixFigureFunc:
		
		func_name: str = func.__name__

		@functools.wraps(func)
		def wrapped(attn_matrix: AttentionMatrix, save_dir: Path) -> None:
			# set up axes
			axes_dict: dict[str, plt.Axes] = {
				name: plt.subplots(figsize=(10, 10))[1] 
				for name in names
			}
			# run the function, make plots
			func(attn_matrix, axes_dict)

			# save the figures
			for name, ax in axes_dict.items():
				fig_path: Path = save_dir / f"{func_name}.{name}.{fmt}"
				ax.figure.tight_layout()
				ax.figure.savefig(fig_path)
				plt.close(ax.figure)

		wrapped.figure_save_fmt = fmt


def matrix_to_image_preprocess(
	matrix: Matrix2D,
	normalize: bool = False,
	cmap: str | Colormap = "viridis",
	diverging_colormap: bool = False,
	normalize_min: float | None = None,
) -> Matrix2Drgb:
	"""preprocess a 2D matrix into a plottable heatmap image

	# Parameters:
	 - `matrix : Matrix2D`
	    input matrix
	 - `normalize : bool`
	    whether to normalize the matrix to range [0, 1]
	   (defaults to `MATRIX_SAVE_NORMALIZE`)
	 - `cmap : str|Colormap`
	    the colormap to use for the matrix
	   (defaults to `MATRIX_SAVE_CMAP`)
	 - `diverging_colormap : bool`
	    if True and using a diverging colormap, ensures 0 values map to the center of the colormap
	    (defaults to False)
	 - `normalize_min : float|None`
	    if a float, then for `normalize=True` and `diverging_colormap=False`, the minimum value to normalize to (generally set this to zero?).
	    if `None`, then the minimum value of the matrix is used.
	    if `diverging_colormap=True` OR `normalize=False`, this **must** be `None`.
	    (defaults to `None`)

	# Returns:
	 - `Matrix2Drgb`
	"""
	# check dims
	assert matrix.ndim == 2, f"Matrix must be 2D, got {matrix.ndim = }"

	# check matrix is not empty
	assert matrix.size > 0, "Matrix cannot be empty"

	if normalize_min is not None:
		assert not diverging_colormap, (
			"normalize_min cannot be used with diverging_colormap=True"
		)
		assert normalize, "normalize_min cannot be used with normalize=False"

	# Normalize the matrix to range [0, 1]
	normalized_matrix: Matrix2D
	if normalize:
		if diverging_colormap:
			# For diverging colormaps, we want to center around 0
			max_abs: float = max(abs(matrix.max()), abs(matrix.min()))
			normalized_matrix = (matrix / (2 * max_abs)) + 0.5
		else:
			max_val: float = matrix.max()
			min_val: float
			if normalize_min is not None:
				min_val = normalize_min
				assert min_val < max_val, "normalize_min must be less than matrix max"
				assert min_val >= matrix.min(), (
					"normalize_min must less than matrix min"
				)
			else:
				min_val = matrix.min()

			normalized_matrix = (matrix - min_val) / (max_val - min_val)
	else:
		if diverging_colormap:
			assert matrix.min() >= -1 and matrix.max() <= 1, (
				"For diverging colormaps without normalization, matrix values must be in range [-1, 1]"
			)
			normalized_matrix = matrix
		else:
			assert matrix.min() >= 0 and matrix.max() <= 1, (
				"Matrix values must be in range [0, 1], or normalize must be True"
			)
			normalized_matrix = matrix

	# get the colormap
	cmap_: Colormap
	if isinstance(cmap, str):
		cmap_ = matplotlib.colormaps[cmap]
	elif isinstance(cmap, Colormap):
		cmap_ = cmap
	else:
		raise TypeError(
			f"Invalid type for {cmap = }, {type(cmap) = }, must be str or Colormap"
		)

	# Apply the colormap
	rgb_matrix: Float[np.ndarray, "n m channels=3"] = (  # noqa: F722
		cmap_(normalized_matrix)[:, :, :3] * 255
	).astype(np.uint8)  # Drop alpha channel

	assert rgb_matrix.shape == (
		matrix.shape[0],
		matrix.shape[1],
		3,
	), f"Matrix after colormap must have 3 channels, got {rgb_matrix.shape = }"

	return rgb_matrix


@overload
def matrix2drgb_to_png_bytes(matrix: Matrix2Drgb, buffer: None = None) -> bytes: ...
@overload
def matrix2drgb_to_png_bytes(matrix: Matrix2Drgb, buffer: io.BytesIO) -> None: ...
def matrix2drgb_to_png_bytes(
	matrix: Matrix2Drgb, buffer: io.BytesIO | None = None
) -> bytes | None:
	"""Convert a `Matrix2Drgb` to valid PNG bytes via PIL

	- if `buffer` is provided, it will write the PNG bytes to the buffer and return `None`
	- if `buffer` is not provided, it will return the PNG bytes

	# Parameters:
	 - `matrix : Matrix2Drgb`
	 - `buffer : io.BytesIO | None`
	   (defaults to `None`, in which case it will return the PNG bytes)

	# Returns:
	 - `bytes|None`
	   `bytes` if `buffer` is `None`, otherwise `None`
	"""

	pil_img: Image.Image = Image.fromarray(matrix, mode="RGB")
	if buffer is None:
		buffer = io.BytesIO()
		pil_img.save(buffer, format="PNG")
		buffer.seek(0)
		return buffer.read()
	else:
		pil_img.save(buffer, format="PNG")
		return None


def matrix_as_svg(
	matrix: Matrix2D,
	normalize: bool = MATRIX_SAVE_NORMALIZE,
	cmap: str | Colormap = MATRIX_SAVE_CMAP,
	diverging_colormap: bool = False,
	normalize_min: float | None = None,
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
	 - `diverging_colormap : bool`
	    if True and using a diverging colormap, ensures 0 values map to the center of the colormap
	    (defaults to False)
	 - `normalize_min : float|None`
	    if a float, then for `normalize=True` and `diverging_colormap=False`, the minimum value to normalize to (generally set this to zero?)
	    if `None`, then the minimum value of the matrix is used
	    if `diverging_colormap=True` OR `normalize=False`, this **must** be `None`
	    (defaults to `None`)


	# Returns:
	 - `str`
	   the SVG content for the matrix
	"""
	# Get the dimensions of the matrix
	m, n = matrix.shape

	# Preprocess the matrix into an RGB image
	matrix_rgb: Matrix2Drgb = matrix_to_image_preprocess(
		matrix,
		normalize=normalize,
		cmap=cmap,
		diverging_colormap=diverging_colormap,
		normalize_min=normalize_min,
	)

	# Convert the RGB image to PNG bytes
	image_data: bytes = matrix2drgb_to_png_bytes(matrix_rgb)

	# Encode the PNG bytes as base64
	png_base64: str = base64.b64encode(image_data).decode("utf-8")

	# Generate the SVG content
	svg_content: str = MATRIX_SAVE_SVG_TEMPLATE.format(m=m, n=n, png_base64=png_base64)

	return svg_content


@overload  # with keyword arguments, returns decorator
def save_matrix_wrapper(
	func: None = None,
	*args,
	fmt: MatrixSaveFormat = MATRIX_SAVE_FMT,
	normalize: bool = MATRIX_SAVE_NORMALIZE,
	cmap: str | Colormap = MATRIX_SAVE_CMAP,
	diverging_colormap: bool = False,
	normalize_min: float | None = None,
) -> Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]: ...
@overload  # without keyword arguments, returns decorated function
def save_matrix_wrapper(
	func: AttentionMatrixToMatrixFunc,
	*args,
	fmt: MatrixSaveFormat = MATRIX_SAVE_FMT,
	normalize: bool = MATRIX_SAVE_NORMALIZE,
	cmap: str | Colormap = MATRIX_SAVE_CMAP,
	diverging_colormap: bool = False,
	normalize_min: float | None = None,
) -> AttentionMatrixFigureFunc: ...
def save_matrix_wrapper(
	func: AttentionMatrixToMatrixFunc | None = None,
	*args,
	fmt: MatrixSaveFormat = MATRIX_SAVE_FMT,
	normalize: bool = MATRIX_SAVE_NORMALIZE,
	cmap: str | Colormap = MATRIX_SAVE_CMAP,
	diverging_colormap: bool = False,
	normalize_min: float | None = None,
) -> (
	AttentionMatrixFigureFunc
	| Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]
):
	"""
	Decorator for functions that process an attention matrix and save it as an SVGZ image.
	Can handle both argumentless usage and with arguments.

	# Parameters:

	 - `func : AttentionMatrixToMatrixFunc|None`
	    Either the function to decorate (in the no-arguments case) or `None` when used with arguments.
	 - `fmt : MatrixSaveFormat, keyword-only`
	    The format to save the matrix as. Defaults to `MATRIX_SAVE_FMT`.
	 - `normalize : bool, keyword-only`
	    Whether to normalize the matrix to range [0, 1]. Defaults to `False`.
	 - `cmap : str, keyword-only`
	    The colormap to use for the matrix. Defaults to `MATRIX_SVG_CMAP`.
	 - `diverging_colormap : bool`
	    if True and using a diverging colormap, ensures 0 values map to the center of the colormap
	    (defaults to False)
	 - `normalize_min : float|None`
	    if a float, then for `normalize=True` and `diverging_colormap=False`, the minimum value to normalize to (generally set this to zero?)
	    if `None`, then the minimum value of the matrix is used
	    if `diverging_colormap=True` OR `normalize=False`, this **must** be `None`
	    (defaults to `None`)

	# Returns:

	`AttentionMatrixFigureFunc|Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]`

	- `AttentionMatrixFigureFunc` if `func` is `AttentionMatrixToMatrixFunc` (no arguments case)
	- `Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]` if `func` is `None` -- returns the decorator which will then be applied to the  (with arguments case)

	# Usage:

	```python
	@save_matrix_wrapper
	def identity_matrix(matrix):
	    return matrix

	@save_matrix_wrapper(normalize=True, fmt="png")
	def scale_matrix(matrix):
	    return matrix * 2

	@save_matrix_wrapper(normalize=True, cmap="plasma")
	def scale_matrix(matrix):
	    return matrix * 2

	```
	"""

	assert len(args) == 0, "This decorator only supports keyword arguments"

	assert (
		fmt in MatrixSaveFormat.__args__  # type: ignore[attr-defined]
	), f"Invalid format {fmt = }, must be one of {MatrixSaveFormat.__args__}"  # type: ignore[attr-defined]

	def decorator(
		func: Callable[[AttentionMatrix], Matrix2D],
	) -> AttentionMatrixFigureFunc:
		@functools.wraps(func)
		def wrapped(attn_matrix: AttentionMatrix, save_dir: Path) -> None:
			fig_path: Path = save_dir / f"{func.__name__}.{fmt}"
			processed_matrix: Matrix2D = func(attn_matrix)

			if fmt == "png":
				processed_matrix_rgb: Matrix2Drgb = matrix_to_image_preprocess(
					processed_matrix,
					normalize=normalize,
					cmap=cmap,
					diverging_colormap=diverging_colormap,
					normalize_min=normalize_min,
				)
				image_data: bytes = matrix2drgb_to_png_bytes(processed_matrix_rgb)
				fig_path.write_bytes(image_data)

			else:
				svg_content: str = matrix_as_svg(
					processed_matrix,
					normalize=normalize,
					cmap=cmap,
					diverging_colormap=diverging_colormap,
					normalize_min=normalize_min,
				)

				if fmt == "svgz":
					with gzip.open(fig_path, "wt") as f:
						f.write(svg_content)

				else:
					fig_path.write_text(svg_content, encoding="utf-8")

		wrapped.figure_save_fmt = fmt  # type: ignore[attr-defined]

		return wrapped

	if callable(func):
		# Handle no-arguments case
		return decorator(func)
	else:
		# Handle arguments case
		return decorator
