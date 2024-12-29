"""implements a bunch of types, default values, and templates which are useful for figure functions

notably, you can use the decorators `matplotlib_figure_saver`, `save_matrix_wrapper` to make your functions save figures
"""

from pathlib import Path
from typing import Callable, Literal, overload, Union
import functools
import base64
import gzip
import io

from PIL import Image
import numpy as np
from jaxtyping import Float, UInt8
import matplotlib
import matplotlib.pyplot as plt

from pattern_lens.consts import AttentionMatrix

AttentionMatrixFigureFunc = Callable[[AttentionMatrix, Path], None]
"Type alias for a function that, given an attention matrix, saves a figure"

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

        return wrapped

    if callable(func):
        # Handle no-arguments case
        return decorator(func)
    else:
        # Handle arguments case
        return decorator


def matrix_to_image_preprocess(
    matrix: Matrix2D,
    normalize: bool = MATRIX_SAVE_NORMALIZE,
    cmap: str = MATRIX_SAVE_CMAP,
) -> Matrix2Drgb:
    """preprocess a 2D matrix into a plottable heatmap image
    
    # Parameters:
     - `matrix : Matrix2D`   
        input matrix
     - `normalize : bool`   
        whether to normalize the matrix to range [0, 1]
       (defaults to `MATRIX_SAVE_NORMALIZE`)
     - `cmap : str`   
        the colormap to use for the matrix
       (defaults to `MATRIX_SAVE_CMAP`)
    
    # Returns:
     - `Matrix2Drgb`
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
        assert (
            matrix.min() >= 0 and matrix.max() <= 1
        ), "Matrix values must be in range [0, 1], or normalize must be True. got: min: {matrix.min() = }, max: {matrix.max() = }"
        normalized_matrix = matrix

    # get the colormap
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    # Apply the viridis colormap
    rgb_matrix: Float[np.ndarray, "n m channels=3"] = (  # noqa: F722
        cmap(normalized_matrix)[:, :, :3] * 255
    ).astype(np.uint8)  # Drop alpha channel

    assert (
        rgb_matrix.shape
        == (matrix.shape[0], matrix.shape[1], 3)
    ), f"Matrix after colormap must have 3 channels, got {rgb_matrix.shape = }"

    return rgb_matrix


@overload
def matrix2drgb_to_png_bytes(matrix: Matrix2Drgb, buffer: None) -> bytes: ...
@overload
def matrix2drgb_to_png_bytes(matrix: Matrix2Drgb, buffer: io.BytesIO) -> None: ...
def matrix2drgb_to_png_bytes(matrix: Matrix2Drgb, buffer: io.BytesIO|None = None) -> bytes|None:
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

    pil_img: Image = Image.fromarray(matrix, mode="RGB")
    if buffer is None:
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.read()
    else:
        pil_img.save(buffer, format="PNG")

def matrix_as_svg(
    matrix: Matrix2D,
    normalize: bool = MATRIX_SAVE_NORMALIZE,
    cmap = MATRIX_SAVE_CMAP,
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
    # Get the dimensions of the matrix
    m, n = matrix.shape

    # Preprocess the matrix into an RGB image
    matrix_rgb: Matrix2Drgb = matrix_to_image_preprocess(matrix, normalize=normalize, cmap=cmap)

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
    cmap: str = MATRIX_SAVE_CMAP,
) -> Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]: ...
@overload  # without keyword arguments, returns decorated function
def save_matrix_wrapper(
    func: AttentionMatrixToMatrixFunc,
    *args,
    fmt: MatrixSaveFormat = MATRIX_SAVE_FMT,
    normalize: bool = MATRIX_SAVE_NORMALIZE,
    cmap: str = MATRIX_SAVE_CMAP,
) -> AttentionMatrixFigureFunc: ...
def save_matrix_wrapper(
    func: AttentionMatrixToMatrixFunc | None = None,
    *args,
    fmt: MatrixSaveFormat = MATRIX_SAVE_FMT,
    normalize: bool = MATRIX_SAVE_NORMALIZE,
    cmap=MATRIX_SAVE_CMAP,
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

    assert fmt in MatrixSaveFormat.__args__, f"Invalid format {fmt = }, must be one of {MatrixSaveFormat.__args__}"

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
                )
                image_data: bytes = matrix2drgb_to_png_bytes(processed_matrix_rgb)
                fig_path.write_bytes(image_data)
                
            else:
                svg_content: str = matrix_as_svg(processed_matrix, normalize=normalize, cmap=cmap)

                if fmt == "svgz":
                    with gzip.open(fig_path, "wt") as f:
                        f.write(svg_content)

                else:
                    fig_path.write_text(svg_content, encoding="utf-8")

        return wrapped

    if callable(func):
        # Handle no-arguments case
        return decorator(func)
    else:
        # Handle arguments case
        return decorator