from pattern_lens.consts import AttentionMatrix
from pattern_lens.figure_util import (
    AttentionMatrixFigureFunc,
    save_matrix_as_svgz_wrapper,
    Matrix2D,
)


ATTENTION_MATRIX_FIGURE_FUNCS: list[AttentionMatrixFigureFunc] = list()


def register_attn_figure_func(
    func: AttentionMatrixFigureFunc,
) -> AttentionMatrixFigureFunc:
    """decorator for registering attention matrix figure function

    if you want to add a new figure function, you should use this decorator

        # Parameters:
         - `func : AttentionMatrixFigureFunc`
           your function, which should take an attention matrix and path

        # Returns:
         - `AttentionMatrixFigureFunc`
           your function, after we add it to `ATTENTION_MATRIX_FIGURE_FUNCS`

    # Usage:
    ```python
    @register_attn_figure_func
    def my_new_figure_func(attn_matrix: AttentionMatrix, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(attn_matrix, cmap="viridis")
        ax.set_title("My New Figure Function")
        ax.axis("off")
        plt.savefig(path / "my_new_figure_func", format="svgz")
        plt.close(fig)
    ```

    """
    global ATTENTION_MATRIX_FIGURE_FUNCS

    ATTENTION_MATRIX_FIGURE_FUNCS.append(func)

    return func


# @register_attn_figure_func
# @matplotlib_figure_saver
# def raw(attn_matrix: AttentionMatrix, ax: plt.Axes) -> None:
#     ax.matshow(attn_matrix, cmap="viridis")
#     ax.set_title("Raw Attention Pattern")
#     ax.axis("off")


@save_matrix_as_svgz_wrapper(normalize=False, cmap="viridis")
def raw(attn_matrix: AttentionMatrix) -> Matrix2D:
    return attn_matrix
