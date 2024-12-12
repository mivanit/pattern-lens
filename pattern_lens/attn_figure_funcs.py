import matplotlib.pyplot as plt

from pattern_lens.consts import AttentionMatrix, AttentionMatrixFigureFunc


def raw(attn_matrix: AttentionMatrix, ax: plt.Axes) -> None:
    ax.matshow(attn_matrix, cmap="viridis")
    ax.set_title("Raw Attention Pattern")
    ax.axis("off")


# Add new functions to the list
ATTENTION_MATRIX_FIGURE_FUNCS: list[AttentionMatrixFigureFunc] = [
    raw,
]
