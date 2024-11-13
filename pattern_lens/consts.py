import re
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

AttentionMatrix = Float[np.ndarray, "n_ctx n_ctx"]
"type alias for attention matrix"

AttentionMatrixFigureFunc = Callable[[AttentionMatrix, plt.Axes], None]
"Type alias for a function that takes an attention matrix and an axes and plots something"

AttentionCache = dict[str, AttentionMatrix]
"type alias for a cache of attention matrices, subset of ActivationCache"

FIGURE_FMT: str = "svgz"
"format for saving figures"

DATA_DIR: str = "attn_data"
"default directory for attention data"

ATTN_PATTERN_REGEX: re.Pattern = re.compile(r"blocks\.(\d+)\.attn\.hook_pattern")
"regex for finding attention patterns in model state dicts"
