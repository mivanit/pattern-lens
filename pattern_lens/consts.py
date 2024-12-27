import re

import numpy as np
from jaxtyping import Float

AttentionMatrix = Float[np.ndarray, "n_ctx n_ctx"]
"type alias for attention matrix"

AttentionCache = dict[str, AttentionMatrix]
"type alias for a cache of attention matrices, subset of ActivationCache"

DATA_DIR: str = "attn_data"
"default directory for attention data"

ATTN_PATTERN_REGEX: re.Pattern = re.compile(r"blocks\.(\d+)\.attn\.hook_pattern")
"regex for finding attention patterns in model state dicts"

SPINNER_KWARGS: dict = dict(
    config=dict(success="✔️ "),
)
