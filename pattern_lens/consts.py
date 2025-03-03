"""implements some constants and types"""

import re
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float

AttentionMatrix = Float[np.ndarray, "n_ctx n_ctx"]
"type alias for attention matrix"

ActivationCacheNp = dict[str, np.ndarray]
"type alias for a cache of activations, like a transformer_lens.ActivationCache"

ActivationCacheTorch = dict[str, torch.Tensor]
"type alias for a cache of activations, like a transformer_lens.ActivationCache but without the extras. useful for when loading from an npz file"

DATA_DIR: str = "attn_data"
"default directory for attention data"

ATTN_PATTERN_REGEX: re.Pattern = re.compile(r"blocks\.(\d+)\.attn\.hook_pattern")
"regex for finding attention patterns in model state dicts"

SPINNER_KWARGS: dict = dict(
	config=dict(success="✔️ "),
)
"default kwargs for `muutils.spinner.Spinner`"

DIVIDER_S1: str = "=" * 70
"divider string for separating sections"

DIVIDER_S2: str = "-" * 50
"divider string for separating subsections"

ReturnCache = Literal[None, "numpy", "torch"]
"return type for a cache of activations"
