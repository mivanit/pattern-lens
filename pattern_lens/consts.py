"""implements some constants and types"""

import re
from typing import Literal

import numpy as np
from jaxtyping import Float

AttentionMatrix = Float[np.ndarray, "n_ctx n_ctx"]
"type alias for attention matrix"

ActivationCacheNp = dict[str, np.ndarray]
"type alias for a cache of activations, like a transformer_lens.ActivationCache"

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

ReturnCache = Literal["numpy", "torch"] | None
"return type for a cache of activations"

_SANITIZE_PATTERN: re.Pattern = re.compile(r"[^a-zA-Z0-9\-_.]")
"regex matching characters that are NOT allowed in sanitized model names"


def sanitize_name_str(name: str) -> str:
	"""Map a string to only letters, digits, and ``-_.`` for safe use as a directory name.

	Forward slashes (common in HuggingFace IDs like ``google/gemma-2b``) are
	replaced with ``-``.  Any remaining disallowed character becomes ``_``.
	The function is idempotent.

	.. note::

		This is a pure character-level transformation with no knowledge of
		model identity.  For a canonical filesystem name that resolves
		aliases, use :func:`pattern_lens.load_model.sanitize_model_name`.
	"""
	name = name.replace("/", "-")
	return _SANITIZE_PATTERN.sub("_", name)
