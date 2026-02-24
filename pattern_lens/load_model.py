"""Load a HookedTransformer by name, resolving sanitized names back to originals.

``sanitize_model_name`` is lossy -- ``google/gemma-2b`` becomes
``google-gemma-2b`` and there is no way to invert that without a lookup
table.  This module builds that table from the TransformerLens model
catalogue (fetched/cached by :mod:`pattern_lens.model_table`) and
exposes :func:`load_model` as a drop-in replacement for
``HookedTransformer.from_pretrained`` that accepts *either* the original
HuggingFace name or the sanitized directory name.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pattern_lens.consts import sanitize_model_name

if TYPE_CHECKING:
	from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Sanitize-inverse cache
# ---------------------------------------------------------------------------

_SANITIZE_INVERSE: dict[str, str] | None = None


def _build_sanitize_inverse() -> dict[str, str]:
	"""Build ``{sanitized_name: original_name}`` from the model table.

	Warns on collisions (two originals mapping to the same sanitized key).
	"""
	from pattern_lens.model_table import fetch_model_table  # noqa: PLC0415

	table = fetch_model_table()
	inverse: dict[str, str] = {}
	for original in table:
		sanitized = sanitize_model_name(original)
		if sanitized in inverse and inverse[sanitized] != original:
			warnings.warn(
				f"sanitize_model_name collision: {original!r} and "
				f"{inverse[sanitized]!r} both map to {sanitized!r}",
				stacklevel=2,
			)
		inverse[sanitized] = original
	return inverse


def get_sanitize_inverse() -> dict[str, str]:
	"""Return the cached sanitized→original mapping, building it on first call."""
	global _SANITIZE_INVERSE  # noqa: PLW0603
	if _SANITIZE_INVERSE is None:
		try:
			_SANITIZE_INVERSE = _build_sanitize_inverse()
		except Exception as exc:  # noqa: BLE001
			warnings.warn(
				f"Failed to build sanitize-inverse table ({exc}); "
				"falling back to pass-through name resolution",
				stacklevel=2,
			)
			_SANITIZE_INVERSE = {}
	return _SANITIZE_INVERSE


def unsanitize_model_name(name: str) -> str:
	"""Resolve a potentially-sanitized model name to the original HF name.

	If *name* is found in the inverse table, returns the original.
	Otherwise returns *name* unchanged (it may already be an original name
	or a custom model not in the TransformerLens catalogue).
	"""
	return get_sanitize_inverse().get(name, name)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_name: str, **kwargs: object) -> HookedTransformer:
	"""Load a ``HookedTransformer``, accepting sanitized or original names.

	Drop-in replacement for ``HookedTransformer.from_pretrained``.
	All *kwargs* are forwarded verbatim.

	The returned model's ``cfg.model_name_sanitized`` is set to the
	**sanitized** form so that downstream path construction is consistent.
	"""
	from transformer_lens import (  # noqa: PLC0415
		HookedTransformer as HookedTransformerCls,
	)

	original_name = unsanitize_model_name(model_name)
	model = HookedTransformerCls.from_pretrained(original_name, **kwargs)
	model.cfg.model_name_sanitized = sanitize_model_name(model_name)
	return model
