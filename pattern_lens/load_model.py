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
from typing import TYPE_CHECKING, Any, Literal

from pattern_lens.consts import sanitize_model_name

if TYPE_CHECKING:
	import torch
	from transformer_lens import HookedTransformer  # type: ignore[import-untyped]
	from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]
	from transformers.tokenization_utils_base import (
		PreTrainedTokenizerBase,  # type: ignore[import-untyped]
	)

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

	If *name* is already a recognised original name in the model table,
	returns it immediately without building the inverse mapping.
	Otherwise falls back to the ``{sanitized: original}`` inverse table.
	If still not found, returns *name* unchanged (it may be a custom model
	not in the TransformerLens catalogue).
	"""
	from pattern_lens.model_table import fetch_model_table  # noqa: PLC0415

	if name in fetch_model_table():
		return name

	return get_sanitize_inverse().get(name, name)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
	model_name: str,
	fold_ln: bool = True,
	center_writing_weights: bool = True,
	center_unembed: bool = True,
	refactor_factored_attn_matrices: bool = False,
	checkpoint_index: int | None = None,
	checkpoint_value: int | None = None,
	hf_model: AutoModelForCausalLM | None = None,
	device: str | torch.device | None = None,
	n_devices: int = 1,
	tokenizer: PreTrainedTokenizerBase | None = None,
	move_to_device: bool = True,
	fold_value_biases: bool = True,
	default_prepend_bos: bool | None = None,
	default_padding_side: Literal["left", "right"] = "right",
	dtype: str = "float32",
	first_n_layers: int | None = None,
	**from_pretrained_kwargs: Any,  # noqa: ANN401
) -> HookedTransformer:
	"""Load a ``HookedTransformer``, accepting sanitized or original names.

	Drop-in replacement for ``HookedTransformer.from_pretrained``.
	All keyword arguments are forwarded verbatim.

	The returned model's ``cfg.model_name_sanitized`` is set to the
	**sanitized** form so that downstream path construction is consistent.
	"""
	from transformer_lens import (  # noqa: PLC0415
		HookedTransformer as HookedTransformerCls,
	)

	original_name: str = unsanitize_model_name(model_name)
	model: HookedTransformerCls = HookedTransformerCls.from_pretrained(
		original_name,
		fold_ln=fold_ln,
		center_writing_weights=center_writing_weights,
		center_unembed=center_unembed,
		refactor_factored_attn_matrices=refactor_factored_attn_matrices,
		checkpoint_index=checkpoint_index,
		checkpoint_value=checkpoint_value,
		hf_model=hf_model,
		device=device,
		n_devices=n_devices,
		tokenizer=tokenizer,
		move_to_device=move_to_device,
		fold_value_biases=fold_value_biases,
		default_prepend_bos=default_prepend_bos,
		default_padding_side=default_padding_side,
		dtype=dtype,
		first_n_layers=first_n_layers,
		**from_pretrained_kwargs,
	)
	model.cfg.model_name_sanitized = sanitize_model_name(model_name)  # type: ignore[attr-defined]
	return model
