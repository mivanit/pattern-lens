"""Load a HookedTransformer by name, resolving model name variants.

Different systems use different names for the same model:

- User alias: ``tiny-stories-1M`` (TransformerLens default alias)
- HuggingFace path: ``roneneldan/TinyStories-1M``
- ``cfg.model_name``: ``TinyStories-1M`` (HF path tail, set by TransformerLens)

This module provides two key functions:

:func:`resolve_model_name`
    Maps *any* variant to the TransformerLens default alias.  This is the
    name you pass to ``HookedTransformer.from_pretrained``.

:func:`sanitize_model_name`
    Maps *any* variant to a canonical filesystem-safe directory name
    (the default alias with disallowed characters replaced).
"""

from __future__ import annotations  # noqa: I001

import warnings
from typing import Any, Literal

import torch  # noqa: TC002
from transformer_lens import HookedTransformer  # type: ignore[import-untyped] # noqa: TC002
from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]  # noqa: TC002
from transformers.tokenization_utils_base import (
	PreTrainedTokenizerBase,  # type: ignore[import-untyped] # noqa: TC002
)

from pattern_lens.consts import sanitize_name_str

# ---------------------------------------------------------------------------
# Name resolver cache: {any_known_variant: default_alias}
# ---------------------------------------------------------------------------

_NAME_RESOLVER: dict[str, str] | None = None


def _build_name_resolver() -> dict[str, str]:
	"""Build ``{any_name_variant: default_alias}`` from the model table CSV.

	For each model, registers these keys, all mapping to ``default_alias``:

	- ``default_alias`` itself  (e.g. ``tiny-stories-1M``)
	- ``hf_name``               (e.g. ``roneneldan/TinyStories-1M``)
	- ``hf_name.split("/")[-1]`` (e.g. ``TinyStories-1M``, = ``cfg.model_name``)
	- ``sanitize_name_str()`` of each of the above

	Warns on collisions (two different models claiming the same key).
	"""
	from pattern_lens.model_table import fetch_model_table_df  # noqa: PLC0415

	df = fetch_model_table_df()
	df = df.dropna(subset=["name.default_alias"])
	df = df[df["name.default_alias"] != ""]

	resolver: dict[str, str] = {}

	for _, row in df.iterrows():
		default_alias: str = str(row["name.default_alias"])
		hf_name: str = str(row.get("name.huggingface", "") or "")

		cfg_model_name: str = (
			hf_name.rsplit("/", maxsplit=1)[-1]
			if ("/" in hf_name and hf_name)
			else hf_name
		)

		# Collect all variants
		variants: list[str] = [default_alias, sanitize_name_str(default_alias)]
		if hf_name:
			variants.extend([hf_name, sanitize_name_str(hf_name)])
		if cfg_model_name:
			variants.extend([cfg_model_name, sanitize_name_str(cfg_model_name)])

		for variant in variants:
			if not variant:
				continue
			if variant in resolver and resolver[variant] != default_alias:
				warnings.warn(
					f"resolve_model_name collision: {variant!r} maps to both "
					f"{resolver[variant]!r} and {default_alias!r}; keeping first",
					stacklevel=2,
				)
			else:
				resolver[variant] = default_alias

	return resolver


def get_name_resolver() -> dict[str, str]:
	"""Return the cached variant→default-alias mapping, building on first call."""
	global _NAME_RESOLVER  # noqa: PLW0603
	if _NAME_RESOLVER is None:
		try:
			_NAME_RESOLVER = _build_name_resolver()
		except Exception as exc:  # noqa: BLE001
			warnings.warn(
				f"Failed to build name resolver ({exc}); "
				"falling back to pass-through name resolution",
				stacklevel=2,
			)
			_NAME_RESOLVER = {}
	return _NAME_RESOLVER


def resolve_model_name(name: str) -> str:
	"""Resolve any model name variant to the TransformerLens default alias.

	Accepts user aliases, HuggingFace paths, ``cfg.model_name`` values,
	or already-sanitized forms.  Returns the default alias that can be
	passed directly to ``HookedTransformer.from_pretrained``.

	Resolution strategy:

	1. Try TransformerLens ``get_official_model_name`` (case-insensitive alias
		lookup) and map the result back to the default alias via CSV.
	2. Look up in the resolver table built from the model table CSV.
	3. Fallback: return *name* unchanged (unknown / custom model).
	"""
	# Step 1: try TL's alias resolution (covers aliases not in our CSV)
	try:
		from transformer_lens.loading_from_pretrained import (  # type: ignore[import-untyped]  # noqa: PLC0415
			get_official_model_name,
		)

		official_hf: str = get_official_model_name(name)
		resolver = get_name_resolver()
		if official_hf in resolver:
			return resolver[official_hf]
		# HF name resolved but not in our CSV — derive cfg.model_name form
		cfg_name: str = (
			official_hf.rsplit("/", maxsplit=1)[-1]
			if "/" in official_hf
			else official_hf
		)
		if cfg_name in resolver:
			return resolver[cfg_name]
		else:
			# Not in resolver at all; return the default alias as TL sees it
			return cfg_name
	except (ImportError, ValueError):
		pass

	# Step 2: look up in resolver table (works without torch)
	resolver = get_name_resolver()
	if name in resolver:
		return resolver[name]
	sanitized: str = sanitize_name_str(name)
	if sanitized in resolver:
		return resolver[sanitized]

	# Step 3: unknown model — return unchanged
	return name


def sanitize_model_name(name: str) -> str:
	"""Resolve any model name variant to the canonical filesystem-safe name.

	Equivalent to ``sanitize_name_str(resolve_model_name(name))``:
	first resolves the name to the TransformerLens default alias,
	then applies character-level sanitization for safe use as a directory name.
	"""
	return sanitize_name_str(resolve_model_name(name))


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
	"""Load a ``HookedTransformer``, accepting any model name variant.

	Drop-in replacement for ``HookedTransformer.from_pretrained``.
	All keyword arguments are forwarded verbatim.

	The returned model's ``cfg.model_name_sanitized`` is set to the
	canonical filesystem name so that downstream path construction is
	consistent.
	"""
	from transformer_lens import (  # noqa: PLC0415
		HookedTransformer as HookedTransformerCls,
	)

	resolved: str = resolve_model_name(model_name)
	model: HookedTransformerCls = HookedTransformerCls.from_pretrained(
		resolved,
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
