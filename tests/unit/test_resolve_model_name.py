"""tests for resolve_model_name and the resolving sanitize_model_name"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from pattern_lens.consts import sanitize_name_str
from pattern_lens.load_model import (
	_build_name_resolver,
	resolve_model_name,
	sanitize_model_name,
)
from pattern_lens.model_table import fetch_model_table_df

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV = pd.DataFrame(
	{
		"name.default_alias": [
			"tiny-stories-1M",
			"gpt2-small",
			"gemma-2b",
		],
		"name.huggingface": [
			"roneneldan/TinyStories-1M",
			"gpt2",
			"google/gemma-2b",
		],
		"n_params.as_int": [393216, 84934656, 2113929216],
	}
)


def _build_resolver_from(df: pd.DataFrame) -> dict[str, str]:
	"""Build a name resolver from a given DataFrame."""
	with patch(
		"pattern_lens.model_table.fetch_model_table_df",
		return_value=df,
	):
		return _build_name_resolver()


@pytest.fixture
def sample_resolver(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
	"""Provide a resolver built from SAMPLE_CSV and patch it into load_model."""
	resolver = _build_resolver_from(SAMPLE_CSV)
	monkeypatch.setattr(
		"pattern_lens.load_model.get_name_resolver",
		lambda: resolver,
	)
	return resolver


@pytest.fixture
def real_resolver(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
	"""Provide a resolver built from the real model table CSV."""
	resolver = _build_resolver_from(fetch_model_table_df())
	monkeypatch.setattr(
		"pattern_lens.load_model.get_name_resolver",
		lambda: resolver,
	)
	return resolver


# ---------------------------------------------------------------------------
# Unit tests with sample data
# ---------------------------------------------------------------------------


class TestResolveModelName:
	"""Test that resolve_model_name maps all variants to the default alias."""

	def test_default_alias_passthrough(self, sample_resolver):
		assert resolve_model_name("tiny-stories-1M") == "tiny-stories-1M"
		assert resolve_model_name("gpt2-small") == "gpt2-small"
		assert resolve_model_name("gemma-2b") == "gemma-2b"

	def test_hf_name_resolves(self, sample_resolver):
		assert resolve_model_name("roneneldan/TinyStories-1M") == "tiny-stories-1M"
		assert resolve_model_name("gpt2") == "gpt2-small"
		assert resolve_model_name("google/gemma-2b") == "gemma-2b"

	def test_cfg_model_name_resolves(self, sample_resolver):
		"""cfg.model_name = hf_name.split('/')[-1], e.g. 'TinyStories-1M'."""
		assert resolve_model_name("TinyStories-1M") == "tiny-stories-1M"

	def test_sanitized_hf_name_resolves(self, sample_resolver):
		"""sanitize_name_str('roneneldan/TinyStories-1M') == 'roneneldan-TinyStories-1M'."""
		assert resolve_model_name("roneneldan-TinyStories-1M") == "tiny-stories-1M"
		assert resolve_model_name("google-gemma-2b") == "gemma-2b"

	def test_unknown_model_passthrough(self, sample_resolver):
		assert resolve_model_name("my-custom-model") == "my-custom-model"

	def test_empty_string(self, sample_resolver):
		assert resolve_model_name("") == ""


class TestSanitizeModelName:
	"""Test that sanitize_model_name resolves + sanitizes."""

	def test_all_variants_same_output(self, sample_resolver):
		expected = "tiny-stories-1M"
		assert sanitize_model_name("tiny-stories-1M") == expected
		assert sanitize_model_name("TinyStories-1M") == expected
		assert sanitize_model_name("roneneldan/TinyStories-1M") == expected
		assert sanitize_model_name("roneneldan-TinyStories-1M") == expected

	def test_unknown_model_sanitized(self, sample_resolver):
		assert sanitize_model_name("org/my model") == "org-my_model"

	def test_idempotent(self, sample_resolver):
		for name in [
			"tiny-stories-1M",
			"TinyStories-1M",
			"roneneldan/TinyStories-1M",
			"my-custom-model",
		]:
			once = sanitize_model_name(name)
			twice = sanitize_model_name(once)
			assert once == twice, f"not idempotent for {name!r}: {once!r} != {twice!r}"


# ---------------------------------------------------------------------------
# Roundtrip tests against the real model table
# ---------------------------------------------------------------------------


def _all_model_rows() -> list[tuple[str, str]]:
	"""Return (default_alias, hf_name) pairs from the real model table."""
	df = fetch_model_table_df()
	df = df.dropna(subset=["name.default_alias"])
	df = df[df["name.default_alias"] != ""]
	return [
		(str(row["name.default_alias"]), str(row.get("name.huggingface", "") or ""))
		for _, row in df.iterrows()
	]


_MODEL_ROWS = _all_model_rows()
_MODEL_IDS = [alias for alias, _ in _MODEL_ROWS]


class TestRoundtripAllModels:
	"""For every model in the real table, all name variants must resolve to the same canonical name."""

	@pytest.mark.parametrize(
		("default_alias", "hf_name"),
		_MODEL_ROWS,
		ids=_MODEL_IDS,
	)
	def test_all_variants_resolve_to_alias(
		self,
		default_alias: str,
		hf_name: str,
		real_resolver: dict[str, str],
	):
		"""default_alias, hf_name, cfg.model_name, and sanitized forms all resolve identically."""
		expected = default_alias

		# The default alias itself
		assert resolve_model_name(default_alias) == expected

		# The sanitized default alias
		assert resolve_model_name(sanitize_name_str(default_alias)) == expected

		if hf_name:
			# The full HuggingFace path
			assert resolve_model_name(hf_name) == expected

			# The sanitized HF path (e.g. "roneneldan-TinyStories-1M")
			assert resolve_model_name(sanitize_name_str(hf_name)) == expected

			# The cfg.model_name form (HF tail)
			cfg_model_name = (
				hf_name.rsplit("/", maxsplit=1)[-1] if "/" in hf_name else hf_name
			)
			assert resolve_model_name(cfg_model_name) == expected

			# Sanitized cfg.model_name
			assert resolve_model_name(sanitize_name_str(cfg_model_name)) == expected

	@pytest.mark.parametrize(
		("default_alias", "hf_name"),
		_MODEL_ROWS,
		ids=_MODEL_IDS,
	)
	def test_sanitize_idempotent(
		self,
		default_alias: str,
		hf_name: str,
		real_resolver: dict[str, str],
	):
		"""sanitize_model_name is idempotent for all known name variants."""
		variants = [default_alias]
		if hf_name:
			cfg_model_name = (
				hf_name.rsplit("/", maxsplit=1)[-1] if "/" in hf_name else hf_name
			)
			variants.extend([hf_name, cfg_model_name])

		for name in variants:
			once = sanitize_model_name(name)
			twice = sanitize_model_name(once)
			assert once == twice, f"not idempotent for {name!r}: {once!r} != {twice!r}"


# ---------------------------------------------------------------------------
# Fallback / degradation tests
# ---------------------------------------------------------------------------


class TestResolverFallback:
	"""Test graceful degradation when model table is unavailable."""

	def test_empty_resolver_falls_back(self, monkeypatch: pytest.MonkeyPatch):
		"""With an empty resolver, resolve_model_name returns the input unchanged."""
		monkeypatch.setattr(
			"pattern_lens.load_model.get_name_resolver",
			dict,
		)
		# Ensure TL import path also fails so we exercise pure fallback
		with patch.dict(
			"sys.modules",
			{
				"transformer_lens": None,
				"transformer_lens.loading_from_pretrained": None,
			},
		):
			assert resolve_model_name("TinyStories-1M") == "TinyStories-1M"

	def test_sanitize_falls_back_to_string_sanitize(
		self, monkeypatch: pytest.MonkeyPatch
	):
		"""With an empty resolver, sanitize_model_name still does char sanitization."""
		monkeypatch.setattr(
			"pattern_lens.load_model.get_name_resolver",
			dict,
		)
		with patch.dict(
			"sys.modules",
			{
				"transformer_lens": None,
				"transformer_lens.loading_from_pretrained": None,
			},
		):
			assert sanitize_model_name("org/model name") == "org-model_name"
