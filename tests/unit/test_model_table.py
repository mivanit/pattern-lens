"""Tests for model_table: CSV fetch, cache, and parsing."""

from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pattern_lens.model_table import (
	ModelInfo,
	_download_csv,
	fetch_model_table,
	get_model_params,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_CSV: str = """\
name.default_alias,n_params.as_int,something_else
gpt2-small,85000000,ignored
pythia-14m,14000000,ignored
pythia-2.8b,2500000000,ignored
"""

SAMPLE_TABLE: dict[str, ModelInfo] = {
	"gpt2-small": ModelInfo(name="gpt2-small", n_params=85_000_000),
	"pythia-14m": ModelInfo(name="pythia-14m", n_params=14_000_000),
	"pythia-2.8b": ModelInfo(name="pythia-2.8b", n_params=2_500_000_000),
}


# ===========================================================================
# get_model_params
# ===========================================================================


class TestGetModelParams:
	def test_get_model_params_found(self) -> None:
		"""Lookup known model returns correct count."""
		assert get_model_params("gpt2-small", SAMPLE_TABLE) == 85_000_000

	def test_get_model_params_not_found(self) -> None:
		"""KeyError for unknown model."""
		with pytest.raises(KeyError, match="nonexistent-model"):
			get_model_params("nonexistent-model", SAMPLE_TABLE)


# ===========================================================================
# fetch_model_table (cache / download logic)
# ===========================================================================


class TestFetchModelTable:
	def test_fetch_model_table_from_cache(self, tmp_path: Path) -> None:
		"""Reads cache file, no network call."""
		cache_path: Path = tmp_path / "model_table.csv"
		cache_path.write_text(SAMPLE_CSV)

		with (
			patch("pattern_lens.model_table.MODEL_TABLE_CACHE", cache_path),
			patch("pattern_lens.model_table._download_csv") as mock_download,
		):
			table: dict[str, ModelInfo] = fetch_model_table()
			mock_download.assert_not_called()
			assert "gpt2-small" in table

	def test_fetch_model_table_downloads(self, tmp_path: Path) -> None:
		"""Mock urlopen, verify download + cache write."""
		cache_path: Path = tmp_path / "model_table.csv"
		# cache doesn't exist yet, so it should download

		with (
			patch("pattern_lens.model_table.MODEL_TABLE_CACHE", cache_path),
			patch(
				"pattern_lens.model_table._download_csv",
				return_value=SAMPLE_CSV,
			) as mock_download,
		):
			table: dict[str, ModelInfo] = fetch_model_table()
			mock_download.assert_called_once()
			assert "gpt2-small" in table

	def test_fetch_model_table_force_refresh(self, tmp_path: Path) -> None:
		"""Cache exists but force_refresh=True → _download_csv still called."""
		cache_path: Path = tmp_path / "model_table.csv"
		cache_path.write_text(SAMPLE_CSV)

		with (
			patch("pattern_lens.model_table.MODEL_TABLE_CACHE", cache_path),
			patch(
				"pattern_lens.model_table._download_csv",
				return_value=SAMPLE_CSV,
			) as mock_download,
		):
			table: dict[str, ModelInfo] = fetch_model_table(force_refresh=True)
			mock_download.assert_called_once()
			assert "gpt2-small" in table


# ===========================================================================
# _download_csv
# ===========================================================================


class TestDownloadCSV:
	@patch("pattern_lens.model_table.urllib.request.urlopen")
	def test_download_csv_network_failure(
		self, mock_urlopen: MagicMock, tmp_path: Path
	) -> None:
		"""urllib.request.urlopen raises URLError → propagates."""
		mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
		with pytest.raises(urllib.error.URLError, match="Connection refused"):
			_download_csv("https://example.com/table.csv", tmp_path / "cache.csv")
