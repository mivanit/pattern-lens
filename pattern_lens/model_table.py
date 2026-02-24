"""Fetch and cache TransformerLens model parameter table from GitHub.

Used by the parallel model scheduler to estimate VRAM requirements
based on parameter counts.

CLI usage::

    python -m pattern_lens.model_table          # print table
    python -m pattern_lens.model_table -f       # force re-download
    python -m pattern_lens.model_table --cache-path  # show cache location
"""

import argparse
import importlib.resources
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MODEL_TABLE_URL: str = "https://raw.githubusercontent.com/mivanit/transformerlens-model-table/main/docs/model_table.csv"

_CACHE_FILENAME: str = "model_table.csv"


def _resolve_cache_path() -> Path:
	"""Resolve model table cache path.

	Uses ``.meta/local/`` when running from a repo clone (detected by the
	presence of a ``.meta/`` directory next to the package root), otherwise
	falls back to ``~/.cache/pattern_lens/``.
	"""
	import pattern_lens  # noqa: PLC0415

	pkg_root: Path = Path(str(importlib.resources.files(pattern_lens)))
	repo_root: Path = pkg_root.parent
	if (repo_root / ".meta").is_dir():
		return repo_root / ".meta" / "local" / _CACHE_FILENAME
	return Path.home() / ".cache" / "pattern_lens" / _CACHE_FILENAME


MODEL_TABLE_CACHE: Path = _resolve_cache_path()


@dataclass(frozen=True)
class ModelInfo:
	"""Basic model metadata from the TransformerLens model table."""

	name: str
	n_params: int


def _download_csv(url: str, cache_path: Path) -> str:
	"""Download CSV from URL and save to cache path.

	Returns the CSV content as a string.
	"""
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	with urllib.request.urlopen(url) as response:  # noqa: S310
		content: str = response.read().decode("utf-8")
	cache_path.write_text(content)
	return content


def _fetch_csv_content(force_refresh: bool = False) -> str:
	"""Return raw CSV content, downloading if necessary."""
	content: str
	if MODEL_TABLE_CACHE.exists() and not force_refresh:
		content = MODEL_TABLE_CACHE.read_text()
		print(f"Using cached model table from {MODEL_TABLE_CACHE}")
	else:
		print(f"Downloading model table from {MODEL_TABLE_URL}")
		content = _download_csv(MODEL_TABLE_URL, MODEL_TABLE_CACHE)
		print(f"Cached model table to {MODEL_TABLE_CACHE}")
	return content


def fetch_model_table_df(force_refresh: bool = False) -> pd.DataFrame:
	"""Fetch model table from GitHub as a pandas DataFrame (cached).

	Downloads the CSV on first call, then reads from cache on subsequent calls.
	Pass ``force_refresh=True`` to re-download.
	"""
	content: str = _fetch_csv_content(force_refresh=force_refresh)
	return pd.read_csv(io.StringIO(content))


def fetch_model_table(force_refresh: bool = False) -> dict[str, ModelInfo]:
	"""Fetch model table from GitHub, using local cache if available.

	Downloads the CSV on first call, then reads from cache on subsequent calls.
	Pass ``force_refresh=True`` to re-download.
	"""
	df: pd.DataFrame = fetch_model_table_df(force_refresh=force_refresh)
	df = df.dropna(subset=["name.default_alias", "n_params.as_int"])
	df = df[df["name.default_alias"] != ""]
	return {
		row["name.default_alias"]: ModelInfo(
			name=row["name.default_alias"],
			n_params=int(row["n_params.as_int"]),
		)
		for row in df.to_dict("records")
	}


def get_model_params(model_name: str, table: dict[str, ModelInfo]) -> int:
	"""Look up parameter count for a model by name.

	Raises ``KeyError`` if the model is not found in the table.
	"""
	if model_name in table:
		return table[model_name].n_params
	msg = (
		f"Model {model_name!r} not found in model table. "
		f"Available models: {sorted(table.keys())}"
	)
	raise KeyError(msg)


def main() -> None:
	"""CLI entrypoint: print the model table to stdout."""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Fetch and display the TransformerLens model parameter table.",
	)
	parser.add_argument(
		"-f",
		"--force-refresh",
		action="store_true",
		help="Re-download the CSV even if a cached copy exists.",
	)
	parser.add_argument(
		"--cache-path",
		action="store_true",
		help="Print the resolved cache file path and exit.",
	)
	args: argparse.Namespace = parser.parse_args()

	if args.cache_path:
		print(MODEL_TABLE_CACHE)
		return

	df: pd.DataFrame = fetch_model_table_df(force_refresh=args.force_refresh)
	print(df)


if __name__ == "__main__":
	main()
