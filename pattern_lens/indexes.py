"""writes indexes to the model directory for the frontend to use or for record keeping"""

import importlib.metadata
import importlib.resources
import inspect
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pattern_lens
from pattern_lens.attn_figure_funcs import (
	_FIGURE_NAMES_KEY,
	ATTENTION_MATRIX_FIGURE_FUNCS,
)


def generate_prompts_jsonl(model_dir: Path) -> None:
	"""creates a `prompts.jsonl` file with all the prompts in the model directory

	looks in all directories in `{model_dir}/prompts` for a `prompt.json` file
	"""
	prompts: list[dict] = list()
	for prompt_dir in (model_dir / "prompts").iterdir():
		prompt_file: Path = prompt_dir / "prompt.json"
		if prompt_file.exists():
			with open(prompt_file, "r") as f:
				prompt_data: dict = json.load(f)
				prompts.append(prompt_data)

	with open(model_dir / "prompts.jsonl", "w") as f:
		for prompt in prompts:
			f.write(json.dumps(prompt))
			f.write("\n")


def generate_models_jsonl(path: Path) -> None:
	"""creates a `models.jsonl` file with all the models"""
	models: list[dict] = list()
	for model_dir in (path).iterdir():
		model_cfg_path: Path = model_dir / "model_cfg.json"
		if model_cfg_path.exists():
			with open(model_cfg_path, "r") as f:
				model_cfg: dict = json.load(f)
				models.append(model_cfg)

	with open(path / "models.jsonl", "w") as f:
		for model in models:
			f.write(json.dumps(model))
			f.write("\n")


def get_func_metadata(func: Callable) -> list[dict[str, str | None]]:
	"""get metadata for a function

	# Parameters:
	- `func : Callable` which has a `_FIGURE_NAMES_KEY` (by default `_figure_names`) attribute

	# Returns:

	`list[dict[str, str | None]]`
	each dictionary is for a function, containing:

	- `name : str` : the name of the figure
	- `func_name : str`
		the name of the function. if not a multi-figure function, this is identical to `name`
		if it is a multi-figure function, then `name` is `{func_name}.{figure_name}`
	- `doc : str` : the docstring of the function
	- `figure_save_fmt : str | None` : the format of the figure that the function saves, using the `figure_save_fmt` attribute of the function. `None` if the attribute does not exist
	- `source : str | None` : the source file of the function
	- `code : str | None` : the source code of the function, split by line. `None` if the source file cannot be read

	"""
	source_file: str | None = inspect.getsourcefile(func)
	output: dict[str, str | None] = dict(
		func_name=func.__name__,
		doc=func.__doc__,
		figure_save_fmt=getattr(func, "figure_save_fmt", None),
		source=Path(source_file).as_posix() if source_file else None,
	)

	try:
		output["code"] = inspect.getsource(func)
	except OSError:
		output["code"] = None

	fig_names: list[str] | None = getattr(func, _FIGURE_NAMES_KEY, None)
	if fig_names:
		return [
			{
				"name": func_name,
				**output,
			}
			for func_name in fig_names
		]
	else:
		return [
			{
				"name": func.__name__,
				**output,
			},
		]


def generate_functions_jsonl(path: Path) -> None:
	"unions all functions from `figures.jsonl` and `ATTENTION_MATRIX_FIGURE_FUNCS` into the file"
	figures_file: Path = path / "figures.jsonl"
	existing_figures: dict[str, dict] = dict()

	if figures_file.exists():
		with open(figures_file, "r") as f:
			for line in f:
				func_data: dict = json.loads(line)
				existing_figures[func_data["name"]] = func_data

	# Add any new functions from ALL_FUNCTIONS
	new_functions_lst: list[dict] = list(
		itertools.chain.from_iterable(
			get_func_metadata(func) for func in ATTENTION_MATRIX_FIGURE_FUNCS
		),
	)
	new_functions: dict[str, dict] = {func["name"]: func for func in new_functions_lst}

	all_functions: list[dict] = list(
		{
			**existing_figures,
			**new_functions,
		}.values(),
	)

	with open(figures_file, "w") as f:
		for func_meta in sorted(all_functions, key=lambda x: x["name"]):
			json.dump(func_meta, f)
			f.write("\n")


def inline_assets(
	html: str,
	assets: list[tuple[Literal["script", "style"], str]],
	base_path: Path,
) -> str:
	"""Inline specified local CSS/JS files into an HTML document.

	Each entry in `assets` should be a tuple like `("script", "app.js")` or `("style", "style.css")`.

	# Parameters:
	- `html : str`
		input HTML content.
	- `assets : list[tuple[Literal["script", "style"], str]]`
		List of (tag_type, filename) tuples to inline.

	# Returns:
	`str` : Modified HTML content with inlined assets.
	"""
	for tag_type, filename in assets:
		if tag_type not in ("style", "script"):
			err_msg: str = f"Unsupported tag type: {tag_type}"
			raise ValueError(err_msg)

		# Dynamically create the pattern for the given tag and filename
		pattern: str = rf'<{tag_type} src="{filename}"></{tag_type}>'
		# assert it's in the text exactly once
		assert html.count(pattern) == 1, (
			f"Pattern {pattern} should be in the html exactly once, found {html.count(pattern) = }"
		)
		# read the content and create the replacement
		content: str = (base_path / filename).read_text()
		replacement: str = f"<{tag_type}>\n{content}\n</{tag_type}>"
		# perform the replacement
		html = html.replace(pattern, replacement)

	return html


def write_html_index(path: Path) -> None:
	"""writes an index.html file to the path"""
	frontend_resources_path: Path = Path(
		importlib.resources.files(pattern_lens).joinpath("frontend"),
	)
	html_index: str = (frontend_resources_path / "index.template.html").read_text(
		encoding="utf-8",
	)
	# inline assets
	html_index = inline_assets(
		html_index,
		[
			("style", "style.css"),
			("script", "util.js"),
			("script", "app.js"),
		],
		base_path=frontend_resources_path,
	)

	# add version
	pattern_lens_version: str = importlib.metadata.version("pattern-lens")
	html_index = html_index.replace("$$PATTERN_LENS_VERSION$$", pattern_lens_version)
	# write the index.html file
	with open(path / "index.html", "w", encoding="utf-8") as f:
		f.write(html_index)
