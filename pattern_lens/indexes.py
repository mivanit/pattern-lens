"""writes indexes to the model directory for the frontend to use or for record keeping"""

import inspect
import json
from pathlib import Path
import importlib.resources
from typing import Callable

import pattern_lens
from pattern_lens.attn_figure_funcs import ATTENTION_MATRIX_FIGURE_FUNCS


def generate_prompts_jsonl(model_dir: Path):
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


def generate_models_jsonl(path: Path):
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


def get_func_metadata(func: Callable) -> dict[str, str | None]:
    """get metadata for a function

    # Parameters:
     - `func : Callable`

    # Returns:

    `dict[str, str | None]`
    dictionary:

    - `name : str` : the name of the function
    - `doc : str` : the docstring of the function
    - `figure_save_fmt : str | None` : the format of the figure that the function saves, using the `figure_save_fmt` attribute of the function. `None` if the attribute does not exist
    - `source : str | None` : the source file of the function
    - `code : str | None` : the source code of the function, split by line. `None` if the source file cannot be read

    """
    output: dict[str, str | None] = dict(
        name=func.__name__,
        doc=func.__doc__,
        figure_save_fmt=getattr(func, "figure_save_fmt", None),
        source=inspect.getsourcefile(func),
    )

    try:
        output["code"] = inspect.getsource(func)
    except OSError:
        output["code"] = None

    return output


def generate_functions_jsonl(path: Path):
    "unions all functions from file and current `ATTENTION_MATRIX_FIGURE_FUNCS` into a `functions.jsonl` file"
    functions_file: Path = path / "functions.jsonl"
    existing_functions: dict[str, dict] = dict()

    if functions_file.exists():
        with open(functions_file, "r") as f:
            for line in f:
                func_data: dict = json.loads(line)
                existing_functions[func_data["name"]] = func_data

    # Add any new functions from ALL_FUNCTIONS
    new_functions: dict[str, dict] = {
        func.__name__: get_func_metadata(func) for func in ATTENTION_MATRIX_FIGURE_FUNCS
    }

    all_functions: list[dict] = list(
        {
            **existing_functions,
            **new_functions,
        }.values()
    )

    with open(functions_file, "w") as f:
        for func_meta in sorted(all_functions, key=lambda x: x["name"]):
            json.dump(func_meta, f)
            f.write("\n")


def write_html_index(path: Path):
    """writes an index.html file to the path"""
    html_index: str = (
        importlib.resources.files(pattern_lens)
        .joinpath("frontend/index.html")
        .read_text(encoding="utf-8")
    )
    with open(path / "index.html", "w", encoding="utf-8") as f:
        f.write(html_index)
