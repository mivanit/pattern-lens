"""writes indexes to the model directory for the frontend to use or for record keeping"""


import json
from pathlib import Path
import importlib.resources

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


def generate_functions_jsonl(path: Path):
    "unions all functions from file and current `ATTENTION_MATRIX_FIGURE_FUNCS` into a `functions.jsonl` file"
    functions_file: Path = path / "functions.jsonl"
    existing_functions: set[str] = set()

    if functions_file.exists():
        with open(functions_file, "r") as f:
            for line in f:
                func_data: dict = json.loads(line)
                existing_functions.add(func_data["name"])

    # Add any new functions from ALL_FUNCTIONS
    all_functions: set[str] = existing_functions.union(
        set([func.__name__ for func in ATTENTION_MATRIX_FIGURE_FUNCS])
    )

    with open(functions_file, "w") as f:
        for func in sorted(all_functions):
            json.dump({"name": func}, f)
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