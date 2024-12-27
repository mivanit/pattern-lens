import argparse
import functools
import importlib.resources
import json
from dataclasses import asdict
from pathlib import Path
import re
from typing import Callable

import numpy as np
import torch
import tqdm
from muutils.spinner import SpinnerContext
from muutils.misc.numerical import shorten_numerical_to_str
from muutils.json_serialize import json_serialize
from transformer_lens import HookedTransformer, HookedTransformerConfig

import pattern_lens
from pattern_lens.consts import (
    ATTN_PATTERN_REGEX,
    DATA_DIR,
    ActivationCacheNp,
    SPINNER_KWARGS,
)
from pattern_lens.indexes import generate_models_jsonl, generate_prompts_jsonl
from pattern_lens.load_activations import (
    ActivationsMissingError,
    augment_prompt_with_hash,
    load_activations,
)
from pattern_lens.prompts import load_text_data


def compute_activations(
    prompt: dict,
    model: HookedTransformer | None = None,
    save_path: Path = Path(DATA_DIR),
    return_cache: bool = True,
    names_filter: Callable[[str], bool] | re.Pattern = ATTN_PATTERN_REGEX,
) -> tuple[Path, ActivationCacheNp | None]:
    """get activations for a given model and prompt, possibly from a cache

    if from a cache, prompt_meta must be passed and contain the prompt hash

    # Parameters:
     - `prompt : dict | None`
       (defaults to `None`)
     - `model : HookedTransformer`
     - `save_path : Path`
       (defaults to `Path(DATA_DIR)`)
     - `return_cache : bool`
       will return `None` as the second element if `False`
       (defaults to `True`)
     - `names_filter : Callable[[str], bool]|re.Pattern`
       a filter for the names of the activations to return. if an `re.Pattern`, will use `lambda key: names_filter.match(key) is not None`
       (defaults to `ATTN_PATTERN_REGEX`)

    # Returns:
     - `tuple[Path, ActivationCacheNp|None]`
    """
    assert model is not None, "model must be passed"
    assert "text" in prompt, "prompt must contain 'text' key"
    prompt_str: str = prompt["text"]

    # compute or get prompt metadata
    prompt_tokenized: list[str] = prompt.get(
        "tokens",
        model.tokenizer.tokenize(prompt_str),
    )
    prompt.update(
        dict(
            n_tokens=len(prompt_tokenized),
            tokens=prompt_tokenized,
        )
    )

    # save metadata
    prompt_dir: Path = save_path / model.model_name / "prompts" / prompt["hash"]
    prompt_dir.mkdir(parents=True, exist_ok=True)
    with open(prompt_dir / "prompt.json", "w") as f:
        json.dump(prompt, f)

    # set up names filter
    names_filter_fn: Callable[[str], bool]
    if isinstance(names_filter, re.Pattern):
        names_filter_fn = lambda key: names_filter.match(key) is not None  # noqa: E731
    else:
        names_filter_fn = names_filter

    # compute activations
    with torch.no_grad():
        # TODO: batching?
        _, cache = model.run_with_cache(
            prompt_str,
            names_filter=names_filter_fn,
            return_type=None,
        )

    cache_np: ActivationCacheNp = {
        k: v.detach().cpu().numpy() for k, v in cache.items()
    }

    # save activations
    activations_path: Path = prompt_dir / "activations.npz"
    np.savez_compressed(
        activations_path,
        **cache_np,
    )

    # return path and cache
    if return_cache:
        return activations_path, cache_np
    else:
        return activations_path, None


def get_activations(
    prompt: dict,
    model: HookedTransformer | str,
    save_path: Path = Path(DATA_DIR),
    allow_disk_cache: bool = True,
    return_cache: bool = True,
) -> tuple[Path, ActivationCacheNp | None]:
    augment_prompt_with_hash(prompt)
    # from cache
    if allow_disk_cache:
        try:
            path, cache = load_activations(
                model_name=model.model_name,
                prompt=prompt,
                save_path=save_path,
            )
            if return_cache:
                return path, cache
            else:
                return path, None
        except ActivationsMissingError:
            pass

    # compute them
    if isinstance(model, str):
        model = HookedTransformer.from_pretrained(model)
        model.model_name = model

    return compute_activations(
        prompt=prompt,
        model=model,
        save_path=save_path,
        return_cache=True,
    )


def activations_main(
    model_name: str,
    save_path: str,
    prompts_path: str,
    raw_prompts: bool,
    min_chars: int,
    max_chars: int,
    force: bool,
    n_samples: int,
    no_index_html: bool,
    shuffle: bool = False,
) -> None:
    with SpinnerContext(message="loading model", **SPINNER_KWARGS):
        model: HookedTransformer = HookedTransformer.from_pretrained(model_name)
        model.model_name = model_name
        model.cfg.model_name = model_name
        n_params: int = sum(p.numel() for p in model.parameters())
    print(
        f"loaded {model_name} with {shorten_numerical_to_str(n_params)} ({n_params}) parameters"
    )

    save_path: Path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model_path: Path = save_path / model_name
    with SpinnerContext(
        message=f"saving model info to {model_path.as_posix()}", **SPINNER_KWARGS
    ):
        model_cfg: HookedTransformerConfig
        model_cfg = model.cfg
        model_path.mkdir(parents=True, exist_ok=True)
        with open(model_path / "model_cfg.json", "w") as f:
            json.dump(json_serialize(asdict(model_cfg)), f)

    # load prompts
    with SpinnerContext(
        message=f"loading prompts from {prompts_path = }", **SPINNER_KWARGS
    ):
        prompts: list[dict]
        if raw_prompts:
            prompts = load_text_data(
                Path(prompts_path),
                min_chars=min_chars,
                max_chars=max_chars,
                shuffle=shuffle,
            )
        else:
            with open(model_path / "prompts.jsonl", "r") as f:
                prompts = [json.loads(line) for line in f.readlines()]
        # truncate to n_samples
        prompts = prompts[:n_samples]

    print(f"{len(prompts)} prompts loaded")
    save_path: Path = Path(save_path)

    # write index.html
    with SpinnerContext(message="writing index.html", **SPINNER_KWARGS):
        if not no_index_html:
            html_index: str = (
                importlib.resources.files(pattern_lens)
                .joinpath("frontend/index.html")
                .read_text(encoding="utf-8")
            )
            with open(save_path / "index.html", "w", encoding="utf-8") as f:
                f.write(html_index)

    # get activations
    list(
        tqdm.tqdm(
            map(
                functools.partial(
                    get_activations,
                    model=model,
                    save_path=save_path,
                    allow_disk_cache=not force,
                    return_cache=False,
                ),
                prompts,
            ),
            total=len(prompts),
            desc="Computing activations",
        )
    )

    with SpinnerContext(
        message="updating jsonl metadata for models and prompts", **SPINNER_KWARGS
    ):
        generate_models_jsonl(save_path)
        generate_prompts_jsonl(save_path / model_name)


def main():
    with SpinnerContext(message="parsing args", **SPINNER_KWARGS):
        arg_parser: argparse.ArgumentParser = argparse.ArgumentParser()
        # input and output
        arg_parser.add_argument(
            "--model",
            "-m",
            type=str,
            required=True,
            help="The model name to use",
        )

        arg_parser.add_argument(
            "--prompts",
            "-p",
            type=str,
            required=False,
            help="The path to the prompts file (jsonl with 'text' key on each line). If `None`, expects that `--figures` is passed and will generate figures for all prompts in the model directory",
            default=None,
        )

        arg_parser.add_argument(
            "--save-path",
            "-s",
            type=str,
            required=False,
            help="The path to save the attention patterns",
            default=DATA_DIR,
        )

        # min and max prompt lengths
        arg_parser.add_argument(
            "--min-chars",
            type=int,
            required=False,
            help="The minimum number of characters for a prompt",
            default=100,
        )
        arg_parser.add_argument(
            "--max-chars",
            type=int,
            required=False,
            help="The maximum number of characters for a prompt",
            default=1000,
        )

        # number of samples
        arg_parser.add_argument(
            "--n-samples",
            "-n",
            type=int,
            required=False,
            help="The max number of samples to process, do all in the file if None",
            default=None,
        )

        # force overwrite
        arg_parser.add_argument(
            "--force",
            "-f",
            action="store_true",
            help="If passed, will overwrite existing files",
        )

        # no index html
        arg_parser.add_argument(
            "--no-index-html",
            action="store_true",
            help="If passed, will not write an index.html file for the model",
        )

        # raw prompts
        arg_parser.add_argument(
            "--raw-prompts",
            "-r",
            action="store_true",
            help="pass if the prompts have not been split and tokenized (still needs keys 'text' and 'meta' for each item)",
        )

        # shuffle
        arg_parser.add_argument(
            "--shuffle",
            action="store_true",
            help="If passed, will shuffle the prompts",
        )

        args: argparse.Namespace = arg_parser.parse_args()

    print(f"args parsed: {args}")

    activations_main(
        model_name=args.model,
        save_path=args.save_path,
        prompts_path=args.prompts,
        raw_prompts=args.raw_prompts,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        force=args.force,
        n_samples=args.n_samples,
        no_index_html=args.no_index_html,
        shuffle=args.shuffle,
    )


if __name__ == "__main__":
    main()
