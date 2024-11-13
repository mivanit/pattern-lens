import argparse
import functools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import tqdm
from muutils.json_serialize import json_serialize
from transformer_lens import HookedTransformer, HookedTransformerConfig

from pattern_lens.consts import ATTN_PATTERN_REGEX, DATA_DIR, AttentionCache
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
    return_attn_cache: bool = True,
) -> tuple[Path, AttentionCache | None]:
    """get activations for a given model and prompt, possibly from a cache

    if from a cache, prompt_meta must be passed and contain the prompt hash

    # Parameters:
     - `prompt : dict | None`
       (defaults to `None`)
     - `model : HookedTransformer`
     - `save_path : Path`
       (defaults to `Path(DATA_DIR)`)
     - `return_attn_cache : bool`
       will return `None` as the second element if `False`
       (defaults to `True`)

    # Returns:
     - `tuple[Path, AttentionCache|None]`
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

    # compute activations
    with torch.no_grad():
        # TODO: batching?
        _, cache = model.run_with_cache(
            prompt_str,
            names_filter=lambda key: ATTN_PATTERN_REGEX.match(key) is not None,
            return_type=None,
        )

    cache_np: AttentionCache = {k: v.detach().cpu().numpy() for k, v in cache.items()}

    # save activations
    activations_path: Path = prompt_dir / "activations.npz"
    np.savez_compressed(
        activations_path,
        **cache_np,
    )

    # return path and cache
    if return_attn_cache:
        return activations_path, cache
    else:
        return activations_path, None


def get_activations(
    prompt: dict,
    model: HookedTransformer | str,
    save_path: Path = Path(DATA_DIR),
    allow_disk_cache: bool = True,
    return_attn_cache: bool = True,
) -> tuple[Path, AttentionCache]:
    augment_prompt_with_hash(prompt)
    # from cache
    if allow_disk_cache:
        try:
            path, cache = load_activations(
                model_name=model.model_name,
                prompt=prompt,
                save_path=save_path,
            )
            if return_attn_cache:
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
        return_attn_cache=True,
    )


def main():

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
        "--not-raw-prompts",
        "-r",
        action="store_true",
        help="If passed, prompts will be treated as pre-processed (truncated, hashed, etc) and not raw text",
    )

    arg_parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        required=False,
        help="The path to save the attention patterns",
        default=DATA_DIR,
    )

    # min and max prompt lengths
    arg_parser.add_argument(
        "--min_chars",
        type=int,
        required=False,
        help="The minimum number of characters for a prompt",
        default=100,
    )
    arg_parser.add_argument(
        "--max_chars",
        type=int,
        required=False,
        help="The maximum number of characters for a prompt",
        default=1000,
    )

    # number of samples
    arg_parser.add_argument(
        "--n_samples",
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

    args: argparse.Namespace = arg_parser.parse_args()

    print(f"args parsed: {args}")

    # load model
    model_name: str = args.model
    model: HookedTransformer = HookedTransformer.from_pretrained(model_name)
    model.model_name = model_name
    model.cfg.model_name = model_name
    print("model loaded")

    # save model info
    save_path: Path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model_path: Path = save_path / model_name
    model_cfg: HookedTransformerConfig
    model_cfg = model.cfg
    model_path.mkdir(parents=True, exist_ok=True)
    with open(model_path / "model_cfg.json", "w") as f:
        json.dump(json_serialize(asdict(model_cfg)), f)

    # load prompts
    prompts: list[dict]
    if args.not_raw_prompts:
        with open(model_path / "prompts.jsonl", "r") as f:
            prompts = [json.loads(line) for line in f.readlines()]
    else:
        prompts = load_text_data(
            Path(args.prompts),
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            shuffle=True,
        )
    # truncate to n_samples
    prompts = prompts[: args.n_samples]
    print(f"{len(prompts)} prompts loaded")

    save_path: Path = Path(args.save_path)

    # get activations
    list(
        tqdm.tqdm(
            map(
                functools.partial(
                    get_activations,
                    model=model,
                    save_path=save_path,
                    allow_disk_cache=not args.force,
                    return_attn_cache=False,
                ),
                prompts,
            ),
            total=len(prompts),
            desc="Computing activations",
        )
    )

    generate_models_jsonl(save_path)
    generate_prompts_jsonl(save_path / model_name)


if __name__ == "__main__":
    main()
