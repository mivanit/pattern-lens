import argparse
from collections import defaultdict
import datetime
import functools
import itertools
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from muutils.json_serialize import json_serialize
from muutils.spinner import SpinnerContext
from muutils.parallel import run_maybe_parallel

from pattern_lens.attn_figure_funcs import ATTENTION_MATRIX_FIGURE_FUNCS
from pattern_lens.consts import DATA_DIR, AttentionMatrix, SPINNER_KWARGS
from pattern_lens.indexes import (
    generate_functions_jsonl,
    generate_models_jsonl,
    generate_prompts_jsonl,
)
from pattern_lens.load_activations import load_activations

# from transformer_lens import HookedTransformer, HookedTransformerConfig


class HTConfigMock:
    def __init__(self, **kwargs):
        self.n_layers: int
        self.n_heads: int
        self.model_name: str
        self.__dict__.update(kwargs)

    def serialize(self):
        return json_serialize(self.__dict__)

    @classmethod
    def load(cls, data: dict):
        return cls(**data)


def process_single_head(
    layer_idx: int,
    head_idx: int,
    attn_pattern: AttentionMatrix,
    save_dir: Path,
    force_overwrite: bool = False,
) -> dict[str, bool|Exception]:
    funcs_status: dict[str, bool|Exception] = dict()

    for func in ATTENTION_MATRIX_FIGURE_FUNCS:
        func_name: str = func.__name__
        fig_path: list[Path] = list(save_dir.glob(f"{func_name}.*"))

        if not force_overwrite and len(fig_path) > 0:
            funcs_status[func_name] = True
            continue

        try:
            func(attn_pattern, save_dir)
            funcs_status[func_name] = True

        except Exception as e:
            error_file = save_dir / f"{func.__name__}.error.txt"
            error_file.write_text(str(e))
            warnings.warn(
                f"Error in {func.__name__} for L{layer_idx}H{head_idx}: {str(e)}"
            )
            funcs_status[func_name] = e

    return funcs_status


def compute_and_save_figures(
    model_cfg: "HookedTransformerConfig|HTConfigMock",  # noqa: F821
    activations_path: Path,
    cache: dict,
    save_path: Path = Path(DATA_DIR),
    force_overwrite: bool = False,
    track_results: bool = False,
) -> None:
    prompt_dir: Path = activations_path.parent

    if track_results:
        results: defaultdict[
            str, # func name
            dict[
                tuple[int, int], # layer, head
                bool|Exception, # success or exception
            ]
        ] = defaultdict(dict)

    for layer_idx, head_idx in itertools.product(
        range(model_cfg.n_layers),
        range(model_cfg.n_heads),
    ):
        attn_pattern: AttentionMatrix = (
            cache[f"blocks.{layer_idx}.attn.hook_pattern"][0, head_idx].cpu().numpy()
        )
        save_dir: Path = prompt_dir / f"L{layer_idx}" / f"H{head_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        head_res: dict[str, bool|Exception] = process_single_head(
            layer_idx=layer_idx,
            head_idx=head_idx,
            attn_pattern=attn_pattern,
            save_dir=save_dir,
            force_overwrite=force_overwrite,
        )

        if track_results:
            for func_name, status in head_res.items():
                results[func_name][(layer_idx, head_idx)] = status
    
    # TODO: do something with results
        

    generate_prompts_jsonl(save_path / model_cfg.model_name)


def process_prompt(
    prompt: dict,
    model_cfg: "HookedTransformerConfig|HTConfigMock",  # noqa: F821
    save_path: Path,
    force_overwrite: bool = False,
) -> None:
    activations_path, cache = load_activations(
        model_name=model_cfg.model_name,
        prompt=prompt,
        save_path=save_path,
    )

    compute_and_save_figures(
        model_cfg=model_cfg,
        activations_path=activations_path,
        cache=cache,
        save_path=save_path,
        force_overwrite=force_overwrite,
    )


def figures_main(
    model_name: str,
    save_path: str,
    n_samples: int,
    force: bool,
    parallel: bool | int = True,
) -> None:
    with SpinnerContext(message="setting up paths", **SPINNER_KWARGS):
        # save model info or check if it exists
        save_path: Path = Path(save_path)
        model_path: Path = save_path / model_name
        with open(model_path / "model_cfg.json", "r") as f:
            model_cfg = HTConfigMock.load(json.load(f))

    with SpinnerContext(message="loading prompts", **SPINNER_KWARGS):
        # load prompts
        with open(model_path / "prompts.jsonl", "r") as f:
            prompts: list[dict] = [json.loads(line) for line in f.readlines()]
        # truncate to n_samples
        prompts = prompts[:n_samples]

    print(f"{len(prompts)} prompts loaded")

    list(
        run_maybe_parallel(
            func=functools.partial(
                process_prompt,
                model_cfg=model_cfg,
                save_path=save_path,
                force_overwrite=force,
            ),
            iterable=prompts,
            parallel=parallel,
            pbar="tqdm",
            pbar_kwargs=dict(
                desc="Making figures",
                ascii=" =",
            ),
        )
    )

    with SpinnerContext(
        message="updating jsonl metadata for models and functions", **SPINNER_KWARGS
    ):
        generate_models_jsonl(save_path)
        generate_functions_jsonl(save_path)


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
            "--save-path",
            "-s",
            type=str,
            required=False,
            help="The path to save the attention patterns",
            default=DATA_DIR,
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
        # force overwrite of existing figures
        arg_parser.add_argument(
            "--force",
            "-f",
            type=bool,
            required=False,
            help="Force overwrite of existing figures",
            default=False,
        )

        args: argparse.Namespace = arg_parser.parse_args()

    print(f"args parsed: {args}")

    figures_main(
        model_name=args.model,
        save_path=args.save_path,
        n_samples=args.n_samples,
        force=args.force,
    )


if __name__ == "__main__":
    main()
