import argparse
import functools
import itertools
import json
import multiprocessing as mp
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import tqdm
from muutils.json_serialize import json_serialize
from muutils.spinner import SpinnerContext

from pattern_lens.attn_figure_funcs import ATTENTION_MATRIX_FIGURE_FUNCS
from pattern_lens.consts import DATA_DIR, FIGURE_FMT, AttentionMatrix, SPINNER_KWARGS
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
) -> None:
    for func in ATTENTION_MATRIX_FIGURE_FUNCS:
        fig_path: Path = save_dir / f"{func.__name__}.{FIGURE_FMT}"

        if not force_overwrite and fig_path.exists():
            continue

        try:
            fig, ax = plt.subplots(figsize=(10, 10))
            func(attn_pattern, ax)
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close(fig)
        except Exception as e:
            error_file = save_dir / f"{func.__name__}.error.txt"
            error_file.write_text(str(e))
            warnings.warn(
                f"Error in {func.__name__} for L{layer_idx}H{head_idx}: {str(e)}"
            )


def compute_and_save_figures(
    model_cfg: "HookedTransformerConfig|HTConfigMock",  # noqa: F821
    activations_path: Path,
    cache: dict,
    save_path: Path = Path(DATA_DIR),
    force_overwrite: bool = False,
) -> None:
    prompt_dir: Path = activations_path.parent

    for layer_idx, head_idx in itertools.product(
        range(model_cfg.n_layers),
        range(model_cfg.n_heads),
    ):
        attn_pattern: AttentionMatrix = (
            cache[f"blocks.{layer_idx}.attn.hook_pattern"][0, head_idx].cpu().numpy()
        )
        save_dir: Path = prompt_dir / f"L{layer_idx}" / f"H{head_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        process_single_head(
            layer_idx=layer_idx,
            head_idx=head_idx,
            attn_pattern=attn_pattern,
            save_dir=save_dir,
            force_overwrite=force_overwrite,
        )

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

    with SpinnerContext(message="setting up paths", **SPINNER_KWARGS):
        # load model
        model_name: str = args.model

        # save model info or check if it exists
        save_path: Path = Path(args.save_path)
        model_path: Path = save_path / model_name
        with open(model_path / "model_cfg.json", "r") as f:
            model_cfg = HTConfigMock.load(json.load(f))

    with SpinnerContext(message="loading prompts", **SPINNER_KWARGS):
        # load prompts
        with open(model_path / "prompts.jsonl", "r") as f:
            prompts: list[dict] = [json.loads(line) for line in f.readlines()]
        # truncate to n_samples
        prompts = prompts[: args.n_samples]

    print(f"{len(prompts)} prompts loaded")

    with mp.Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap(
                    functools.partial(
                        process_prompt,
                        model_cfg=model_cfg,
                        save_path=save_path,
                        force_overwrite=args.force,
                    ),
                    prompts,
                ),
                total=len(prompts),
                desc="Making figures",
            )
        )

    with SpinnerContext(message=f"updating jsonl metadata for models and functions", **SPINNER_KWARGS):
        generate_models_jsonl(save_path)
        generate_functions_jsonl(save_path)


if __name__ == "__main__":
    main()
