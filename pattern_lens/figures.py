"""code for generating figures from attention patterns, using the functions decorated with `register_attn_figure_func`"""

import argparse
import functools
import itertools
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from jaxtyping import Float

# custom utils
from muutils.json_serialize import json_serialize
from muutils.parallel import run_maybe_parallel
from muutils.spinner import SpinnerContext

# pattern_lens
from pattern_lens.attn_figure_funcs import ATTENTION_MATRIX_FIGURE_FUNCS
from pattern_lens.consts import (
	DATA_DIR,
	DIVIDER_S1,
	DIVIDER_S2,
	SPINNER_KWARGS,
	ActivationCacheNp,
	AttentionMatrix,
)
from pattern_lens.indexes import (
	generate_functions_jsonl,
	generate_models_jsonl,
	generate_prompts_jsonl,
)
from pattern_lens.load_activations import load_activations


class HTConfigMock:
	"""Mock of `transformer_lens.HookedTransformerConfig` for type hinting and loading config json

	can be initialized with any kwargs, and will update its `__dict__` with them. does, however, require the following attributes:
	- `n_layers: int`
	- `n_heads: int`
	- `model_name: str`

	we do this to avoid having to import `torch` and `transformer_lens`, since this would have to be done for each process in the parallelization and probably slows things down significantly
	"""

	def __init__(self, **kwargs) -> None:
		self.n_layers: int
		self.n_heads: int
		self.model_name: str
		self.__dict__.update(kwargs)

	def serialize(self):
		"""serialize the config to json. values which aren't serializable will be converted via `muutils.json_serialize.json_serialize`"""
		return json_serialize(self.__dict__)

	@classmethod
	def load(cls, data: dict):
		"try to load a config from a dict, using the `__init__` method"
		return cls(**data)


def process_single_head(
	layer_idx: int,
	head_idx: int,
	attn_pattern: AttentionMatrix,
	save_dir: Path,
	force_overwrite: bool = False,
) -> dict[str, bool | Exception]:
	"""process a single head's attention pattern, running all the functions in `ATTENTION_MATRIX_FIGURE_FUNCS` on the attention pattern

	> [gotcha:] if `force_overwrite` is `False`, and we used a multi-figure function,
	> it will skip all figures for that function if any are already saved
	> and it assumes a format of `{func_name}.{figure_name}.{fmt}` for the saved figures

	# Parameters:
	- `layer_idx : int`
	- `head_idx : int`
	- `attn_pattern : AttentionMatrix`
		attention pattern for the head
	- `save_dir : Path`
		directory to save the figures to
	- `force_overwrite : bool`
		whether to overwrite existing figures. if `False`, will skip any functions which have already saved a figure
		(defaults to `False`)

	# Returns:
	- `dict[str, bool | Exception]`
		a dictionary of the status of each function, with the function name as the key and the status as the value
	"""
	funcs_status: dict[str, bool | Exception] = dict()

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
				f"Error in {func.__name__} for L{layer_idx}H{head_idx}: {e!s}", stacklevel=2,
			)
			funcs_status[func_name] = e

	return funcs_status


def compute_and_save_figures(
	model_cfg: "HookedTransformerConfig|HTConfigMock",  # type: ignore[name-defined] # noqa: F821
	activations_path: Path,
	cache: ActivationCacheNp | Float[np.ndarray, "n_layers n_heads n_ctx n_ctx"],
	save_path: Path = Path(DATA_DIR),
	force_overwrite: bool = False,
	track_results: bool = False,
) -> None:
	"""compute and save figures for all heads in the model, using the functions in `ATTENTION_MATRIX_FIGURE_FUNCS`

	# Parameters:
	- `model_cfg : HookedTransformerConfig|HTConfigMock`
	- `cache : ActivationCacheNp | Float[np.ndarray, "n_layers n_heads n_ctx n_ctx"]`
	- `save_path : Path`
		(defaults to `Path(DATA_DIR)`)
	- `force_overwrite : bool`
		force overwrite of existing figures. if `False`, will skip any functions which have already saved a figure
		(defaults to `False`)
	- `track_results : bool`
		whether to track the results of each function for each head. Isn't used for anything yet, but this is a TODO
		(defaults to `False`)
	"""
	prompt_dir: Path = activations_path.parent

	if track_results:
		results: defaultdict[
			str,  # func name
			dict[
				tuple[int, int],  # layer, head
				bool | Exception,  # success or exception
			],
		] = defaultdict(dict)

	for layer_idx, head_idx in itertools.product(
		range(model_cfg.n_layers),
		range(model_cfg.n_heads),
	):
		attn_pattern: AttentionMatrix
		if isinstance(cache, dict):
			attn_pattern = cache[f"blocks.{layer_idx}.attn.hook_pattern"][0, head_idx]
		elif isinstance(cache, np.ndarray):
			attn_pattern = cache[layer_idx, head_idx]
		else:
			msg = f"cache must be a dict or np.ndarray, not {type(cache) = }\n{cache = }"
			raise TypeError(
				msg,
			)

		save_dir: Path = prompt_dir / f"L{layer_idx}" / f"H{head_idx}"
		save_dir.mkdir(parents=True, exist_ok=True)
		head_res: dict[str, bool | Exception] = process_single_head(
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
	model_cfg: "HookedTransformerConfig|HTConfigMock",  # type: ignore[name-defined] # noqa: F821
	save_path: Path,
	force_overwrite: bool = False,
) -> None:
	"""process a single prompt, loading the activations and computing and saving the figures

	basically just calls `load_activations` and then `compute_and_save_figures`

	# Parameters:
	- `prompt : dict`
	- `model_cfg : HookedTransformerConfig|HTConfigMock`
	- `force_overwrite : bool`
		(defaults to `False`)
	"""
	activations_path: Path
	cache: ActivationCacheNp | Float[np.ndarray, "n_layers n_heads n_ctx n_ctx"]
	activations_path, cache = load_activations(
		model_name=model_cfg.model_name,
		prompt=prompt,
		save_path=save_path,
		return_fmt="numpy",
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
	"""main function for generating figures from attention patterns, using the functions in `ATTENTION_MATRIX_FIGURE_FUNCS`

	# Parameters:
	- `model_name : str`
		model name to use, used for loading the model config, prompts, activations, and saving the figures
	- `save_path : str`
		base path to look in
	- `n_samples : int`
		max number of samples to process
	- `force : bool`
		force overwrite of existing figures. if `False`, will skip any functions which have already saved a figure
	- `parallel : bool | int`
		whether to run in parallel. if `True`, will use all available cores. if `False`, will run in serial. if an int, will try to use that many cores
		(defaults to `True`)
	"""
	with SpinnerContext(message="setting up paths", **SPINNER_KWARGS):
		# save model info or check if it exists
		save_path_p: Path = Path(save_path)
		model_path: Path = save_path_p / model_name
		with open(model_path / "model_cfg.json", "r") as f:
			model_cfg = HTConfigMock.load(json.load(f))

	with SpinnerContext(message="loading prompts", **SPINNER_KWARGS):
		# load prompts
		with open(model_path / "prompts.jsonl", "r") as f:
			prompts: list[dict] = [json.loads(line) for line in f.readlines()]
		# truncate to n_samples
		prompts = prompts[:n_samples]

	print(f"{len(prompts)} prompts loaded")

	print(f"{len(ATTENTION_MATRIX_FIGURE_FUNCS)} figure functions loaded")
	print("\t" + ", ".join([func.__name__ for func in ATTENTION_MATRIX_FIGURE_FUNCS]))

	list(
		run_maybe_parallel(
			func=functools.partial(
				process_prompt,
				model_cfg=model_cfg,
				save_path=save_path_p,
				force_overwrite=force,
			),
			iterable=prompts,
			parallel=parallel,
			pbar="tqdm",
			pbar_kwargs=dict(
				desc="Making figures",
				unit="prompt",
			),
		),
	)

	with SpinnerContext(
		message="updating jsonl metadata for models and functions",
		**SPINNER_KWARGS,
	):
		generate_models_jsonl(save_path_p)
		generate_functions_jsonl(save_path_p)


def main() -> None:
	print(DIVIDER_S1)
	with SpinnerContext(message="parsing args", **SPINNER_KWARGS):
		arg_parser: argparse.ArgumentParser = argparse.ArgumentParser()
		# input and output
		arg_parser.add_argument(
			"--model",
			"-m",
			type=str,
			required=True,
			help="The model name(s) to use. comma separated with no whitespace if multiple",
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

	models: list[str]
	if "," in args.model:
		models = args.model.split(",")
	else:
		models = [args.model]

	n_models: int = len(models)
	for idx, model in enumerate(models):
		print(DIVIDER_S2)
		print(f"processing model {idx + 1} / {n_models}: {model}")
		print(DIVIDER_S2)
		figures_main(
			model_name=model,
			save_path=args.save_path,
			n_samples=args.n_samples,
			force=args.force,
		)

	print(DIVIDER_S1)


if __name__ == "__main__":
	main()
