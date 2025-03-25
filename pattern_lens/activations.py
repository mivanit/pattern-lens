"""computing and saving activations given a model and prompts

# Usage:

from the command line:

```bash
python -m pattern_lens.activations --model <model_name> --prompts <prompts_path> --save-path <save_path> --min-chars <min_chars> --max-chars <max_chars> --n-samples <n_samples>
```

from a script:

```python
from pattern_lens.activations import activations_main
activations_main(
	model_name="gpt2",
	save_path="demo/"
	prompts_path="data/pile_1k.jsonl",
)
```

"""

import argparse
import functools
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Literal, overload

import numpy as np
import torch
import tqdm
from jaxtyping import Float
from muutils.json_serialize import json_serialize
from muutils.misc.numerical import shorten_numerical_to_str

# custom utils
from muutils.spinner import SpinnerContext
from transformer_lens import (  # type: ignore[import-untyped]
	ActivationCache,
	HookedTransformer,
	HookedTransformerConfig,
)

# pattern_lens
from pattern_lens.consts import (
	ATTN_PATTERN_REGEX,
	DATA_DIR,
	DIVIDER_S1,
	DIVIDER_S2,
	SPINNER_KWARGS,
	ActivationCacheNp,
	ReturnCache,
)
from pattern_lens.indexes import (
	generate_models_jsonl,
	generate_prompts_jsonl,
	write_html_index,
)
from pattern_lens.load_activations import (
	ActivationsMissingError,
	augment_prompt_with_hash,
	load_activations,
)
from pattern_lens.prompts import load_text_data


# return nothing, but `stack_heads` still affects how we save the activations
@overload
def compute_activations(
	prompt: dict,
	model: HookedTransformer | None = None,
	save_path: Path = Path(DATA_DIR),
	names_filter: Callable[[str], bool] | re.Pattern = ATTN_PATTERN_REGEX,
	return_cache: Literal[None] = None,
	stack_heads: bool = False,
) -> tuple[Path, None]: ...
# return stacked heads in numpy or torch form
@overload
def compute_activations(
	prompt: dict,
	model: HookedTransformer | None = None,
	save_path: Path = Path(DATA_DIR),
	names_filter: Callable[[str], bool] | re.Pattern = ATTN_PATTERN_REGEX,
	return_cache: Literal["torch"] = "torch",
	stack_heads: Literal[True] = True,
) -> tuple[Path, Float[torch.Tensor, "n_layers n_heads n_ctx n_ctx"]]: ...
@overload
def compute_activations(
	prompt: dict,
	model: HookedTransformer | None = None,
	save_path: Path = Path(DATA_DIR),
	names_filter: Callable[[str], bool] | re.Pattern = ATTN_PATTERN_REGEX,
	return_cache: Literal["numpy"] = "numpy",
	stack_heads: Literal[True] = True,
) -> tuple[Path, Float[np.ndarray, "n_layers n_heads n_ctx n_ctx"]]: ...
# return dicts in numpy or torch form
@overload
def compute_activations(
	prompt: dict,
	model: HookedTransformer | None = None,
	save_path: Path = Path(DATA_DIR),
	names_filter: Callable[[str], bool] | re.Pattern = ATTN_PATTERN_REGEX,
	return_cache: Literal["numpy"] = "numpy",
	stack_heads: Literal[False] = False,
) -> tuple[Path, ActivationCacheNp]: ...
@overload
def compute_activations(
	prompt: dict,
	model: HookedTransformer | None = None,
	save_path: Path = Path(DATA_DIR),
	names_filter: Callable[[str], bool] | re.Pattern = ATTN_PATTERN_REGEX,
	return_cache: Literal["torch"] = "torch",
	stack_heads: Literal[False] = False,
) -> tuple[Path, ActivationCache]: ...
# actual function body
def compute_activations(  # noqa: PLR0915
	prompt: dict,
	model: HookedTransformer | None = None,
	save_path: Path = Path(DATA_DIR),
	names_filter: Callable[[str], bool] | re.Pattern = ATTN_PATTERN_REGEX,
	return_cache: ReturnCache = "torch",
	stack_heads: bool = False,
) -> tuple[
	Path,
	ActivationCacheNp
	| ActivationCache
	| Float[np.ndarray, "n_layers n_heads n_ctx n_ctx"]
	| Float[torch.Tensor, "n_layers n_heads n_ctx n_ctx"]
	| None,
]:
	"""get activations for a given model and prompt, possibly from a cache

	if from a cache, prompt_meta must be passed and contain the prompt hash

	# Parameters:
	- `prompt : dict | None`
		(defaults to `None`)
	- `model : HookedTransformer`
	- `save_path : Path`
		(defaults to `Path(DATA_DIR)`)
	- `names_filter : Callable[[str], bool]|re.Pattern`
		a filter for the names of the activations to return. if an `re.Pattern`, will use `lambda key: names_filter.match(key) is not None`
		(defaults to `ATTN_PATTERN_REGEX`)
	- `return_cache : Literal[None, "numpy", "torch"]`
		will return `None` as the second element if `None`, otherwise will return the cache in the specified tensor format. `stack_heads` still affects whether it will be a dict (False) or a single tensor (True)
		(defaults to `None`)
	- `stack_heads : bool`
		whether the heads should be stacked in the output. this causes a number of changes:
	- `npy` file with a single `(n_layers, n_heads, n_ctx, n_ctx)` tensor saved for each prompt instead of `npz` file with dict by layer
	- `cache` will be a single `(n_layers, n_heads, n_ctx, n_ctx)` tensor instead of a dict by layer if `return_cache` is `True`
		will assert that everything in the activation cache is only attention patterns, and is all of the attention patterns. raises an exception if not.

	# Returns:
	```
	tuple[
		Path,
		Union[
			None,
			ActivationCacheNp, ActivationCache,
			Float[np.ndarray, "n_layers n_heads n_ctx n_ctx"], Float[torch.Tensor, "n_layers n_heads n_ctx n_ctx"],
		]
	]
	```
	"""
	# check inputs
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
		),
	)

	# save metadata
	prompt_dir: Path = save_path / model.cfg.model_name / "prompts" / prompt["hash"]
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
	cache_torch: ActivationCache
	with torch.no_grad():
		model.eval()
		# TODO: batching?
		_, cache_torch = model.run_with_cache(
			prompt_str,
			names_filter=names_filter_fn,
			return_type=None,
		)

	activations_path: Path
	# saving and returning
	if stack_heads:
		n_layers: int = model.cfg.n_layers
		key_pattern: str = "blocks.{i}.attn.hook_pattern"
		# NOTE: this only works for stacking heads at the moment
		# activations_specifier: str = key_pattern.format(i=f'0-{n_layers}')
		activations_specifier: str = key_pattern.format(i="-")
		activations_path = prompt_dir / f"activations-{activations_specifier}.npy"

		# check the keys are only attention heads
		head_keys: list[str] = [key_pattern.format(i=i) for i in range(n_layers)]
		cache_torch_keys_set: set[str] = set(cache_torch.keys())
		assert cache_torch_keys_set == set(head_keys), (
			f"unexpected keys!\n{set(head_keys).symmetric_difference(cache_torch_keys_set) = }\n{cache_torch_keys_set} != {set(head_keys)}"
		)

		# stack heads
		patterns_stacked: Float[torch.Tensor, "n_layers n_heads n_ctx n_ctx"] = (
			torch.stack([cache_torch[k] for k in head_keys], dim=1)
		)
		# check shape
		pattern_shape_no_ctx: tuple[int, ...] = tuple(patterns_stacked.shape[:3])
		assert pattern_shape_no_ctx == (1, n_layers, model.cfg.n_heads), (
			f"unexpected shape: {patterns_stacked.shape[:3] = } ({pattern_shape_no_ctx = }), expected {(1, n_layers, model.cfg.n_heads) = }"
		)

		patterns_stacked_np: Float[np.ndarray, "n_layers n_heads n_ctx n_ctx"] = (
			patterns_stacked.cpu().numpy()
		)

		# save
		np.save(activations_path, patterns_stacked_np)

		# return
		match return_cache:
			case "numpy":
				return activations_path, patterns_stacked_np
			case "torch":
				return activations_path, patterns_stacked
			case None:
				return activations_path, None
			case _:
				msg = f"invalid return_cache: {return_cache = }"
				raise ValueError(msg)
	else:
		activations_path = prompt_dir / "activations.npz"

		# save
		cache_np: ActivationCacheNp = {
			k: v.detach().cpu().numpy() for k, v in cache_torch.items()
		}

		np.savez_compressed(
			activations_path,
			**cache_np,
		)

		# return
		match return_cache:
			case "numpy":
				return activations_path, cache_np
			case "torch":
				return activations_path, cache_torch
			case None:
				return activations_path, None
			case _:
				msg = f"invalid return_cache: {return_cache = }"
				raise ValueError(msg)


@overload
def get_activations(
	prompt: dict,
	model: HookedTransformer | str,
	save_path: Path = Path(DATA_DIR),
	allow_disk_cache: bool = True,
	return_cache: Literal[None] = None,
) -> tuple[Path, None]: ...
@overload
def get_activations(
	prompt: dict,
	model: HookedTransformer | str,
	save_path: Path = Path(DATA_DIR),
	allow_disk_cache: bool = True,
	return_cache: Literal["torch"] = "torch",
) -> tuple[Path, ActivationCache]: ...
@overload
def get_activations(
	prompt: dict,
	model: HookedTransformer | str,
	save_path: Path = Path(DATA_DIR),
	allow_disk_cache: bool = True,
	return_cache: Literal["numpy"] = "numpy",
) -> tuple[Path, ActivationCacheNp]: ...
def get_activations(
	prompt: dict,
	model: HookedTransformer | str,
	save_path: Path = Path(DATA_DIR),
	allow_disk_cache: bool = True,
	return_cache: ReturnCache = "numpy",
) -> tuple[Path, ActivationCacheNp | ActivationCache | None]:
	"""given a prompt and a model, save or load activations

	# Parameters:
	- `prompt : dict`
		expected to contain the 'text' key
	- `model : HookedTransformer | str`
		either a `HookedTransformer` or a string model name, to be loaded with `HookedTransformer.from_pretrained`
	- `save_path : Path`
		path to save the activations to (and load from)
		(defaults to `Path(DATA_DIR)`)
	- `allow_disk_cache : bool`
		whether to allow loading from disk cache
		(defaults to `True`)
	- `return_cache : Literal[None, "numpy", "torch"]`
		whether to return the cache, and in what format
		(defaults to `"numpy"`)

	# Returns:
	- `tuple[Path, ActivationCacheNp | ActivationCache | None]`
		the path to the activations and the cache if `return_cache is not None`

	"""
	# add hash to prompt
	augment_prompt_with_hash(prompt)

	# get the model
	model_name: str = (
		model.cfg.model_name if isinstance(model, HookedTransformer) else model
	)

	# from cache
	if allow_disk_cache:
		try:
			path, cache = load_activations(
				model_name=model_name,
				prompt=prompt,
				save_path=save_path,
			)
			if return_cache:
				return path, cache
			else:
				# TODO: this basically does nothing, since we load the activations and then immediately get rid of them.
				# maybe refactor this so that load_activations can take a parameter to simply assert that the cache exists?
				# this will let us avoid loading it, which slows things down
				return path, None
		except ActivationsMissingError:
			pass

	# compute them
	if isinstance(model, str):
		model = HookedTransformer.from_pretrained(model_name)

	return compute_activations(
		prompt=prompt,
		model=model,
		save_path=save_path,
		return_cache=return_cache,
	)


DEFAULT_DEVICE: torch.device = torch.device(
	"cuda" if torch.cuda.is_available() else "cpu",
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
	stacked_heads: bool = False,
	device: str | torch.device = DEFAULT_DEVICE,
) -> None:
	"""main function for computing activations

	# Parameters:
	- `model_name : str`
		name of a model to load with `HookedTransformer.from_pretrained`
	- `save_path : str`
		path to save the activations to
	- `prompts_path : str`
		path to the prompts file
	- `raw_prompts : bool`
		whether the prompts are raw, not filtered by length. `load_text_data` will be called if `True`, otherwise just load the "text" field from each line in `prompts_path`
	- `min_chars : int`
		minimum number of characters for a prompt
	- `max_chars : int`
		maximum number of characters for a prompt
	- `force : bool`
		whether to overwrite existing files
	- `n_samples : int`
		maximum number of samples to process
	- `no_index_html : bool`
		whether to write an index.html file
	- `shuffle : bool`
		whether to shuffle the prompts
		(defaults to `False`)
	- `stacked_heads : bool`
		whether	to stack the heads in the output tensor. will save as `.npy` instead of `.npz` if `True`
		(defaults to `False`)
	- `device : str | torch.device`
		the device to use. if a string, will be passed to `torch.device`
	"""
	# figure out the device to use
	device_: torch.device
	if isinstance(device, torch.device):
		device_ = device
	elif isinstance(device, str):
		device_ = torch.device(device)
	else:
		msg = f"invalid device: {device}"
		raise TypeError(msg)

	print(f"using device: {device_}")

	with SpinnerContext(message="loading model", **SPINNER_KWARGS):
		model: HookedTransformer = HookedTransformer.from_pretrained(
			model_name,
			device=device_,
		)
		model.model_name = model_name
		model.cfg.model_name = model_name
		n_params: int = sum(p.numel() for p in model.parameters())
	print(
		f"loaded {model_name} with {shorten_numerical_to_str(n_params)} ({n_params}) parameters",
	)
	print(f"\tmodel devices: { {p.device for p in model.parameters()} }")

	save_path_p: Path = Path(save_path)
	save_path_p.mkdir(parents=True, exist_ok=True)
	model_path: Path = save_path_p / model_name
	with SpinnerContext(
		message=f"saving model info to {model_path.as_posix()}",
		**SPINNER_KWARGS,
	):
		model_cfg: HookedTransformerConfig
		model_cfg = model.cfg
		model_path.mkdir(parents=True, exist_ok=True)
		with open(model_path / "model_cfg.json", "w") as f:
			json.dump(json_serialize(asdict(model_cfg)), f)

	# load prompts
	with SpinnerContext(
		message=f"loading prompts from {prompts_path = }",
		**SPINNER_KWARGS,
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

	# write index.html
	with SpinnerContext(message="writing index.html", **SPINNER_KWARGS):
		if not no_index_html:
			write_html_index(save_path_p)

	# TODO: not implemented yet
	if stacked_heads:
		raise NotImplementedError("stacked_heads not implemented yet")

	# get activations
	list(
		tqdm.tqdm(
			map(
				functools.partial(
					get_activations,
					model=model,
					save_path=save_path_p,
					allow_disk_cache=not force,
					return_cache=None,
					# stacked_heads=stacked_heads,
				),
				prompts,
			),
			total=len(prompts),
			desc="Computing activations",
			unit="prompt",
		),
	)

	with SpinnerContext(
		message="updating jsonl metadata for models and prompts",
		**SPINNER_KWARGS,
	):
		generate_models_jsonl(save_path_p)
		generate_prompts_jsonl(save_path_p / model_name)


def main() -> None:
	"generate attention pattern activations for a model and prompts"
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

		# stack heads
		arg_parser.add_argument(
			"--stacked-heads",
			action="store_true",
			help="If passed, will stack the heads in the output tensor",
		)

		# device
		arg_parser.add_argument(
			"--device",
			type=str,
			required=False,
			help="The device to use for the model",
			default="cuda" if torch.cuda.is_available() else "cpu",
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

		activations_main(
			model_name=model,
			save_path=args.save_path,
			prompts_path=args.prompts,
			raw_prompts=args.raw_prompts,
			min_chars=args.min_chars,
			max_chars=args.max_chars,
			force=args.force,
			n_samples=args.n_samples,
			no_index_html=args.no_index_html,
			shuffle=args.shuffle,
			stacked_heads=args.stacked_heads,
			device=args.device,
		)
		del model

	print(DIVIDER_S1)


if __name__ == "__main__":
	main()
