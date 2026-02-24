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
import gc
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
	activations_exist,
	augment_prompt_with_hash,
	load_activations,
)
from pattern_lens.prompts import load_text_data


def _rel_path(p: Path) -> str:
	"""Return path relative to cwd if possible, otherwise absolute."""
	try:
		return p.relative_to(Path.cwd()).as_posix()
	except ValueError:
		return p.as_posix()


# return nothing, but `stack_heads` still affects how we save the activations
@overload
def compute_activations(
	prompt: dict,
	model: HookedTransformer | None = None,
	save_path: Path = Path(DATA_DIR),
	names_filter: Callable[[str], bool] | re.Pattern = ATTN_PATTERN_REGEX,
	return_cache: None = None,
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
	"""compute activations for a single prompt and save to disk

	always runs a forward pass -- does NOT load from disk cache.
	for cache-aware loading, use `get_activations` which tries disk first.

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
	assert model.tokenizer is not None
	prompt_tokenized: list[str] = prompt.get(
		"tokens",
		model.tokenizer.tokenize(prompt_str),
	)
	# n_tokens counts subword tokens (no BOS); attention patterns include BOS
	# so have dim n_tokens+1. see also compute_activations_batched Phase B.
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
	# NOTE: no padding_side kwarg here -- it's only meaningful for multi-sequence
	# batches where padding is needed. single-string input has no padding.
	# see compute_activations_batched for the batched path that passes padding_side="right".
	cache_torch: ActivationCache
	with torch.no_grad():
		model.eval()
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
			**cache_np,  # type: ignore[arg-type]
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


def compute_activations_batched(
	prompts: list[dict],
	model: HookedTransformer,
	save_path: Path = Path(DATA_DIR),
	names_filter: Callable[[str], bool] | re.Pattern = ATTN_PATTERN_REGEX,
	seq_lens: list[int] | None = None,
) -> list[Path]:
	"""compute and save activations for a batch of prompts in a single forward pass

	Batched companion to `compute_activations` -- instead of one forward pass per
	prompt, this runs a single `model.run_with_cache(list_of_strings)` call for the
	whole batch. TransformerLens tokenizes and right-pads automatically. Each prompt's
	attention patterns are then trimmed to their actual (unpadded) size and saved
	individually, producing files identical to the single-prompt path.

	Does not support `stack_heads` or `return_cache` -- this function is intended for
	the bulk processing path in `activations_main`, not for interactive use. Use
	`compute_activations` directly for single-prompt use cases that need those features.

	## Why right-padding makes trimming correct without an explicit attention mask

	With right-padding, pad tokens sit at positions seq_len, seq_len+1, ...,
	max_seq_len-1 (higher than any real token). The causal attention mask prevents
	position i from attending to any j > i. So for real tokens at positions
	0..seq_len-1, they can only attend to 0..i -- all real tokens. The softmax is computed over the same set of positions
	as in single-prompt inference, producing identical attention patterns.

	We explicitly pass `padding_side="right"` to `run_with_cache` to guarantee this
	regardless of the model's default padding side.

	# Parameters:
	- `prompts : list[dict]`
		each prompt must contain 'text' and 'hash' keys. call
		`augment_prompt_with_hash` on each prompt before passing them here.
	- `model : HookedTransformer`
		the model to compute activations with
	- `save_path : Path`
		path to save the activations to
		(defaults to `Path(DATA_DIR)`)
	- `names_filter : Callable[[str], bool] | re.Pattern`
		filter for which activations to save. must only match activations with
		4D shape `[batch, n_heads, seq, seq]` (e.g. attention patterns).
		non-attention activations will cause incorrect trimming.
		(defaults to `ATTN_PATTERN_REGEX`)
	- `seq_lens : list[int] | None`
		pre-computed model sequence lengths per prompt (from `model.to_tokens`).
		if `None`, will be computed internally. pass this to avoid redundant
		tokenization when lengths are already known (e.g. from length-sorting).
		**important**: these must be from `model.to_tokens()` (includes BOS),
		NOT from `model.tokenizer.tokenize()` (excludes BOS).
		(defaults to `None`)

	# Returns:
	- `list[Path]`
		paths to the saved activations files, one per prompt

	# Modifies:
	each prompt dict in `prompts` -- adds/overwrites `n_tokens` and `tokens` keys
	with tokenization metadata (same mutation as `compute_activations`).
	"""
	assert model is not None, "model must be passed"
	assert len(prompts) > 0, "prompts must not be empty"
	assert "text" in prompts[0], f"prompt must contain 'text' key: {prompts[0].keys()}"
	assert "hash" in prompts[0], (
		f"prompt must contain 'hash' key (call augment_prompt_with_hash first): {prompts[0].keys()}"
	)

	# --- Phase A: get actual model sequence lengths ---
	# model.to_tokens() includes BOS if applicable, matching the attention pattern dims
	# model.tokenizer.tokenize() gives subword strings WITHOUT BOS, used for metadata
	# these differ by 1 when BOS is prepended -- using the wrong one for trimming
	# would silently truncate or include garbage
	if seq_lens is None:
		seq_lens = [model.to_tokens(p["text"]).shape[1] for p in prompts]
	assert len(seq_lens) == len(prompts), (
		f"seq_lens length mismatch: {len(seq_lens)} != {len(prompts)}"
	)

	# --- Phase B: save prompt metadata (mirrors compute_activations's metadata logic) ---
	assert model.tokenizer is not None
	for p in prompts:
		prompt_str: str = p["text"]
		prompt_tokenized: list[str] = p.get(
			"tokens",
			model.tokenizer.tokenize(prompt_str),
		)
		# n_tokens counts subword tokens (no BOS); attention patterns include BOS so have dim n_tokens+1
		p.update(
			dict(
				n_tokens=len(prompt_tokenized),
				tokens=prompt_tokenized,
			),
		)
		prompt_dir: Path = save_path / model.cfg.model_name / "prompts" / p["hash"]
		prompt_dir.mkdir(parents=True, exist_ok=True)
		with open(prompt_dir / "prompt.json", "w") as f:
			json.dump(p, f)

	# --- Phase C: batched forward pass ---
	names_filter_fn: Callable[[str], bool]
	if isinstance(names_filter, re.Pattern):
		names_filter_fn = lambda key: names_filter.match(key) is not None  # noqa: E731
	else:
		names_filter_fn = names_filter

	texts: list[str] = [p["text"] for p in prompts]
	cache_torch: ActivationCache
	with torch.no_grad():
		model.eval()
		_, cache_torch = model.run_with_cache(
			texts,
			names_filter=names_filter_fn,
			return_type=None,
			padding_side="right",
		)

	# --- Phase D: split, trim padding, and save per-prompt ---
	# For each prompt i with actual sequence length seq_len_i:
	#   v[i : i+1, :, :seq_len_i, :seq_len_i]
	#     ^^^^^^^                               i:i+1 not i -- keeps batch dim [1,...] for
	#                                           format compatibility with compute_activations
	#              ^^                           all attention heads
	#                  ^^^^^^^^^^  ^^^^^^^^^^   trim both query and key dims to actual length,
	#                                           discarding meaningless padding positions
	paths: list[Path] = []
	for i, (prompt, seq_len) in enumerate(zip(prompts, seq_lens, strict=True)):
		prompt_dir = save_path / model.cfg.model_name / "prompts" / prompt["hash"]
		activations_path: Path = prompt_dir / "activations.npz"
		cache_np: ActivationCacheNp = {}
		for k, v in cache_torch.items():
			assert v.ndim == 4, (  # noqa: PLR2004
				f"expected 4D attention pattern tensor for {k!r}, "
				f"got shape {v.shape}. names_filter must only match "
				f"attention pattern activations [batch, n_heads, seq, seq]"
			)
			cache_np[k] = v[i : i + 1, :, :seq_len, :seq_len].detach().cpu().numpy()

		np.savez_compressed(
			activations_path,
			**cache_np,  # type: ignore[arg-type]
		)
		paths.append(activations_path)

	return paths


@overload
def get_activations(
	prompt: dict,
	model: HookedTransformer | str,
	save_path: Path = Path(DATA_DIR),
	allow_disk_cache: bool = True,
	return_cache: None = None,
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
		if return_cache is None:
			# fast path: check file existence without loading data into memory.
			# activations_exist just calls .exists() on two paths, whereas
			# load_activations would decompress the full .npz into numpy arrays
			# only for us to discard them immediately.
			if activations_exist(model_name, prompt, save_path):
				prompt_dir: Path = save_path / model_name / "prompts" / prompt["hash"]
				return prompt_dir / "activations.npz", None
		else:
			try:
				path, cache = load_activations(
					model_name=model_name,
					prompt=prompt,
					save_path=save_path,
				)
			except ActivationsMissingError:
				pass
			else:
				return path, cache

	# compute them
	if isinstance(model, str):
		model = HookedTransformer.from_pretrained(model_name)

	return compute_activations(  # type: ignore[return-value]
		prompt=prompt,
		model=model,
		save_path=save_path,
		return_cache=return_cache,
	)


DEFAULT_DEVICE: torch.device = torch.device(
	"cuda" if torch.cuda.is_available() else "cpu",
)


def activations_main(  # noqa: C901, PLR0912, PLR0915
	model_name: str,
	save_path: str | Path,
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
	batch_size: int = 32,
) -> None:
	"""main function for computing activations

	# Parameters:
	- `model_name : str`
		name of a model to load with `HookedTransformer.from_pretrained`
	- `save_path : str | Path`
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
	- `batch_size : int`
		number of prompts per forward pass. prompts are sorted by token length
		(longest first) and grouped so that similar-length prompts share a batch,
		minimizing padding waste. use `batch_size=1` for one prompt per forward
		pass (largely equivalent to the old sequential behavior, but note: prompts
		are still sorted by length and cache checking uses file-existence only,
		unlike the old path which processed prompts in order and validated cache
		contents via `load_activations`).
		the single-prompt functions `compute_activations` and `get_activations`
		are still available for programmatic use outside of `activations_main`.
		(defaults to `32`)
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
		model.model_name = model_name  # type: ignore[unresolved-attribute]
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
		message=f"saving model info to {_rel_path(model_path)}",
		**SPINNER_KWARGS,
	):
		model_cfg: HookedTransformerConfig
		model_cfg = model.cfg
		model_path.mkdir(parents=True, exist_ok=True)
		with open(model_path / "model_cfg.json", "w") as f:
			json.dump(json_serialize(asdict(model_cfg)), f)

	# load prompts
	with SpinnerContext(
		message=f"loading prompts from {Path(prompts_path).as_posix()}",
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

	print(f"  {len(prompts)} prompts loaded")

	# write index.html
	with SpinnerContext(
		message=f"writing {_rel_path(save_path_p / 'index.html')}",
		**SPINNER_KWARGS,
	):
		if not no_index_html:
			write_html_index(save_path_p)

	# TODO: not implemented yet
	if stacked_heads:
		raise NotImplementedError("stacked_heads not implemented yet")

	# augment all prompts with hashes
	for prompt in prompts:
		augment_prompt_with_hash(prompt)

	# filter out cached prompts
	if not force:
		uncached: list[dict] = [
			p for p in prompts if not activations_exist(model_name, p, save_path_p)
		]
		n_cached: int = len(prompts) - len(uncached)
		if n_cached > 0:
			print(f"  {n_cached} prompts already cached, {len(uncached)} to compute")
	else:
		uncached = list(prompts)

	if uncached:
		# sort by token length descending so that:
		# 1. the longest (slowest, most memory-hungry) batches run first --
		#    OOM errors surface immediately rather than after all the cheap work,
		#    and tqdm's ETA stabilizes early for better progress estimation
		# 2. similar-length prompts are grouped together, minimizing padding waste
		#
		# pre-tokenization is a separate step from compute_activations_batched because
		# we need token lengths *before* batching to sort and group. the resulting
		# seq_lens are then passed through so compute_activations_batched can skip
		# re-tokenizing each prompt internally.
		with SpinnerContext(
			message="pre-tokenizing prompts for length sorting",
			**SPINNER_KWARGS,
		):
			uncached_with_lens: list[tuple[dict, int]] = [
				(p, model.to_tokens(p["text"]).shape[1]) for p in uncached
			]
			uncached_with_lens.sort(key=lambda x: x[1], reverse=True)
			sorted_uncached: list[dict] = [p for p, _ in uncached_with_lens]
			sorted_seq_lens: list[int] = [sl for _, sl in uncached_with_lens]

		# process in batches
		n_prompts: int = len(sorted_uncached)
		with tqdm.tqdm(
			total=n_prompts,
			desc="Computing activations",
			unit="prompt",
		) as pbar:
			for batch_start in range(0, n_prompts, batch_size):
				batch_end: int = min(batch_start + batch_size, n_prompts)
				batch: list[dict] = sorted_uncached[batch_start:batch_end]
				batch_seq_lens: list[int] = sorted_seq_lens[batch_start:batch_end]
				pbar.set_postfix(
					n_ctx=batch_seq_lens[0],
				)  # longest in batch (sorted descending)
				compute_activations_batched(
					prompts=batch,
					model=model,
					save_path=save_path_p,
					seq_lens=batch_seq_lens,
				)
				pbar.update(len(batch))
	else:
		print("  all prompts cached, nothing to compute")

	with SpinnerContext(
		message="updating jsonl metadata for models and prompts",
		**SPINNER_KWARGS,
	):
		generate_models_jsonl(save_path_p)
		generate_prompts_jsonl(save_path_p / model_name)

	# free model memory before returning
	del model
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


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

		# batch size
		arg_parser.add_argument(
			"--batch-size",
			"-b",
			type=int,
			required=False,
			help="Batch size for computing activations (number of prompts per forward pass)",
			default=32,
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
			batch_size=args.batch_size,
		)
		# defense-in-depth: collect any remaining torch objects from activations_main
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	print(DIVIDER_S1)


if __name__ == "__main__":
	main()
