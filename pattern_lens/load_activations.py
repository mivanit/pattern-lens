"loading activations from .npz on disk. implements some custom Exception classes"

import base64
import hashlib
import json
from pathlib import Path
from typing import Literal, overload

import numpy as np

from pattern_lens.consts import ReturnCache


class GetActivationsError(ValueError):
	"""base class for errors in getting activations"""

	pass


class ActivationsMissingError(GetActivationsError, FileNotFoundError):
	"""error for missing activations -- can't find the activations file"""

	pass


class ActivationsMismatchError(GetActivationsError):
	"""error for mismatched activations -- the prompt text or hash do not match

	raised by `compare_prompt_to_loaded`
	"""

	pass


class InvalidPromptError(GetActivationsError):
	"""error for invalid prompt -- the prompt does not have fields "hash" or "text"

	raised by `augment_prompt_with_hash`
	"""

	pass


def compare_prompt_to_loaded(prompt: dict, prompt_loaded: dict) -> None:
	"""compare a prompt to a loaded prompt, raise an error if they do not match

	# Parameters:
	- `prompt : dict`
	- `prompt_loaded : dict`

	# Returns:
	- `None`

	# Raises:
	- `ActivationsMismatchError` : if the prompt text or hash do not match
	"""
	for key in ("text", "hash"):
		if prompt[key] != prompt_loaded[key]:
			msg = f"Prompt file does not match prompt at key {key}:\n{prompt}\n{prompt_loaded}"
			raise ActivationsMismatchError(
				msg,
			)


def augment_prompt_with_hash(prompt: dict) -> dict:
	"""if a prompt does not have a hash, add one

	not having a "text" field is allowed, but only if "hash" is present

	# Parameters:
	- `prompt : dict`

	# Returns:
	- `dict`

	# Modifies:
	the input `prompt` dictionary, if it does not have a `"hash"` key
	"""
	if "hash" not in prompt:
		if "text" not in prompt:
			msg = f"Prompt does not have 'text' field or 'hash' field: {prompt}"
			raise InvalidPromptError(
				msg,
			)
		prompt_str: str = prompt["text"]
		prompt_hash: str = (
			# we don't need this to be a secure hash
			base64.urlsafe_b64encode(hashlib.md5(prompt_str.encode()).digest())  # noqa: S324
			.decode()
			.rstrip("=")
		)
		prompt.update(hash=prompt_hash)
	return prompt


@overload
def load_activations(
	model_name: str,
	prompt: dict,
	save_path: Path,
	return_fmt: Literal["torch"] = "torch",
) -> "tuple[Path, dict[str, torch.Tensor]]":  # type: ignore[name-defined] # noqa: F821
	...
@overload
def load_activations(
	model_name: str,
	prompt: dict,
	save_path: Path,
	return_fmt: Literal["numpy"] = "numpy",
) -> "tuple[Path, dict[str, np.ndarray]]": ...
def load_activations(
	model_name: str,
	prompt: dict,
	save_path: Path,
	return_fmt: ReturnCache = "torch",
) -> "tuple[Path, dict[str, torch.Tensor]|dict[str, np.ndarray]]":  # type: ignore[name-defined] # noqa: F821
	"""load activations for a prompt and model, from an npz file

	# Parameters:
	- `model_name : str`
	- `prompt : dict`
	- `save_path : Path`
	- `return_fmt : Literal["torch", "numpy"]`
		(defaults to `"torch"`)

	# Returns:
	- `tuple[Path, dict[str, torch.Tensor]|dict[str, np.ndarray]]`
		the path to the activations file and the activations as a dictionary
		of numpy arrays or torch tensors, depending on `return_fmt`

	# Raises:
	- `ActivationsMissingError` : if the activations file is missing
	- `ValueError` : if `return_fmt` is not `"torch"` or `"numpy"`
	"""
	if return_fmt not in ("torch", "numpy"):
		msg = f"Invalid return_fmt: {return_fmt}, expected 'torch' or 'numpy'"
		raise ValueError(
			msg,
		)
	if return_fmt == "torch":
		import torch

	augment_prompt_with_hash(prompt)

	prompt_dir: Path = save_path / model_name / "prompts" / prompt["hash"]
	prompt_file: Path = prompt_dir / "prompt.json"
	if not prompt_file.exists():
		msg = f"Prompt file {prompt_file} does not exist"
		raise ActivationsMissingError(msg)
	with open(prompt_dir / "prompt.json", "r") as f:
		prompt_loaded: dict = json.load(f)
		compare_prompt_to_loaded(prompt, prompt_loaded)

	activations_path: Path = prompt_dir / "activations.npz"

	cache: dict

	with np.load(activations_path) as npz_data:
		if return_fmt == "numpy":
			cache = dict(npz_data.items())
		elif return_fmt == "torch":
			cache = {k: torch.from_numpy(v) for k, v in npz_data.items()}

	return activations_path, cache


# def load_activations_stacked()
