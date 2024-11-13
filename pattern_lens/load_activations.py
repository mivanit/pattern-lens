import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pattern_lens.consts import AttentionMatrix


class GetActivationsError(ValueError):
    pass


class ActivationsMissingError(GetActivationsError):
    pass


class ActivationsMismatchError(GetActivationsError):
    pass


def compare_prompt_to_loaded(prompt: dict, prompt_loaded: dict) -> None:
    for key in ("text", "hash"):
        if prompt[key] != prompt_loaded[key]:
            raise ActivationsMismatchError(
                f"Prompt file does not match prompt at key {key}:\n{prompt}\n{prompt_loaded}"
            )


def augment_prompt_with_hash(prompt: dict) -> dict:
    if "hash" not in prompt:
        prompt_str: str = prompt["text"]
        prompt_hash: str = (
            base64.urlsafe_b64encode(hashlib.md5(prompt_str.encode()).digest())
            .decode()
            .rstrip("=")
        )
        prompt.update(hash=prompt_hash)
    return prompt


def load_activations(
    model_name: str,
    prompt: dict,
    save_path: Path,
) -> tuple[Path, dict[str, AttentionMatrix]]:
    augment_prompt_with_hash(prompt)

    prompt_dir: Path = save_path / model_name / "prompts" / prompt["hash"]
    prompt_file: Path = prompt_dir / "prompt.json"
    if not prompt_file.exists():
        raise ActivationsMissingError(f"Prompt file {prompt_file} does not exist")
    with open(prompt_dir / "prompt.json", "r") as f:
        prompt_loaded: dict = json.load(f)
        compare_prompt_to_loaded(prompt, prompt_loaded)

    activations_path: Path = prompt_dir / "activations.npz"

    with np.load(activations_path) as data:
        cache = {k: torch.from_numpy(v) for k, v in data.items()}

    return activations_path, cache
