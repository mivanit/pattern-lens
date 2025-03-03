"implements `load_text_data` for loading prompts"

import json
import random
from pathlib import Path


def load_text_data(
	fname: Path,
	min_chars: int | None = None,
	max_chars: int | None = None,
	shuffle: bool = False,
) -> list[dict]:
	"""given `fname`, the path to a jsonl file, split prompts up into more reasonable sizes

	# Parameters:
	- `fname : Path`
		jsonl file with prompts. Expects a list of dicts with a "text" key
	- `min_chars : int | None`
		(defaults to `None`)
	- `max_chars : int | None`
		(defaults to `None`)
	- `shuffle : bool`
		(defaults to `False`)

	# Returns:
	- `list[dict]`
		processed list of prompts. Each prompt has a "text" key w/ a string value and some metadata.
		this is not guaranteed to be the same length as the input list!
	"""
	# read raw data
	with open(fname, "r") as f:
		data_raw: list[dict] = [json.loads(d) for d in f.readlines()]

	# add fname metadata
	for d in data_raw:
		d["source_fname"] = fname.as_posix()

	# trim too-short samples
	if min_chars is not None:
		data_raw = list(
			filter(
				lambda x: len(x["text"]) >= min_chars,
				data_raw,
			),
		)

	# split up too-long samples
	if max_chars is not None:
		data_new: list[dict] = []
		for d in data_raw:
			d_text: str = d["text"]
			while len(d_text) > max_chars:
				data_new.append(
					{
						**d,
						"text": d_text[:max_chars],
					},
				)
				d_text = d_text[max_chars:]
			data_new.append(
				{
					**d,
					"text": d_text,
				},
			)
		data_raw = data_new

	# trim too-short samples again
	if min_chars is not None:
		data_raw = list(
			filter(
				lambda x: len(x["text"]) >= min_chars,
				data_raw,
			),
		)

	# shuffle
	if shuffle:
		random.shuffle(data_raw)

	return data_raw
