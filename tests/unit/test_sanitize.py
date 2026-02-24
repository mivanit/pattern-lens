"""tests for sanitize_name_str (pure string sanitization)"""

import pytest

from pattern_lens.consts import sanitize_name_str


class TestSanitizeNameStr:
	"test sanitize_name_str with various inputs to ensure it behaves as expected"

	@pytest.mark.parametrize(
		"name",
		["gpt2", "pythia-14m", "gpt2-small", "my_model", "model.v2"],
	)
	def test_passthrough_clean_names(self, name: str):
		assert sanitize_name_str(name) == name

	@pytest.mark.parametrize(
		("name", "expected"),
		[
			("google/gemma-2b", "google-gemma-2b"),
			("meta/llama-7b", "meta-llama-7b"),
			("org/sub/model", "org-sub-model"),
		],
	)
	def test_slash_replaced(self, name: str, expected: str):
		assert sanitize_name_str(name) == expected

	@pytest.mark.parametrize(
		("name", "expected"),
		[
			("model@v2", "model_v2"),
			("model name", "model_name"),
			("model:v2", "model_v2"),
		],
	)
	def test_other_special_chars_replaced(self, name: str, expected: str):
		assert sanitize_name_str(name) == expected

	@pytest.mark.parametrize(
		"name",
		["google/gemma-2b", "gpt2", "org/sub/model", "model@v2"],
	)
	def test_idempotent(self, name: str):
		once = sanitize_name_str(name)
		twice = sanitize_name_str(once)
		assert once == twice

	def test_empty_string(self):
		assert sanitize_name_str("") == ""
