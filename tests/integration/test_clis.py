import sys
from unittest import mock

import pytest

from pattern_lens.activations import main as activations_main
from pattern_lens.figures import main as figures_main
from pattern_lens.server import main as server_main


def test_activations_cli():
	"""Test the activations command line interface."""
	test_args = [
		"pattern_lens.activations",
		"--model",
		"gpt2",
		"--prompts",
		"test_prompts.jsonl",
		"--save-path",
		"test_data",
		"--min-chars",
		"100",
		"--max-chars",
		"1000",
		"--n-samples",
		"5",
		"--force",
		"--raw-prompts",
		"--shuffle",
		"--device",
		"cpu",
	]

	with (
		mock.patch.object(sys, "argv", test_args),
		mock.patch(
			"pattern_lens.activations.activations_main",
		) as mock_activations_main,
	):
		# Mock SpinnerContext to prevent actual spinner during tests
		with mock.patch("pattern_lens.activations.SpinnerContext"):
			activations_main()

		# Check that activations_main was called with the right arguments
		mock_activations_main.assert_called_once()
		args, kwargs = mock_activations_main.call_args

		assert kwargs["model_name"] == "gpt2"
		assert kwargs["prompts_path"] == "test_prompts.jsonl"
		assert kwargs["save_path"] == "test_data"
		assert kwargs["min_chars"] == 100
		assert kwargs["max_chars"] == 1000
		assert kwargs["n_samples"] == 5
		assert kwargs["force"] is True
		assert kwargs["raw_prompts"] is True
		assert kwargs["shuffle"] is True
		assert kwargs["device"] == "cpu"


def test_figures_cli():
	"""Test the figures command line interface."""
	test_args = [
		"pattern_lens.figures",
		"--model",
		"gpt2",
		"--save-path",
		"test_data",
		"--n-samples",
		"5",
		"--force",
		"True",
	]

	with (
		mock.patch.object(sys, "argv", test_args),
		mock.patch("pattern_lens.figures.figures_main") as mock_figures_main,
	):
		# Mock SpinnerContext to prevent actual spinner during tests
		with mock.patch("pattern_lens.figures.SpinnerContext"):
			figures_main()

		# Check that figures_main was called with the right arguments
		mock_figures_main.assert_called_once()
		args, kwargs = mock_figures_main.call_args

		assert kwargs["model_name"] == "gpt2"
		assert kwargs["save_path"] == "test_data"
		assert kwargs["n_samples"] == 5
		assert kwargs["force"] is True


def test_figures_cli_with_multiple_models():
	"""Test the figures command line interface with multiple models."""
	test_args = [
		"pattern_lens.figures",
		"--model",
		"gpt2,pythia-70m",
		"--save-path",
		"test_data",
	]

	with (
		mock.patch.object(sys, "argv", test_args),
		mock.patch("pattern_lens.figures.figures_main") as mock_figures_main,
	):
		# Mock SpinnerContext to prevent actual spinner during tests
		with mock.patch("pattern_lens.figures.SpinnerContext"):
			figures_main()

		# Check that figures_main was called for each model
		assert mock_figures_main.call_count == 2

		# First call should be for gpt2
		args1, kwargs1 = mock_figures_main.call_args_list[0]
		assert kwargs1["model_name"] == "gpt2"

		# Second call should be for pythia-70m
		args2, kwargs2 = mock_figures_main.call_args_list[1]
		assert kwargs2["model_name"] == "pythia-70m"


def test_server_cli():
	"""Test the server command line interface."""
	test_args = [
		"pattern_lens.server",
		"--path",
		"custom_path",
		"--port",
		"8080",
		"--rewrite-index",
	]

	with (
		mock.patch.object(sys, "argv", test_args),
		mock.patch("pattern_lens.server.main") as mock_server_main,
		mock.patch("pattern_lens.server.write_html_index") as mock_write_html,
	):
		# Mock to prevent actual server startup
		mock_server_main.side_effect = KeyboardInterrupt()

		# Call main with the test arguments
		with pytest.raises(KeyboardInterrupt):
			server_main()

		# Check that write_html_index was called
		mock_write_html.assert_called_once()

		# Check that server_main was called with the right arguments
		mock_server_main.assert_called_once_with(path="custom_path", port=8080)
