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
		"--port",
		"8080",
		# "--path",
		# "test_path",
		"--rewrite-index",
	]

	with (
		mock.patch.object(sys, "argv", test_args),
		mock.patch("socketserver.TCPServer") as mock_server,
		# mock.patch("pattern_lens.server.write_html_index") as mock_write_html,
		# mock.patch("os.chdir") as mock_chdir,
	):
		# Set up the mock server to raise KeyboardInterrupt when serve_forever is called
		mock_server_instance = mock_server.return_value.__enter__.return_value
		mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

		# Call server_main
		with pytest.raises(SystemExit):
			server_main()

		# TODO: make these mock checks work -- I have no idea how to use mock properly
		# # Check that write_html_index was called
		# mock_write_html.assert_called_once()

		# # Check that chdir was called with the right path
		# mock_chdir.assert_called_once_with("test_path")

		# Check that server was started with the right port
		# mock_server.assert_called_once_with(("", 8080), mock.ANY)

		# Check that serve_forever was called
		# mock_server_instance.serve_forever.assert_called_once()
