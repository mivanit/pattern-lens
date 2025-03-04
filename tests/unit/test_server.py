# tests/unit/test_server.py
from unittest import mock

from pattern_lens.server import main as server_main


def test_server_main():
	"""Test the server main function."""
	# Mock the TCPServer and other components
	with (
		mock.patch("pattern_lens.server.socketserver.TCPServer") as mock_server,
		mock.patch(
			"pattern_lens.server.http.server.SimpleHTTPRequestHandler",
		) as mock_handler,
		mock.patch("pattern_lens.server.os.chdir") as mock_chdir,
		mock.patch("pattern_lens.server.sys.exit") as mock_exit,
	):
		# Setup the server to raise KeyboardInterrupt after being called
		mock_server_instance = mock_server.return_value.__enter__.return_value
		mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

		# Call the server main function
		server_main("test_path", 8080)

		# Check that the function changed to the right directory
		mock_chdir.assert_called_once_with("test_path")

		# Check that the server was initialized with the right parameters
		mock_server.assert_called_once_with(("", 8080), mock_handler)

		# Check that serve_forever was called
		mock_server_instance.serve_forever.assert_called_once()

		# Check that sys.exit was called with 0 (clean exit)
		mock_exit.assert_called_once_with(0)
