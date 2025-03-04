"""cli for starting the server to show the web ui.

can also run with --rewrite-index to update the index.html file.
this is useful for working on the ui.
"""

import argparse
import http.server
import os
import socketserver
import sys
from pathlib import Path

from pattern_lens.indexes import write_html_index


def main(path: str | None = None, port: int = 8000) -> None:
	"move to the given path and start the server"
	if path is not None:
		os.chdir(path)
	try:
		with socketserver.TCPServer(
			("", port),
			http.server.SimpleHTTPRequestHandler,
		) as httpd:
			print(f"Serving at http://localhost:{port}")
			httpd.serve_forever()
	except KeyboardInterrupt:
		print("Server stopped")
		sys.exit(0)


if __name__ == "__main__":
	arg_parser: argparse.ArgumentParser = argparse.ArgumentParser()
	arg_parser.add_argument(
		"--path",
		type=str,
		required=False,
		help="The path to serve, defaults to the current directory",
		default=None,
	)
	arg_parser.add_argument(
		"--port",
		type=int,
		required=False,
		help="The port to serve on, defaults to 8000",
		default=8000,
	)
	arg_parser.add_argument(
		"--rewrite-index",
		action="store_true",
		help="Whether to write the latest index.html file",
	)
	args: argparse.Namespace = arg_parser.parse_args()

	if args.rewrite_index:
		write_html_index(path=Path(args.path))

	main(path=args.path, port=args.port)
