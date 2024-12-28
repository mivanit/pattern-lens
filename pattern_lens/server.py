from pathlib import Path
import sys
import os
import argparse
import http.server
import socketserver

from pattern_lens.indexes import write_html_index


def main(path: str, port: int = 8000):
    os.chdir(path)
    try:
        with socketserver.TCPServer(
            ("", port), http.server.SimpleHTTPRequestHandler
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
        default=".",
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
