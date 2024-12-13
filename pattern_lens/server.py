import sys
import os
import argparse
import http.server
import socketserver

def main(path: str, port: int = 8000):
    os.chdir(path)
    with socketserver.TCPServer(
        ("", port), http.server.SimpleHTTPRequestHandler
    ) as httpd:
        print(f"Serving at http://localhost:{port}")
        httpd.serve_forever()


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
    args: argparse.Namespace = arg_parser.parse_args()

    main(path=args.path, port=args.port)



