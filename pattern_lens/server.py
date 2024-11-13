import argparse
import base64
import hashlib
import json
from pathlib import Path

def main():
    import http.server
    import socketserver

    port = 8000
    with socketserver.TCPServer(
        ("", port), http.server.SimpleHTTPRequestHandler
    ) as httpd:
        print(f"Serving at http://localhost:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
