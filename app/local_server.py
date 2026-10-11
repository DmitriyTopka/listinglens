"""Run the Lambda handler locally: python app/local_server.py  ->  http://localhost:8765"""
import base64
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(__file__))
from lambda_function import handler  # noqa: E402


class H(BaseHTTPRequestHandler):
    def _go(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        ev = {"rawPath": self.path.split("?")[0], "requestContext": {"http": {"method": method}},
              "body": base64.b64encode(body).decode(), "isBase64Encoded": True}
        r = handler(ev)
        out = base64.b64decode(r["body"]) if r.get("isBase64Encoded") else r["body"].encode()
        self.send_response(r["statusCode"])
        for k, v in r["headers"].items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        self._go("GET")

    def do_POST(self):
        self._go("POST")


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("PORT", 8765))), H).serve_forever()
