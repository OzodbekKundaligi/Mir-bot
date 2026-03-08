import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = os.getenv("WEB_HOST", "0.0.0.0").strip() or "0.0.0.0"
PORT = int((os.getenv("PORT") or os.getenv("WEB_PORT") or "8000").strip())


class HealthHandler(BaseHTTPRequestHandler):
    def _write_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"", "/"}:
            self._write_json(200, {"ok": True, "service": "kino-bot"})
            return
        if self.path == "/health":
            self._write_json(200, {"ok": True})
            return
        self._write_json(404, {"ok": False, "detail": "Not found"})

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), HealthHandler)
    print(f"[health] listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()
