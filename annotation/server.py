#!/usr/bin/env python3
import sys, json, re, pathlib, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
SUB = HERE / "submissions"; SUB.mkdir(exist_ok=True)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", (name or "anon"))[:60]


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/tiles.json":
            self._send(200, (HERE / "tiles.json").read_bytes(), "application/json")
        elif path == "/health":
            n = len(list(SUB.glob("*.jsonl")))
            self._send(200, json.dumps({"ok": True, "students_so_far": n}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path.split("?")[0] != "/submit":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            rec = json.loads(self.rfile.read(n).decode())
        except Exception as e:
            return self._send(400, json.dumps({"error": f"bad json: {e}"}))
        student = safe(rec.get("student"))
        rec["server_time"] = datetime.datetime.now().isoformat(timespec="seconds")
        rec["remote"] = self.client_address[0]
        with open(SUB / f"{student}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._send(200, json.dumps({"ok": True}))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Annotation server on http://0.0.0.0:{PORT}  (submissions -> {SUB})")
    print("Share your machine's LAN/public IP + port with annotators. Ctrl-C to stop.")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
