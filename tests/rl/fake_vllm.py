"""A stand-in for a vLLM server (the HTTP API the trainer uses), for tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeVLLM:
    """Answers /health, LoRA (un)loading and /v1/completions. Each completion is
    ``reply`` (token ids) with log-probability -0.5 per token; ``token_ids_field`` picks
    between vLLM's two ways of returning ids, and ``with_eos`` whether the stop token is
    included in them."""

    def __init__(self, reply: list[int], token_ids_field: bool = True, with_eos: bool = True):
        self.reply, self.token_ids_field, self.with_eos = reply, token_ids_field, with_eos
        self.loaded: list[str] = []
        self.unloaded: list[str] = []
        self.requests: list[dict[str, Any]] = []

    def completion(self, body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(body)
        ids = self.reply if self.with_eos else self.reply[:-1]
        logps = [-0.5] * len(ids)
        choices = []
        for i in range(body["n"]):
            choice: dict[str, Any] = {"index": i, "finish_reason": "stop", "text": "x"}
            if self.token_ids_field:
                choice["token_ids"] = ids
                choice["logprobs"] = {"token_logprobs": logps}
            else:
                tokens = [f"token_id:{t}" for t in ids]
                choice["logprobs"] = {"tokens": tokens, "token_logprobs": logps}
            choices.append(choice)
        return {"choices": choices[::-1]}  # out of order, as the client must not assume


@contextmanager
def serve(fake: FakeVLLM) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, payload: Any) -> None:
            data = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send("ok")

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/load_lora_adapter":
                fake.loaded.append(body["lora_name"])
                self._send("Success")
            elif self.path == "/v1/unload_lora_adapter":
                fake.unloaded.append(body["lora_name"])
                self._send("Success")
            else:
                self._send(fake.completion(body))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
