"""
OllamaBackend — local Ollama API wrapper.

Refactor of the existing chatbot.py logic into the backend interface.
No behavior change — same endpoint, same payload shape, same defaults.

Owns the Ollama lifecycle: starts the server if not running, waits for
readiness on construction. If `ollama` is not installed on the system,
logs a warning and continues — the backend just won't work, but
construction won't crash (matters for cloud deployments where local is
intentionally absent).
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
import requests
from typing import Callable, List, Optional

from brain.nlp.backends.base import LLMBackend
from config_metrics.logging import log_debug


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(
        self,
        model: str = "argus-40b",
        host: str = "localhost",
        port: int = 11434,
        auto_start: bool = True,
    ):
        self.model = model
        self.host = host
        self.port = port
        if auto_start:
            self.ensure_running()

    # ── Lifecycle ──────────────────────────────────────────
    def is_running(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=2):
                return True
        except socket.error:
            return False

    def wait_for_ready(self, timeout: int = 20) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.is_running():
                return True
            time.sleep(1)
        return False

    def ensure_running(self):
        if self.is_running():
            log_debug(f"[OllamaBackend] Already running on {self.host}:{self.port}")
            return
        log_debug(f"[OllamaBackend] Not running. Starting model '{self.model}'...")
        try:
            subprocess.Popen(["ollama", "run", self.model])
            if not self.wait_for_ready():
                log_debug("[OllamaBackend] Ollama did not become ready in time. Local fallback may not work.")
        except FileNotFoundError:
            log_debug("[OllamaBackend] `ollama` binary not found on PATH. Local fallback unavailable.")
        except Exception as e:
            log_debug(f"[OllamaBackend] Could not start Ollama: {e}. Local fallback unavailable.")

    # ── Payload builder ────────────────────────────────────
    def _payload(self, system: str, user: str, opts: dict, stream: bool) -> dict:
        return {
            "model": opts.get("model", self.model),
            "stream": stream,
            "messages": [
                {"role": "system", "content": str(system or "")},
                {"role": "user",   "content": str(user   or "")},
            ],
            "options": {
                "temperature": float(opts.get("temperature", 0.7)),
                "num_predict": int(opts.get("num_predict", 2048)),
                "top_k": int(opts.get("top_k", 30)),
                "top_p": float(opts.get("top_p", 0.9)),
                "stop": opts.get("stop") or ["<<SYSTEM CONTEXT", "USER_MESSAGE:", "DRAFT_ANSWER:"],
            },
        }

    # ── Main API ───────────────────────────────────────────
    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.7,
        num_predict: int = 2048,
        timeout: float = 120.0,
        stop: Optional[List[str]] = None,
        **opts,
    ) -> str:
        opts = {**opts, "temperature": temperature, "num_predict": num_predict, "stop": stop}
        actual_timeout = max(timeout, 120 + int(num_predict) / 3)
        try:
            r = requests.post(
                f"http://{self.host}:{self.port}/api/chat",
                json=self._payload(system, user, opts, stream=False),
                timeout=actual_timeout,
            )
            r.raise_for_status()
            data = r.json()
            return (data.get("message", {}) or {}).get("content", "").strip()
        except Exception as e:
            log_debug(f"[OllamaBackend] generate error: {e}")
            return ""

    def stream(
        self,
        system: str,
        user: str,
        on_chunk: Callable[[str], None],
        *,
        temperature: float = 0.7,
        num_predict: int = 2048,
        timeout: float = 600.0,
        stop: Optional[List[str]] = None,
        **opts,
    ) -> str:
        opts = {**opts, "temperature": temperature, "num_predict": num_predict, "stop": stop}
        actual_timeout = max(timeout, 120 + int(num_predict) / 3)
        full = ""
        try:
            r = requests.post(
                f"http://{self.host}:{self.port}/api/chat",
                json=self._payload(system, user, opts, stream=True),
                timeout=actual_timeout,
                stream=True,
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full += token
                    if on_chunk:
                        try:
                            on_chunk(token)
                        except Exception as ce:
                            log_debug(f"[OllamaBackend] on_chunk error: {ce}")
                if chunk.get("done", False):
                    break
        except Exception as e:
            log_debug(f"[OllamaBackend] stream error: {e}")
        return full.strip()
