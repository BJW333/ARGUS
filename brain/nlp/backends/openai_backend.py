"""
OpenAIBackend — OpenAI Chat Completions API.

Alternative primary backend if you prefer GPT over Claude. Uses raw
requests instead of the SDK. Streaming uses SSE.

Required env: OPENAI_API_KEY

Default model: gpt-4o. Override with ARGUS_API_MODEL.

Note: Anthropic is still recommended as primary for ARGUS because:
  - Native MCP support in the API (Phase 5)
  - Computer Use API (Phase 9)
  - More permissive on legitimate technical/security topics
"""
from __future__ import annotations

import json
import os
import requests
from typing import Callable, List, Optional

from brain.nlp.backends.base import LLMBackend
from config_metrics.logging import log_debug


class OpenAIBackend(LLMBackend):
    name = "openai"
    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            log_debug("[OpenAIBackend] WARNING: OPENAI_API_KEY not set. Backend disabled.")
        self.url = base_url or self.BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }

    def _body(self, system: str, user: str, opts: dict, stream: bool) -> dict:
        body = {
            "model": opts.get("model", self.model),
            "messages": [
                {"role": "system", "content": str(system or "")},
                {"role": "user",   "content": str(user   or "")},
            ],
            "max_tokens": int(opts.get("num_predict", 2048)),
            "temperature": float(opts.get("temperature", 0.7)),
            "stream": stream,
        }
        if opts.get("top_p") is not None:
            body["top_p"] = float(opts["top_p"])
        if opts.get("stop"):
            body["stop"] = list(opts["stop"])
        return body

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
        if not self.api_key:
            return ""
        opts = {**opts, "temperature": temperature, "num_predict": num_predict, "stop": stop}
        try:
            r = requests.post(
                self.url,
                headers=self.headers,
                json=self._body(system, user, opts, stream=False),
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "")[:500]
            log_debug(f"[OpenAIBackend] HTTP error: {e}  body={body}")
            return ""
        except Exception as e:
            log_debug(f"[OpenAIBackend] generate error: {e}")
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
        if not self.api_key:
            return ""
        opts = {**opts, "temperature": temperature, "num_predict": num_predict, "stop": stop}
        full = ""
        try:
            r = requests.post(
                self.url,
                headers=self.headers,
                json=self._body(system, user, opts, stream=True),
                timeout=timeout,
                stream=True,
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                try:
                    delta = evt["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        full += token
                        if on_chunk:
                            try:
                                on_chunk(token)
                            except Exception as ce:
                                log_debug(f"[OpenAIBackend] on_chunk error: {ce}")
                except (KeyError, IndexError):
                    continue
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "")[:500]
            log_debug(f"[OpenAIBackend] HTTP error: {e}  body={body}")
        except Exception as e:
            log_debug(f"[OpenAIBackend] stream error: {e}")
        return full.strip()
