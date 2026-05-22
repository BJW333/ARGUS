"""
AnthropicBackend — Claude API via /v1/messages.

Uses raw requests instead of the SDK to avoid adding dependencies.
Streaming uses Server-Sent Events.

Required env: ANTHROPIC_API_KEY

Default model: claude-opus-4-7 (most capable, $5/$25 per MTok).
Override with ARGUS_API_MODEL for cheaper alternatives:
    - claude-sonnet-4-6: $3/$15
    - claude-haiku-4-5:  $1/$5  (recommended for scheduled cron skills)
"""
from __future__ import annotations

import json
import os
import requests
from typing import Callable, List, Optional

from brain.nlp.backends.base import LLMBackend
from config_metrics.logging import log_debug


class AnthropicBackend(LLMBackend):
    name = "anthropic"
    BASE_URL = "https://api.anthropic.com/v1/messages"
    # Per-model max output token limits (from Anthropic docs).
    # max_tokens is required by the API but doesn't factor into rate limits,
    # so we set it to the model's max — no cost or rate penalty for unused
    # budget. Callers who want to cap output should pass num_predict explicitly.
    MODEL_MAX_OUTPUT = {
        "opus-4-7":   128_000,
        "opus-4-6":   128_000,
        "opus-4-5":   128_000,
        "opus-4-1":    32_000,
        "sonnet-4-6":  64_000,
        "sonnet-4-5":  64_000,
        "haiku-4-5":   64_000,
        "haiku-4":      8_192,
    }

    @classmethod
    def _max_output_for(cls, model: str) -> int:
        """Return the max output tokens for a model, defaulting to 8192 for unknowns."""
        m = (model or "").lower()
        for key, limit in cls.MODEL_MAX_OUTPUT.items():
            if key in m:
                return limit
        return 8192
    
    @staticmethod
    def _is_thinking_model(model: str) -> bool:
        """Thinking models (Opus 4.7+) deprecate temperature/top_p.
        They sample internally; passing those params returns 400."""
        m = (model or "").lower()
        return "opus-4-7" in m or "opus-4-8" in m  # forward-compatibility for future thinking models
    
    def __init__(
        self,
        model: str = "claude-opus-4-7",
        api_key: Optional[str] = None,
        anthropic_version: str = "2023-06-01",
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            log_debug("[AnthropicBackend] WARNING: ANTHROPIC_API_KEY not set. Backend disabled.")
        self.headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": anthropic_version,
            "content-type": "application/json",
        }

    def _body(self, system: str, user: str, opts: dict, stream: bool) -> dict:
        model = opts.get("model", self.model)
        body = {
            "model": model,
            "system": str(system or ""),
            "messages": [{"role": "user", "content": str(user or "")}],
            # Always send the model's max — no rate-limit penalty for unused
            # budget per Anthropic docs. Caller can override via opts["num_predict"]
            # if they want a hard cap (e.g., to bound latency).
            "max_tokens": min(
                int(opts.get("num_predict", 10**9)),
                self._max_output_for(model),
            ),
        }
        # Temperature & top_p deprecated on Opus 4.7+ thinking models.
        # Sonnet/Haiku/older Opus still accept them.
        if not self._is_thinking_model(model):
            body["temperature"] = float(opts.get("temperature", 0.7))
            if opts.get("top_p") is not None:
                body["top_p"] = float(opts["top_p"])
        if opts.get("stop"):
            body["stop_sequences"] = list(opts["stop"])
        if stream:
            body["stream"] = True
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
                self.BASE_URL,
                headers=self.headers,
                json=self._body(system, user, opts, stream=False),
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            return "".join(
                b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
            ).strip()
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "")[:500]
            log_debug(f"[AnthropicBackend] HTTP error: {e}  body={body}")
            return ""
        except Exception as e:
            log_debug(f"[AnthropicBackend] generate error: {e}")
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
                self.BASE_URL,
                headers=self.headers,
                json=self._body(system, user, opts, stream=True),
                timeout=timeout,
                stream=True,
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                etype = evt.get("type")
                if etype == "content_block_delta":
                    delta = evt.get("delta", {})
                    if delta.get("type") == "text_delta":
                        token = delta.get("text", "")
                        if token:
                            full += token
                            if on_chunk:
                                try:
                                    on_chunk(token)
                                except Exception as ce:
                                    log_debug(f"[AnthropicBackend] on_chunk error: {ce}")
                elif etype == "message_stop":
                    break
                elif etype == "error":
                    log_debug(f"[AnthropicBackend] stream error event: {evt}")
                    break
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "")[:500]
            log_debug(f"[AnthropicBackend] HTTP error: {e}  body={body}")
        except Exception as e:
            log_debug(f"[AnthropicBackend] stream error: {e}")
        return full.strip()
    
    # ════════════════════════════════════════════════════════════
    # PHASE 2: Tool-use streaming
    # ════════════════════════════════════════════════════════════

    def stream_with_tools(
        self,
        system: str,
        messages: list,
        tools: list,
        on_text_chunk=None,
        *,
        timeout: float = 600.0,
        num_predict=None,
        **opts,
    ) -> dict:
        """Stream a response with tool-use enabled. Returns dict with
        stop_reason, text, content_blocks (text + tool_use blocks)."""
        if not self.api_key:
            return {"stop_reason": "error", "text": "", "content_blocks": []}

        model = opts.get("model", self.model)
        body = {
            "model": model,
            "system": str(system or ""),
            "messages": messages,
            "tools": tools,
            "stream": True,
            "max_tokens": min(
                int(num_predict if num_predict is not None else 10**9),
                self._max_output_for(model),
            ),
        }
        if not self._is_thinking_model(model):
            if "temperature" in opts:
                body["temperature"] = float(opts["temperature"])
            if opts.get("top_p") is not None:
                body["top_p"] = float(opts["top_p"])

        content_blocks = []
        active_block = None
        active_tool_json_buffer = ""
        stop_reason = ""
        full_text = ""

        try:
            r = requests.post(
                self.BASE_URL, headers=self.headers, json=body,
                timeout=timeout, stream=True,
            )
            r.raise_for_status()

            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                try:
                    evt = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                etype = evt.get("type")

                if etype == "content_block_start":
                    block = evt.get("content_block", {}) or {}
                    btype = block.get("type")
                    if btype == "text":
                        active_block = {"type": "text", "text": ""}
                    elif btype == "tool_use":
                        active_block = {
                            "type": "tool_use",
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": {},
                        }
                        active_tool_json_buffer = ""
                    else:
                        active_block = {"type": btype, "raw": block}

                elif etype == "content_block_delta":
                    if active_block is None:
                        continue
                    delta = evt.get("delta", {}) or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        token = delta.get("text", "") or ""
                        if token:
                            active_block["text"] = active_block.get("text", "") + token
                            full_text += token
                            if on_text_chunk:
                                try:
                                    on_text_chunk(token)
                                except Exception as ce:
                                    log_debug(f"[AnthropicBackend] on_text_chunk error: {ce}")
                    elif dtype == "input_json_delta":
                        active_tool_json_buffer += delta.get("partial_json", "") or ""

                elif etype == "content_block_stop":
                    if active_block is not None:
                        if active_block.get("type") == "tool_use":
                            try:
                                parsed = json.loads(active_tool_json_buffer or "{}")
                                active_block["input"] = parsed if isinstance(parsed, dict) else {"_raw": parsed}
                            except json.JSONDecodeError as je:
                                log_debug(
                                    f"[AnthropicBackend] tool input JSON parse fail: {je}; "
                                    f"buffer={active_tool_json_buffer[:200]!r}"
                                )
                                active_block["input"] = {}
                            active_tool_json_buffer = ""
                        content_blocks.append(active_block)
                        active_block = None

                elif etype == "message_delta":
                    sr = (evt.get("delta", {}) or {}).get("stop_reason")
                    if sr:
                        stop_reason = sr

                elif etype == "message_stop":
                    break

                elif etype == "error":
                    log_debug(f"[AnthropicBackend] stream error event: {evt}")
                    break

        except requests.HTTPError as e:
            body_text = getattr(e.response, "text", "")[:500]
            log_debug(f"[AnthropicBackend] tool-use HTTP error: {e}  body={body_text}")
        except Exception as e:
            log_debug(f"[AnthropicBackend] tool-use stream error: {e}")

        return {
            "stop_reason": stop_reason or "end_turn",
            "text": full_text,
            "content_blocks": content_blocks,
        }
