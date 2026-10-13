"""Streaming OpenAI-compatible client for the vLLM server on the pod.

STREAMING IS NOT OPTIONAL HERE. The RunPod proxy terminates any request that
produces no bytes for roughly 100 seconds, and a 15-page multimodal prefill alone
can exceed that. With stream=true the first token starts a steady byte flow and
the proxy stays happy. complete() therefore refuses to run non-streaming unless
inference.allow_nonstream is explicitly set: the timeout is a property of the
deployment, not a knob.

Three separate clocks, because a slow prefill and a stalled generation are
different failures needing different responses:
  connect_timeout_s     - the pod is down, or the port is not exposed
  first_byte_timeout_s  - prefill is taking too long (too many image tokens)
  inter_chunk_timeout_s - generation stalled after starting

Nothing generated is ever thrown away. A stream that dies mid-flight returns its
partial text with partial=True; that text may still repair into a scoreable JSON
prefix, and it is written to disk before any retry.

MockClient has the same surface and no network, so the entire matrix - writing,
parsing, repair, scoring, reporting - can be exercised with zero GPU.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests


@dataclass
class ChatResult:
    text: str = ""
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    first_byte_s: float | None = None
    total_s: float = 0.0
    n_chunks: int = 0
    http_status: int | None = None
    error: str | None = None
    partial: bool = False
    attempts: int = 1
    max_tokens_used: int | None = None
    # Latency is reported per attempt AND in total. total_s/first_byte_s above
    # describe the attempt whose text is kept; wall_s is the whole call
    # including failed attempts, the 1.5x length retry and retry backoff - the
    # number a caller actually waited.
    wall_s: float = 0.0
    attempt_log: list = field(default_factory=list)

    @property
    def decode_s(self) -> float | None:
        if self.first_byte_s is None:
            return None
        return round(max(0.0, self.total_s - self.first_byte_s), 3)

    @property
    def cached_tokens(self) -> int | None:
        """Prompt tokens served from vLLM's prefix cache, when the server reports
        them (vllm serve --enable-prompt-tokens-details)."""
        details = (self.usage or {}).get("prompt_tokens_details") or {}
        return details.get("cached_tokens")

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)

    @property
    def hit_max_tokens(self) -> bool:
        return self.finish_reason == "length"


class VLLMClient:
    """One model, one port, one proxy URL."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 inf: dict | None = None, verify_tls: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.inf = inf or {}
        self.verify = verify_tls
        self.session = requests.Session()

    # ---------------------------------------------------------------- plumbing

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _url(self, path: str) -> str:
        base = self.base_url
        if not base.endswith("/v1"):
            base = base + "/v1"
        return f"{base}{path}"

    def models(self) -> list[str]:
        r = self.session.get(self._url("/models"), headers=self._headers(),
                             timeout=(self.inf.get("connect_timeout_s", 30), 30),
                             verify=self.verify)
        r.raise_for_status()
        return [m.get("id") for m in (r.json().get("data") or [])]

    def health(self) -> tuple[bool, str]:
        try:
            ids = self.models()
        except Exception as exc:  # noqa: BLE001 - preflight reports, never raises
            return False, f"{type(exc).__name__}: {exc}"
        if self.model not in ids:
            return False, (f"server is up but serves {ids!r}, not {self.model!r} - "
                           f"check --served-model-name against models[].served_name")
        return True, f"serving {self.model}"

    # ---------------------------------------------------------------- generation

    def _payload(self, messages: list[dict], max_tokens: int) -> dict:
        inf = self.inf
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(inf.get("temperature", 0.0)),
            "top_p": float(inf.get("top_p", 1.0)),
            "stream": True,
            "stream_options": {"include_usage": bool(inf.get("include_usage", True))},
        }
        if inf.get("seed") is not None:
            payload["seed"] = int(inf["seed"])
        payload.update(inf.get("extra_body") or {})
        return payload

    def _stream_once(self, messages: list[dict], max_tokens: int) -> ChatResult:
        inf = self.inf
        connect_t = float(inf.get("connect_timeout_s", 30))
        first_t = float(inf.get("first_byte_timeout_s", 180))
        inter_t = float(inf.get("inter_chunk_timeout_s", 60))
        total_t = float(inf.get("total_timeout_s", 1800))

        res = ChatResult(max_tokens_used=int(max_tokens))
        parts: list[str] = []
        t0 = time.monotonic()
        last = t0

        try:
            # The socket read timeout must cover the prefill, so it is set to the
            # first-byte budget. The tighter inter-chunk budget is enforced between
            # yields; once decoding starts, chunks arrive continuously and the two
            # agree. A prefill that never finishes is caught by the read timeout.
            resp = self.session.post(
                self._url("/chat/completions"),
                headers=self._headers(),
                json=self._payload(messages, max_tokens),
                stream=True,
                timeout=(connect_t, first_t),
                verify=self.verify,
            )
            res.http_status = resp.status_code
            if resp.status_code >= 400:
                body = resp.text[:800]
                res.error = f"HTTP {resp.status_code}: {body}"
                return res

            for raw in resp.iter_lines(decode_unicode=True):
                now = time.monotonic()
                if now - t0 > total_t:
                    res.error = f"total timeout after {total_t:.0f}s"
                    break
                if raw is None or raw == "":
                    continue
                if res.first_byte_s is None:
                    # First byte. Its budget is first_byte_timeout_s, enforced by
                    # the socket read timeout above - NOT the inter-chunk budget,
                    # which is much tighter. Applying the inter-chunk budget here
                    # would report a slow multimodal prefill as a stalled
                    # generation, which is both wrong and indistinguishable from
                    # a model failure in the results.
                    res.first_byte_s = round(now - t0, 3)
                elif now - last > inter_t:
                    res.error = f"generation stalled: no chunk for {now - last:.0f}s"
                    break
                last = now

                if not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                res.n_chunks += 1
                if obj.get("usage"):
                    res.usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        parts.append(piece)
                    if choice.get("finish_reason"):
                        res.finish_reason = choice["finish_reason"]
        except requests.exceptions.ReadTimeout:
            res.error = (f"no first byte within {first_t:.0f}s - the prefill is too "
                         f"slow or the proxy dropped the request")
        except requests.exceptions.ConnectTimeout:
            res.error = f"could not connect within {connect_t:.0f}s"
        except requests.exceptions.RequestException as exc:
            res.error = f"{type(exc).__name__}: {exc}"

        res.text = "".join(parts)
        res.total_s = round(time.monotonic() - t0, 3)
        res.partial = bool(res.error) and bool(res.text)
        return res

    def complete(self, messages: list[dict], max_tokens: int | None = None,
                 on_partial=None) -> ChatResult:
        """Generate, retrying transient failures and once on a length cutoff.

        on_partial(text, attempt) is called with any salvageable text BEFORE a
        retry, so a stream that died is on disk even if every retry then fails.
        """
        inf = self.inf
        if not inf.get("stream", True) and not inf.get("allow_nonstream", False):
            raise RuntimeError(
                "inference.stream is false. The RunPod proxy kills requests that "
                "send no bytes for ~100s, and a multimodal prefill exceeds that. "
                "Set inference.allow_nonstream: true only if you know the "
                "deployment does not have that limit."
            )

        budget = int(max_tokens or inf.get("max_tokens", 8192))
        retries = int(inf.get("retries", 2))
        backoff = float(inf.get("retry_backoff_s", 10))

        t_start = time.monotonic()
        log: list[dict] = []

        def done(res: ChatResult) -> ChatResult:
            res.attempt_log = log
            res.wall_s = round(time.monotonic() - t_start, 3)
            return res

        attempt = 0
        last: ChatResult | None = None
        while attempt <= retries:
            attempt += 1
            res = self._stream_once(messages, budget)
            res.attempts = attempt
            last = res
            log.append({
                "attempt": attempt,
                "max_tokens": budget,
                "first_byte_s": res.first_byte_s,
                "total_s": res.total_s,
                "decode_s": res.decode_s,
                "prompt_tokens": (res.usage or {}).get("prompt_tokens"),
                "completion_tokens": (res.usage or {}).get("completion_tokens"),
                "cached_tokens": res.cached_tokens,
                "finish_reason": res.finish_reason,
                "http_status": res.http_status,
                "error": res.error,
                "chars": len(res.text or ""),
            })

            if res.error and res.text and on_partial:
                on_partial(res.text, attempt)

            if not res.error:
                if res.hit_max_tokens and inf.get("retry_on_length", True) and attempt == 1:
                    if on_partial:
                        on_partial(res.text, attempt)
                    budget = int(budget * 1.5)
                    continue
                return done(res)

            # A 4xx is a request problem: a bigger budget or another try will not
            # fix a payload the server rejected.
            if res.http_status and 400 <= res.http_status < 500:
                return done(res)
            if attempt <= retries:
                time.sleep(backoff)

        return done(last or ChatResult(error="no attempt was made"))


# --------------------------------------------------------------------------- mock


class MockClient:
    """Same surface as VLLMClient, synthesized from the gold. No network.

    Lets the whole matrix - writing, parsing, repair, schema validation, scoring,
    reporting - be exercised before the pod exists. The `garbage` profile must
    produce a clean INVALID cell rather than a stack trace; that is the point of
    having it.
    """

    PROFILES = ("perfect", "degraded", "truncated", "garbage")

    def __init__(self, gold: dict, profile: str = "degraded", model: str = "mock",
                 seed: int = 0):
        if profile not in self.PROFILES:
            raise ValueError(f"unknown mock profile {profile!r}; expected {self.PROFILES}")
        self.gold = gold or {}
        self.profile = profile
        self.model = model
        self.seed = seed

    def health(self) -> tuple[bool, str]:
        return True, f"mock client, profile={self.profile}"

    def models(self) -> list[str]:
        return [self.model]

    def _degrade(self, rng: random.Random) -> dict:
        import scoring
        obj = copy.deepcopy(self.gold)

        # Reorder every list of objects: a correct scorer must not care.
        def reverse(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: reverse(v) for k, v in node.items()}
            if isinstance(node, list):
                items = [reverse(v) for v in node]
                if any(isinstance(v, dict) for v in node):
                    items.reverse()
                return items
            return node

        # Reformat dates and drop roughly one leaf in six: a real degradation the
        # scorer must see, layered on a cosmetic one it must not.
        def touch(node: Any, path: str = "") -> Any:
            if isinstance(node, dict):
                out = {}
                for k, v in node.items():
                    key = scoring.child_spec(path, k)
                    if not isinstance(v, (dict, list)) and rng.random() < 0.17:
                        continue
                    out[k] = touch(v, key)
                return out
            if isinstance(node, list):
                return [touch(v, scoring.item_spec(path)) for v in node]
            if isinstance(node, str):
                iso = scoring.as_date(node)
                if iso:
                    return iso if "/" in node else iso
            return node

        return touch(reverse(obj))

    def complete(self, messages: list[dict], max_tokens: int | None = None,
                 on_partial=None) -> ChatResult:
        rng = random.Random(self.seed + len(json.dumps(messages, default=str)))
        t0 = time.monotonic()

        if self.profile == "garbage":
            text = ("I was unable to read this document reliably. The pages appear "
                    "to be a commercial auto declaration, but the tables are not "
                    "legible enough to transcribe.")
            finish = "stop"
        elif self.profile == "perfect":
            text = json.dumps(self.gold, ensure_ascii=False)
            finish = "stop"
        elif self.profile == "truncated":
            blob = json.dumps(self.gold, ensure_ascii=False)
            text = blob[: int(len(blob) * 0.7)]
            finish = "length"
        else:
            text = json.dumps(self._degrade(rng), ensure_ascii=False)
            finish = "stop"

        elapsed = round(time.monotonic() - t0, 3)
        return ChatResult(
            text=text,
            finish_reason=finish,
            usage={"prompt_tokens": 0, "completion_tokens": len(text) // 4,
                   "total_tokens": len(text) // 4},
            first_byte_s=0.0,
            total_s=elapsed,
            n_chunks=1,
            http_status=200,
            wall_s=elapsed,
            attempt_log=[{"attempt": 1, "first_byte_s": 0.0, "total_s": elapsed,
                          "finish_reason": finish, "mock": True}],
        )
