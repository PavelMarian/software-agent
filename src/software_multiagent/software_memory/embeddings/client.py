from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingClientError(RuntimeError):
    """A safe, provider-independent embedding API failure."""


@runtime_checkable
class EmbeddingClient(Protocol):
    provider: str
    model: str
    dimensions: int | None

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...


Transport = Callable[[Request, float], bytes]


def _default_transport(request: Request, timeout: float) -> bytes:
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured API endpoint
        return response.read()


@dataclass
class OpenAICompatibleEmbeddingClient:
    """Small dependency-free client for OpenAI-compatible embedding endpoints."""

    api_key: str
    model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    provider: str = "openai-compatible"
    dimensions: int | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    extra_headers: dict[str, str] = field(default_factory=dict)
    transport: Transport = field(default=_default_transport, repr=False)

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/embeddings") else f"{base}/embeddings"

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        if not self.api_key.strip():
            raise EmbeddingClientError("embedding API key is empty")
        payload: dict[str, object] = {"model": self.model, "input": list(texts)}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.extra_headers,
        }
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        response_body: bytes | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response_body = self.transport(request, self.timeout_seconds)
                break
            except HTTPError as error:
                retryable = error.code == 429 or error.code >= 500
                if not retryable or attempt >= self.max_retries:
                    raise EmbeddingClientError(
                        f"embedding API returned HTTP {error.code}"
                    ) from error
            except (URLError, TimeoutError, OSError) as error:
                if attempt >= self.max_retries:
                    raise EmbeddingClientError("embedding API request failed") from error
            time.sleep(min(4.0, 0.5 * (2**attempt)))
        if response_body is None:  # pragma: no cover - defensive invariant
            raise EmbeddingClientError("embedding API returned no response")
        try:
            decoded = json.loads(response_body.decode("utf-8"))
            rows = decoded["data"]
            ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
            vectors = tuple(tuple(float(value) for value in row["embedding"]) for row in ordered)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise EmbeddingClientError("embedding API returned an invalid response") from error
        if len(vectors) != len(texts):
            raise EmbeddingClientError("embedding API returned a different number of vectors")
        widths = {len(vector) for vector in vectors}
        if len(widths) != 1 or not widths or 0 in widths:
            raise EmbeddingClientError("embedding vectors have inconsistent dimensions")
        if self.dimensions is not None and widths != {self.dimensions}:
            raise EmbeddingClientError("embedding API ignored the requested dimensions")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise EmbeddingClientError("embedding API returned non-finite vector values")
        return vectors
