"""Internet research primitives for autonomous learning.

The researcher retrieves public HTTP(S) resources and preserves provenance.
Search providers can be layered on top later; this module deliberately keeps
retrieval separate from verification and learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from urllib.request import Request, urlopen
from urllib.parse import urlparse


@dataclass(frozen=True)
class ResearchDocument:
    uri: str
    content: str
    content_hash: str
    retrieved_at: str
    content_type: str = ""


class InternetResearcher:
    """Fetch Internet documents without treating retrieved text as truth."""

    def __init__(self, *, timeout: float = 15.0, max_bytes: int = 2_000_000, user_agent: str = "AI-Agent-Researcher/0.1"):
        if timeout <= 0 or max_bytes <= 0:
            raise ValueError("timeout and max_bytes must be positive")
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.user_agent = user_agent

    def fetch(self, uri: str) -> ResearchDocument:
        parsed = urlparse(uri)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("only absolute HTTP(S) URLs are supported")
        request = Request(uri, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("response exceeds configured maximum size")
        content = data.decode("utf-8", errors="replace")
        return ResearchDocument(
            uri=uri,
            content=content,
            content_hash=sha256(data).hexdigest(),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            content_type=content_type,
        )
