"""JSON persistence for long-term learned knowledge."""
from __future__ import annotations

import json
from pathlib import Path

from .knowledge import KnowledgeItem, KnowledgeKind, KnowledgeSource, KnowledgeStatus


class KnowledgeStore:
    """Small durable store for knowledge; replaceable by a database later."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, items: list[KnowledgeItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([item.to_dict() for item in items], indent=2), encoding="utf-8")

    def load(self) -> list[KnowledgeItem]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            KnowledgeItem(
                id=item["id"],
                statement=item["statement"],
                kind=KnowledgeKind(item["kind"]),
                status=KnowledgeStatus(item["status"]),
                confidence=float(item.get("confidence", 0.0)),
                evidence=list(item.get("evidence", [])),
                sources=[KnowledgeSource(**source) for source in item.get("sources", [])],
                tags=list(item.get("tags", [])),
                created_at=item["created_at"],
                updated_at=item["updated_at"],
            )
            for item in data
        ]
