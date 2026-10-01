"""Canary-only AIKA chat API entrypoint for Warm Image Worker v6.

Production keeps the existing entrypoint until v6 passes real reuse + batch E2E.
"""
from __future__ import annotations

from ai_agent.chat_session import CHAT_BROKER
from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6


# Replace only the warm image lifecycle. All routing, job storage, cold fallback,
# auth, chat, translation, TTS and other skills remain unchanged.
CHAT_BROKER._warm_image = WarmImageWorkerManagerV6()

from ai_agent.chat_api import main as _main  # noqa: E402


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
