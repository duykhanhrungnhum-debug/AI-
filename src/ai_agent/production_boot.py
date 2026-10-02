"""Production bootstrap for AIKA.

Production has one stable Kaggle brain identity. Worker lifecycle/reuse is owned
by AIKAAgentBroker, so deploys do not replace an already healthy/starting
notebook merely to run a test.
"""
from __future__ import annotations

from ai_agent.agent_broker import AGENT_BROKER
from ai_agent.chat_api import main as chat_main


PRODUCTION_BRAIN_SLUG = "aika-brain-prod-v2"
PRODUCTION_BRAIN_TITLE = "AIKA Brain Prod V2"


def configure() -> None:
    AGENT_BROKER.kernel_slug = PRODUCTION_BRAIN_SLUG
    AGENT_BROKER.kernel_title = PRODUCTION_BRAIN_TITLE


def main() -> None:
    configure()
    chat_main()


if __name__ == "__main__":
    main()
