"""Production bootstrap for AIKA.

Production injects one production-specific brain broker into the HTTP layer.
The generic broker still owns AIKA semantics; the production adapter owns only
stable accelerator selection and hardware validation.
"""
from __future__ import annotations

from ai_agent import agent_broker as agent_broker_module
from ai_agent import chat_api
from ai_agent.production_agent_broker import ProductionAIKAAgentBroker


PRODUCTION_BRAIN_SLUG = "aika-brain-prod-v2"
PRODUCTION_BRAIN_TITLE = "AIKA Brain Prod V2"


def configure() -> ProductionAIKAAgentBroker:
    current = chat_api.AGENT_BROKER
    if isinstance(current, ProductionAIKAAgentBroker):
        broker = current
    else:
        broker = ProductionAIKAAgentBroker()
    broker.kernel_slug = PRODUCTION_BRAIN_SLUG
    broker.kernel_title = PRODUCTION_BRAIN_TITLE

    # Keep the module export and HTTP layer pointed at the exact same broker.
    # This is explicit bootstrap dependency injection, not runtime routing.
    agent_broker_module.AGENT_BROKER = broker
    chat_api.AGENT_BROKER = broker
    return broker


def main() -> None:
    configure()
    chat_api.main()


if __name__ == "__main__":
    main()
