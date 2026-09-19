"""Composition helpers for the model-backed runtime."""
from __future__ import annotations

from .engine import ExecutionEngine
from .experience_runtime import RuntimeExperienceRecorder
from .knowledge_retrieval import KnowledgeRetriever
from .model import ModelProvider
from .model_agent import ModelAgent
from .model_factory import ModelSettings, build_model_provider
from .model_runtime import ActionExecutor, ModelRuntime


def build_model_runtime(
    *,
    executor: ActionExecutor,
    engine: ExecutionEngine,
    provider: ModelProvider | None = None,
    settings: ModelSettings | None = None,
    experience_recorder: RuntimeExperienceRecorder | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
) -> ModelRuntime:
    selected_provider = provider or build_model_provider(settings)
    return ModelRuntime(
        ModelAgent(selected_provider),
        executor,
        engine,
        experience_recorder=experience_recorder,
        knowledge_retriever=knowledge_retriever,
    )
