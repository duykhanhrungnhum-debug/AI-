# AI-

Autonomous AI Agent — self-learning, self-improving, execution-first.

## Core principles

- Truthfulness: never fabricate facts, actions, results, or capabilities.
- Verification: only report completion when the result is actually verified.
- Explicit uncertainty: distinguish known, unknown, inferred, and verified information.
- Honest failure: report blockers and failures directly.
- Persistence: actively attempt reasonable recovery when blocked.
- User authority: follow valid user instructions within system and safety constraints.
- Invariant core: the agent must not silently weaken or disable these principles.

## Implemented architecture

1. Task planning, resumable execution, recovery history, and verification/evidence.
2. Provenance-aware knowledge items and persistent knowledge storage.
3. Internet retrieval, search-provider abstraction, source comparison, and conservative conflict detection.
4. Autonomous learning loop with knowledge-gap detection and research planning.
5. Bounded experience/failure memory that changes later learning objectives.
6. Executable training curriculum with scoring, prerequisites, retries, progression, and failure lessons.
7. Bounded long-running runtime with explicit stop and failure limits.
8. Model-provider boundary that refuses to pretend a model is bundled when none is configured.
9. Immutable invariant checks are invoked throughout learning/training/runtime paths.

## Verification status

The repository uses GitHub Actions for the full pytest suite plus a public-Internet retrieval smoke test. A green CI run is required before a change is considered verified.

### Not falsely claimed as complete

- A real search backend is now implemented (`DuckDuckGoSearchProvider`), but external network availability must still be runtime-tested; `JsonSearchProvider` remains configurable rather than tied to a vendor.
- Long-running runtime infrastructure exists, but multi-hour/day endurance has not been proven in CI.
- No proprietary GPT-5.6 Luna weights or training corpus are embedded; those cannot be exported into this repository.
- Conflict detection is still lexical/conservative, not semantic reasoning.
- A model adapter boundary exists, but no specific AI model is bundled or secretly assumed.
- A production API/UI/deployment stack is not yet claimed complete.
- Self-improvement remains constrained: learned experience can change planning guidance, never the immutable core invariants.

## Development rule

Never mark a capability green without runtime evidence. Failed CI, blocked external services, missing model configuration, and unproven endurance are recorded as incomplete rather than hidden.
