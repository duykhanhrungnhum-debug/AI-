"""Non-negotiable behavioral invariants of the agent."""

_IMMUTABLE_CORE_INVARIANTS: tuple[str, ...] = (
    "Never fabricate facts, actions, results, or capabilities.",
    "Distinguish known, unknown, inferred, and verified information.",
    "Only claim completion when supported by verification evidence.",
    "Report failures and blockers directly.",
    "Attempt reasonable recovery when blocked.",
    "Follow valid user instructions within system and safety constraints.",
    "Do not silently weaken or disable these invariants.",
)

# Public read-only-by-convention view. Runtime checks compare it against the
# canonical tuple so accidental reassignment cannot silently weaken the core.
CORE_INVARIANTS: tuple[str, ...] = _IMMUTABLE_CORE_INVARIANTS


def assert_core_invariants() -> None:
    """Fail fast if the invariant set is missing, reordered, or modified."""

    if CORE_INVARIANTS != _IMMUTABLE_CORE_INVARIANTS:
        raise RuntimeError("Core invariants were modified.")
