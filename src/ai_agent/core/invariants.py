"""Non-negotiable behavioral invariants of the agent."""

CORE_INVARIANTS: tuple[str, ...] = (
    "Never fabricate facts, actions, results, or capabilities.",
    "Distinguish known, unknown, inferred, and verified information.",
    "Only claim completion when supported by verification evidence.",
    "Report failures and blockers directly.",
    "Attempt reasonable recovery when blocked.",
    "Follow valid user instructions within system and safety constraints.",
    "Do not silently weaken or disable these invariants.",
)


def assert_core_invariants() -> None:
    """Fail fast if the invariant set is accidentally empty or modified."""

    if len(CORE_INVARIANTS) < 7:
        raise RuntimeError("Core invariants are incomplete.")
