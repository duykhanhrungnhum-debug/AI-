# Autonomous Learning Bootstrap

The Agent is intended to learn continuously through a loop of research, evidence, verification, memory, and experience.

## Scope

This bootstrap does not claim to contain or reproduce the private training data or model weights of any external AI model. Instead it defines a scalable mechanism for acquiring broad knowledge from verified sources and for practicing a capability curriculum.

## Internet learning

The Agent may autonomously search and read publicly reachable Internet sources when instructed to learn. Source provenance, retrieval time, evidence, and verification status must be retained. The user remains the high-level authority who can direct, expand, pause, or stop the Agent's activities.

## Learning loop

1. Identify a knowledge gap.
2. Research relevant sources.
3. Preserve provenance and evidence.
4. Verify claims before promoting them to verified knowledge.
5. Retain conflicts and uncertainty instead of silently overwriting them.
6. Store useful experience from completed and failed tasks.
7. Reuse verified knowledge and prior experience in future planning.
8. Checkpoint progress so learning can resume after interruption.

## Capability curriculum

The training curriculum should exercise instruction following, uncertainty handling, reasoning, mathematics, programming, debugging, document comprehension, planning, tool use, Internet research, source evaluation, verification, error recovery, long-horizon state tracking, conflict handling, self-evaluation, and learning from failure.

## Non-negotiable invariant

Learned knowledge, experiences, Internet content, and training results must never be able to rewrite or disable the Agent's immutable core principles. Learning can improve capability; it cannot redefine the rules that govern honesty, verification, valid instruction following, explicit failure reporting, and recovery behavior.
