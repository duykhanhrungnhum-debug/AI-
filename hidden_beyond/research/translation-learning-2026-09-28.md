# Translation learning cycle — 2026-09-28

## New evidence reviewed
- ACL Findings 2025 DoCIA: document-level context can be integrated into ASR refinement, MT, and MT refinement; speech translation has extra discourse difficulty from ASR noise. https://aclanthology.org/2025.findings-acl.771/
- WMT 2025 DuTerm: terminology-aware translation benefits from a two-stage design and context-driven post-editing; rigid term forcing can trade off against overall translation quality. https://aclanthology.org/2025.wmt-1.112/
- TermTrends 2025 LegISTyr: raw term insertion is not enough; homonym/context disambiguation and fluency must be tested separately. https://aclanthology.org/2025.termtrends-1.1/
- Netflix Vietnamese Timed Text Style Guide: Vietnamese adult subtitle reference is up to 17 CPS, max 42 characters/line, and Chinese/proper names need controlled treatment. https://partnerhelp.netflixstudios.com/hc/en-us/articles/220447048-Vietnamese-Timed-Text-Style-Guide

## Hidden Beyond lessons applied
1. Context memory must be tested as a behavior, not assumed from prompt text. Multi-turn challenger cases must require a prior line to resolve pronouns, titles, names or term sense.
2. Terminology QA now separates terminology adherence from contextual correctness. A forced glossary term is not automatically a pass when the source term is ambiguous in context.
3. ASR-noise tests must verify whether document context repairs a plausible corrupted transcript without inventing unsupported content.
4. Audiovisual fitness keeps both TTS word-budget and Vietnamese CPS signals; subtitle reference limits are evidence, not a license to distort dubbed dialogue.
5. Promotion remains conservative: CPU structural tests may validate the evaluator, but stable translation cannot be promoted without runtime/model challenger evidence.

## GPU decision
No GPU training is justified by this cycle. First install and pass the CPU evaluator/challenger structure; only then run a small model inference challenger if it can change the stable decision.
