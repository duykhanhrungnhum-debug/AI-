# Translation learning cycle — 2026-09-27

## Evidence reviewed

- Netflix Vietnamese Timed Text Style Guide: Vietnamese subtitles are limited to 42 characters per line; adult reading speed is up to 17 characters/second. Repeated words/phrases by one speaker are normally translated once, except when repetition changes tone/mood or is rhythmically meaningful.
  Source: https://partnerhelp.netflixstudios.com/hc/en-us/articles/220447048-Vietnamese-Timed-Text-Style-Guide
- Netflix Timed Text General Requirements: subtitle events should normally last 5/6 second to 7 seconds, with at most two lines.
  Source: https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements
- MQM: translation QA should classify errors by type such as Accuracy, Fluency, Terminology and Style, with severity used for measurable quality gates.
  Source: https://www.themqm.org/
- Atwany et al., ACL Findings 2025: ASR hallucination rate is not captured reliably by WER alone; noise/distribution shift increases hallucination, so Hidden Beyond ASR regression must track hallucination-like insertions separately.
  Source: https://aclanthology.org/2025.findings-acl.1190/

## Hidden Beyond lessons applied

1. Timing fitness must not rely only on Vietnamese word count. Add a character-per-second signal aligned with the Vietnamese 17 CPS reference, while preserving the existing TTS word-budget gate.
2. Repetition needs two classes: accidental MT/ASR loops versus expressive source repetition. Never automatically erase source-supported emotional repetition.
3. ASR regression must measure hallucinated insertions independently from transcript edit distance/WER.
4. Promotion remains conservative: these new checks are challenger/regression evidence. They do not replace the last VERIFIED stable until runtime/model regression passes.

## GPU policy

This cycle changes CPU-side QA/test logic only. No GPU rerun is justified until the CPU regression gates are installed and green.
