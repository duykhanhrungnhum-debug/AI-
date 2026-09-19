# Kaggle GPU smoke worker

This private Kaggle kernel is infrastructure for AI-'s on-demand GPU worker.

It performs only a hardware smoke test: it queries the assigned NVIDIA GPU with
`nvidia-smi` and writes `gpu_report.json` to Kaggle output. It does not call
OpenAI, Gemini, Groq, OpenArt, or any other external AI provider.

The GitHub Actions workflow dynamically replaces the placeholder Kaggle owner
with the username resolved from `KAGGLE_API_TOKEN`, pushes the private kernel,
waits for completion, downloads the report, and verifies that a real GPU was
present.

The smoke workflow runs automatically only when these GPU-smoke files change,
and can also be started manually. This avoids consuming Kaggle GPU quota on
ordinary AI- commits.
