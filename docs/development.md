# Development

Run all commands below from the SEmoEdit repository root.

Install the Git hook to run Ruff's safe lint fixes and formatting automatically on staged Python files at each commit:

```bash
uvx --from pre-commit==4.2.0 pre-commit install
uvx --from pre-commit==4.2.0 pre-commit run --all-files
```

If a hook changes files, review and stage those changes, then commit again. Ruff's version and rules are fixed in `ruff.toml`; third-party backbones are excluded. GitHub Actions checks lint and formatting on every push and pull request without downloading model weights or installing backbone dependencies.

Run the CPU inference and CLI tests in a backbone environment:

```bash
conda run -n f5-tts python -m unittest discover -s tests -v
```

