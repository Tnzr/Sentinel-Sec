# Contributing to Sentinel-Sec

Thanks for your interest in contributing!

- Use issues to propose features and report bugs with clear steps to reproduce and environment info.
- For PRs:
  - Fork the repo and create a feature branch.
  - Keep changes focused and include tests or reproducible examples when possible.
  - Follow existing code style and add docstrings where helpful.
  - Ensure `streamlit run src/sentinel.py` starts and basic flows work.

## Development setup
- Python 3.10 recommended
- Create a virtualenv and install deps:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements_pipreqs.txt`

## Code quality
- Prefer explicit Streamlit widget keys to avoid duplicate key errors.
- Cache heavy resources with `st.cache_resource`.
- Keep model and DB initialization before usage within a rerun-safe pattern.

## Git hygiene
- Write clear commit messages.
- Avoid committing large binaries. The `.gitignore` is configured to exclude models, databases, and outputs.

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
