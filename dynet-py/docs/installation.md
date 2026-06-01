# Installation

## Requirements

- Python 3.9+

## Install from Source (recommended for this repo)

```bash
git clone <your-repo-url>
cd dynet-py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Install with Development Dependencies

```bash
python -m pip install -e ".[dev]"
```

## Install with Documentation Dependencies

```bash
python -m pip install -e ".[docs]"
```

## Verify Installation

```bash
python -c "import dynet_py, sys; print(dynet_py.__version__); print(sys.executable)"
```

## Build Distribution Artifacts

```bash
python -m pip install build
python -m build
```

This creates:

- source distribution in `dist/*.tar.gz`
- wheel in `dist/*.whl`

## Troubleshooting: `ModuleNotFoundError: No module named dynet_py`

This usually means install and runtime are using different Python interpreters.

Check alignment:

```bash
python3 -m pip --version
python3 -c "import sys; print(sys.executable)"
```

If versions/paths differ from your runtime, activate your virtualenv and reinstall with:

```bash
python -m pip install -e .
```
