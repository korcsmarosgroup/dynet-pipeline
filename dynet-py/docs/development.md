# Development

## Local Setup

```bash
git clone <your-repo-url>
cd dynet-py
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,docs]"
```

## Run Tests

```bash
python -m pytest -q
```
