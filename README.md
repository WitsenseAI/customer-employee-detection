# Customer vs Employee Detection

> CCTV footage analysis pipeline that detects persons with YOLOv8 and classifies
> them as **Staff** (blue box) or **Customer** (green box) using ResNet50
> cosine-similarity re-identification. Two-tab Gradio UI: Tab 1 to label staff
> by clicking person numbers, Tab 2 for live frame-by-frame detection with CSV export.

![CI](https://github.com/witsenseAI/customer-employee-detection/actions/workflows/ci.yml/badge.svg)

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager

Install uv if you don't have it:
```bash
pip install uv
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/witsenseAI/customer-employee-detection.git
cd customer-employee-detection

# 2. Create and activate virtual environment
uv venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies (including dev tools)
uv pip install -e ".[dev]"

# 4. Set up pre-commit hooks — ONE-TIME per developer machine
pre-commit install

# 5. Copy environment variables
cp .env.example .env
# Open .env and fill in real values (it is gitignored — never committed)
```

---

## How to run

```bash
# CPU (default)
python app.py --device cpu

# NVIDIA GPU
python app.py --device cuda

# Jetson (uses TensorRT yolov8n.engine if present, else falls back to .pt)
python app.py --device jetson
```

Then open `http://127.0.0.1:7860` in your browser.

---

## What happens on every `git commit`

After running `pre-commit install` once, the following checks fire automatically
every time you run `git commit`. If any check fails, **the commit is blocked**
until you fix the issue.

| Check | Tool | What it does |
|-------|------|-------------|
| Format | `ruff format` | Rewrites code to consistent style (spacing, quotes, line length) |
| Lint | `ruff check` | Catches unused imports, bad patterns, basic security issues |
| Type check | `mypy` | Verifies type hints are consistent — catches type mismatches before runtime |
| File hygiene | `pre-commit` | Removes trailing whitespace, ensures files end with newline |
| Secret scan | `pre-commit` | Blocks accidentally committed private keys |
| Large file check | `pre-commit` | Blocks files over 500 KB (e.g. accidental model weights) |

**Example — what a blocked commit looks like:**
```
$ git commit -m "add feature"
ruff format..............................................................Failed
  - hook id: ruff-format
  - files were modified by this hook
  src/mypackage/core.py reformatted
```
Ruff reformats the file in place. Simply `git add` the reformatted file and
commit again — it will pass.

---

## What happens on every Pull Request

When you open a PR, GitHub Actions runs the full CI pipeline automatically.
**A PR cannot be merged unless all jobs pass.**

```
PR opened / new commit pushed
        │
        ▼
┌─────────────────────┐
│  Lint & Type Check  │  ruff format check → ruff lint → mypy
└────────┬────────────┘
         │ passes
         ▼
┌─────────────────────┐
│       Tests         │  pytest on Python 3.11 and 3.12
└────────┬────────────┘
         │ passes
         ▼
      ✅ PR can be merged
```

If any job fails, GitHub blocks the merge button and shows exactly which
check failed and why.

---

## Development commands

```bash
# Run tests
pytest

# Format code
ruff format .

# Lint and auto-fix
ruff check --fix .

# Type check
mypy src/

# Run the full check suite locally (mirrors CI exactly)
ruff format . && ruff check --fix . && mypy src/ && pytest

# Run pre-commit checks manually on all files (without committing)
pre-commit run --all-files
```

---

## Project structure

```
app.py                            ← main Gradio application (single entry point)
requirements.txt                  ← pinned runtime dependencies
src/customer_employee_detection/  ← importable package
tests/            ← pytest tests (mirror src/ structure)
.env.example      ← documents required environment variables (commit this)
.env              ← your local secrets — gitignored, never committed
pyproject.toml    ← project metadata + all tool config (ruff, mypy, pytest)
.pre-commit-config.yaml  ← defines what runs on every git commit
.github/workflows/ci.yml ← defines what runs on every pull request
```

---

## For repo admins — enforcing CI on pull requests

To prevent anyone (including admins) from merging a PR that fails CI,
enable branch protection on `main`:

1. Go to **Settings → Branches → Add rule**
2. Set **Branch name pattern** to `main`
3. Enable the following:
   - ✅ Require status checks to pass before merging
     - Add checks: `Lint & Type Check` and `Tests`
   - ✅ Require branches to be up to date before merging
   - ✅ Do not allow bypassing the above settings
4. Click **Save changes**
