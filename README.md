# TGFL Market Scenario Simulator

## Transformer Based Generative Federated Learning Models to Simulate Market Scenarios

A zero cost MVP demonstrating privacy preserving federated learning for financial market scenario generation using lightweight Transformers.

## Core Value Proposition

Generate realistic market scenarios via federated learning without centralizing sensitive data, enabling stress testing and risk analysis while preserving privacy.

## Architecture

```
├── web/          # Next.js frontend (TypeScript)
├── api/          # FastAPI backend (Python)
├── ml/           # ML models and notebooks (Python)
├── shared/       # Common schemas and types
├── data/         # Cached datasets and artifacts
└── docs/         # Documentation and demo materials
```

## Quick Start

### Prerequisites

- Python 3.11 (recommended for CPU PyTorch wheels)
- Node.js 18+
- 8GB+ RAM (CPU-only training supported)

### Setup

```bash
# Backend setup (from repo root)
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional: install full ML extras inside the venv if you have compatible hardware
# pip install --index-url https://download.pytorch.org/whl/cpu torch

# Frontend setup
cd web
npm install

# Run development servers (in separate shells)
cd api && uvicorn main:app --reload --port 8000
cd ../web && npm run dev
```

## Project summary and scope

One-line summary

Lightweight, CPU-friendly framework for transformer-based market scenario generation with an option for federated simulation — designed for reproducible demos, developer ergonomics, and privacy-aware experimentation.

Project scope and goals

- Provide a minimal, reproducible MVP to generate synthetic market scenarios using small transformer models.
- Support both centralized (baseline) training and a quick federated simulation mode for demoing privacy-preserving workflows.
- Optimize for CPU-only execution on modest hardware (8GB RAM) so contributors can run experiments without GPUs.
- Emphasize reproducibility, editor/CI friendliness, and safe fallbacks when heavy ML dependencies are not available.

Key contributions

- Baseline transformer trainer with CPU‑friendly configs (tiny models, small batches, short sequences).
- Federated quick-simulation orchestrator to demo client/server aggregation behavior without centralizing raw data.
- Evaluation utilities computing KS-tests, ACF similarity, volatility alignment, and moment statistics.
- API (FastAPI) for scheduling background runs and persisting incremental progress to JSON artifacts.
- Notebooks and scripts (including a deterministic golden-run) for demonstration and reproducible artifacts.
- Engineering fixes: robust type hints, numpy/pandas coercions, lazy imports and dev-friendly fallbacks.

Typical workflows

- Baseline training (notebook): open `ml/notebooks/02_baseline_model.executed.ipynb`, follow cells to generate synthetic data, train tiny transformer, generate scenarios, and evaluate.
- Federated quick demo: use API or CLI orchestrator to run `run_quick_simulation(clients=3, rounds=3, quick=True)` — writes incremental progress to `data/results/<run_id>.json`.
- Golden deterministic run: `python scripts/golden_run.py` produces reproducible examples and saved artifacts in `data/results/golden/`.

API (high level)

- POST /train/start — schedule a training job (baseline or federated). Returns run_id.
- GET /train/status/{run_id} — poll status (scheduled → running → completed) and read progress.
- Results persisted to `data/results/<run_id>.json` (intermediate progress + final summary).

Reproducibility and artifacts

- Primary artifacts: `data/models/` (checkpoints), `data/results/<run_id>.json` (progress + metrics), `data/results/golden/` (deterministic demo outputs).
- Notebook and scripts include explicit steps to convert pandas Series to numpy arrays (`ml/utils.to_numpy_array`) to avoid type mismatches and ensure deterministic evaluation.
- Recommended environment: Python 3.11 venv (.venv311). If PyTorch wheel for your Python version is not available, use the included mock fallbacks (the system will run in demo mode without real training).

MVP success criteria

- Primary: Generated scenarios produce KS test p-value ≥ 0.05 vs historical returns (mean across scenarios).
- Secondary: ACF similarity and rolling volatility alignment metrics exceed configured thresholds.
- Demo target: an end-to-end federated simulation and scenario generation within ≈10 minutes on CPU for the quick mode.

Testing & CI

- CI is configured to run lightweight checks (py_compile, import smoke tests, and selected unit tests) on pull requests.
- Heavy tests requiring PyTorch/Scipy are run via a manual/optional CI job to avoid blocking contributions on machines without GPUs.
- Developers should run `pytest -q` in `.venv311` after installing `requirements-dev.txt` for full dev testing.

Development notes & known issues

- Python/PyTorch compatibility: prebuilt PyTorch wheels target specific Python versions. If import fails, recreate venv using Python 3.11 or use conda.
- NumPy ABI: if you see ABI warnings after installing torch, pin NumPy `<2.0` (e.g., `numpy>=1.25,<2.0`) to match binary builds.
- Editor typing: fixes were applied (TYPE_CHECKING guarded imports, to_numpy_array helper, np.array coercions) to reduce Pylance/mypy warnings.
- Large assets: `data/` is gitignored; use release assets or Git LFS for large shared artifacts.

Ethics, privacy & licensing

- Federated mode aims to reduce centralized data transfer, but does NOT guarantee privacy by itself. For production, integrate secure aggregation, differential privacy, or other privacy-preserving protocols.
- Generated scenarios are synthetic and intended for testing/education — not financial advice or real predictions.
- License: MIT (educational/research use). Acknowledge third‑party libraries (PyTorch, NumPy, SciPy, pandas, FastAPI, etc.) under their respective licenses.

Business discovery — problems solved

- Generate realistic synthetic market scenarios for stress-testing and strategy validation without centralizing raw sensitive data. This enables organizations to test models and capital allocations without sharing proprietary historical series.
- Provide a reproducible, CPU-friendly pipeline for transformer-based sequence modeling that runs on modest hardware so teams can prototype without GPUs.
- Offer an easy-to-run demo for federated learning workflows to evaluate decentralized training and aggregation behavior in a local or constrained environment.
- Standardize evaluation with statistical metrics (KS test, ACF similarity, volatility alignment) and persistable run artifacts for auditability and reproducibility.
- Improve developer ergonomics by handling dependency fallbacks, type-safety issues, and pandas→numpy coercions to avoid runtime/type-checker issues.

Target clients

- Quantitative researchers and ML engineers at hedge funds and prop trading firms who need scenario generation but cannot centralize raw data.
- Risk teams in banks and fintechs requiring stress-test scenarios while minimizing data sharing and maintaining audit trails.
- Fintech startups exploring federated approaches to collaborate on model improvements without exchanging raw datasets.
- Academic researchers and students who need reproducible, low-cost infrastructure for experiments and teaching.

Key bottlenecks

- Data privacy concerns that prevent aggregating raw time-series into a central location for training.
- Compute resource constraints (no GPU), limiting model size and the feasibility of standard research pipelines.
- Dependency and binary-compatibility issues (PyTorch wheel ↔ Python version, NumPy ABI) which hinder reproducible, out-of-the-box runs.
- Evaluation challenges: ensuring metrics are robust, deterministic, and accept consistent input types (pandas vs numpy arrays).
- Federated orchestration complexity, including client heterogeneity, aggregation protocol correctness, and safe background-task handling.

Existing alternatives

- Centralized scenario-generation pipelines built on PyTorch/TF and custom simulation code; accurate but require data centralization.
- Federated learning frameworks (Flower, TensorFlow Federated) that provide robust primitives but often assume heavier infra and more complex deployment.
- Commercial stress-testing and scenario-generation platforms; turnkey but expensive and closed-source.
- Ad-hoc research notebooks and scripts lacking engineering features such as background scheduling, persistent runs, and standard evaluation.

Single high‑impact problem we solve

Enable statistically realistic market scenario generation on commodity hardware while preserving data locality/privacy via a lightweight federated simulation pipeline.

Early adopter profile and workflow

- Description: a quant researcher or risk engineer at a mid-sized trading firm or fintech who wants to prototype scenario generation and evaluate decentralized training without transferring raw datasets.
- Current workflow: collect historical series locally, perform preprocessing and local training in Jupyter or scripts, export CSVs for centralized analysis, and manually aggregate metrics offline. Federated experiments are rare because of infra and tooling gaps.

Minimum viable feature set

- Tiny transformer baseline trainer (CPU-optimized) with configurable sequence length and small batch sizes.
- Quick federated-simulation orchestrator (3 clients, configurable rounds) that runs locally and simulates client aggregation.
- Evaluation module computing KS-test, ACF similarity, volatility comparison, and basic moments.
- API endpoint or simple CLI to schedule runs and persist incremental progress and final summaries to `data/results/<run_id>.json`.
- Reproducibility helpers: deterministic golden-run script, `ml/utils.to_numpy_array`, and clear environment instructions (Python 3.11 venv).

Primary user action that delivers value

Start a baseline or federated run (API POST `/train/start` or a notebook call) and inspect generated scenarios and the persisted JSON summary to validate scenario realism.

How we are better

A reproducible, developer-friendly demoable federated learning pipeline that runs on CPUs, includes built-in evaluation, fault-tolerant fallbacks for missing ML deps, and background scheduling — lowering the barrier for privacy-aware scenario generation compared to heavier production systems.

Critical hypothesis the MVP tests

A small transformer trained in a federated quick-simulation (CPU, small model) can generate scenarios that are statistically indistinguishable from historical returns as measured by mean KS-test p-value ≥ 0.05 within reasonable compute time.

Primary quantitative metric

Mean KS-test p-value across generated scenarios versus historical returns (target: >= 0.05).

Primary technical risks and dependencies

- PyTorch binary/wheel compatibility with the developer's Python version — if incompatible, real training is blocked and users fall back to mocked/demo paths.
- NumPy ABI mismatches with compiled extensions (e.g., PyTorch) — mismatches can cause runtime crashes; pinning required.
- Missing heavy ML libraries (PyTorch, SciPy) in CI or contributor environments — leads to broken or skipped tests.
- Federated orchestration correctness and background-task handling — risk of incorrect progress persistence or concurrency bugs.
- Evaluation library compatibility (SciPy, statsmodels) — mismatched return types or API changes can break evaluation scripts.

Single most important non-functional requirement

Data privacy and security: raw sensitive data must remain local to clients in federated mode (no central storage of raw series) and the system should persist only sanitized metrics and artifacts.

Out-of-scope for the MVP

- Full production-grade GUI/dashboard beyond basic demo plots.
- Built-in secure aggregation or differential privacy guarantees — these are planned follow-ups for production deployments.
- Large-scale distributed GPU training and automated hyperparameter search.
- Enterprise connectors and full ETL pipelines (MVP accepts CSV/pandas inputs).

External dependencies and associated risks

- PyTorch — Binary wheels are platform-/Python-version-specific; lack of a compatible wheel prevents real training and forces mock fallbacks.
- NumPy — ABI mismatches between NumPy and compiled extensions can cause runtime failures; pin to compatible versions.
- Flower (federated learning) — Orchestration layer compatibility and heavy deps; missing/incompatible Flower may require simplified simulators.
- SciPy / statsmodels — Required for statistical tests; absence prevents some evaluations or forces approximations.
- FastAPI / Uvicorn — Server runtime; misconfiguration can break background tasks and the API surface.
- pandas — ExtensionArray types must be coerced to numpy arrays; failing to do so causes type/runtime mismatches.
- Next.js (web) — Frontend build dependencies may block contributors who only want to run backend/ML demos.

Validation: each topic above is stated clearly and with actionable implications; if you'd like, I can now commit and push this README change and then run the smoke test to validate one quick federated run.

## MVP Success Metrics

- **Primary**: KS test p-value ≥ 0.05 between generated and historical returns
- **Secondary**: ACF similarity and rolling volatility alignment
- **Demo**: 10-minute end-to-end federated training and scenario generation

## Hardware Optimization

Optimized for CPU only training on modest hardware:

- Model size: <1M parameters
- Sequence length: ≤256 tokens
- Federated rounds: 5-10 maximum
- Precomputed artifacts for demo reliability

## Tech Stack

- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI, SQLModel, Pydantic v2
- **ML**: PyTorch, Flower (FL), pandas, scikit-learn
- **Storage**: SQLite, Redis (optional)
- **Deployment**: Local-first, optional Vercel/Railway

## License

MIT License - Educational/Research Use
