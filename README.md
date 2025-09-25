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

- Python 3.10+
- Node.js 18+
- 8GB+ RAM (CPU-only training supported)

### Setup

```bash
# Backend setup
cd api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../web
npm install

# Run development servers
cd ../api && uvicorn main:app --reload --port 8000
cd ../web && npm run dev
```

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
