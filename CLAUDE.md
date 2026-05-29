# pinn-curvas-sinteticas

Physics-Informed Neural Network for synthetic well log prediction. Predicts RHOB (bulk density) from [GR, RILD, RILM, CNLS, SP] using a physics constraint (RHOB-CNLS relation) in the loss function. Benchmarked with Leave-One-Well-Out on Kraft Prusa field (Kansas Geological Survey). Dataset: 25 clean wells, ~157k records, depth in feet.

## Rules and skills

Rules: `.claude/rules/` (always loaded — code-style, verification, memory-policy, etc.)
Skills: `.claude/skills/` — use `/checkpoint`, `/bitacora`, `/plan-writing`, `/phase-executor`, `/bug-fix`

## Language conventions

- Code and docstrings: **English**
- Comments in code: **English**
- Bitácora (`todo/bitacora-*.md`) and `todo/PLAN.md`: **Spanish**

## Stack

Python 3.11 · PyTorch 2.x · lasio · pandas · numpy · scikit-learn · matplotlib · ruff · mypy · pytest

Hardware: RTX 4080 16GB · WSL2

## Verification

```bash
pytest -q tests/
mypy src/
ruff check src/ tests/
ruff format src/ tests/
```

## Key design decisions

- Inputs: [GR, RILD, RILM, CNLS, SP] → Output: RHOB — architecture 5→64→64→32→1
- Loss: `Loss_total = Loss_datos + λ · Loss_física`; λ=0 must exactly reproduce the baseline MLP
- Physical term: `RHOB_expected = a · CNLS + b` (coefficients calibrated from Kraft Prusa data in Phase 0)
- Evaluation: Leave-One-Well-Out — **no random splits across wells**
- Dataset: 25 wells, Kraft Prusa field (KGS); curves: GR, RILD, RILM, CNLS/CNPOR, SP → RHOB
- Normalization: per-well (decided at EDA phase)
- `src/physics.py` is the most critical module — must have its own unit tests

## Operational source of truth

`todo/PLAN.md` — active phase, pending tasks, decisions log
