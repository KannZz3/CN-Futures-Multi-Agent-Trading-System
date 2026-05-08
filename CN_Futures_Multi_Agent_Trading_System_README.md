# CN Futures Multi-Agent Trading System

A modular quantitative research framework for Chinese commodity futures trading, focused on:

- Machine-learning-based factor selection
- Leakage-safe medium-frequency research
- Multi-agent trading architecture
- Technical + fundamental signal integration
- Weekly regime-aware risk filtering
- Robust out-of-sample validation

---

## System Architecture

```text
Technical Agent
        +
Fundamental Agent
        +
Risk Agent
```

Designed for systematic CTA-style futures research.

The current implementation emphasizes:

> Single-contract medium-frequency futures trading  
> with strict anti-leakage design and model-comparison consistency.

---

# Research Motivation

Chinese commodity futures exhibit:

- Strong regime switching
- High noise-to-signal ratio
- Highly correlated technical factors
- Inventory and basis-driven structural behavior
- Asymmetric long/short dynamics

Traditional single-factor statistical testing often struggles under:

```text
Nonlinearity
+
Feature interaction
+
Regime dependence
+
High factor collinearity
```

This project instead focuses on:

> Out-of-sample predictive robustness

through unified ML-based factor selection and leakage-safe backtesting.

---

# Related Research

The project is conceptually related to recent work on:

- Multi-agent trading systems
- AI-assisted quantitative research pipelines
- ML-based factor selection
- Regime-aware trading architectures

Related references include:

- [TradingAgents](https://arxiv.org/abs/2412.20138)
- [QuantAgents](https://arxiv.org/abs/2510.04643)

---

# Disclaimer

This repository is intended solely for quantitative research and educational purposes and does not constitute financial advice.
