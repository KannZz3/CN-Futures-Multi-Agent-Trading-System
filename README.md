# CN Futures Multi-Agent Trading System  
# 中国期货多智能体交易系统

A modular quantitative research framework for Chinese commodity futures trading. The project aims to systematize live-market trading logic into a testable, leakage-safe, and extensible research pipeline.

本项目是一个面向中国商品期货市场的模块化量化研究框架，目标是将实盘交易逻辑系统化为可测试、严格防止信息泄露，并且便于扩展的研究流程。

The framework focuses on identifying medium-frequency directional opportunities in Chinese commodity futures when liquidity, open interest, capital concentration, fundamental pressure, and technical trend begin to align.

该框架主要关注中国商品期货中的中频方向性机会，尤其是当流动性、持仓量、沉淀资金、基本面压力与技术趋势开始共振时，如何识别并验证潜在交易机会。

---

## 1. Author Background  
## 1. 作者背景

This project is informed by the author's live trading experience in Chinese commodity futures markets. Before formalizing the system into a quantitative research framework, the author developed and applied a discretionary trading process based on full-market screening, abnormal open interest, capital accumulation, long-short positioning intensity, and fundamental confirmation of deliverable commodities.

本项目受到作者在中国商品期货市场实盘交易经验的启发。在将交易逻辑正式系统化为量化研究框架之前，作者已经形成并实践了一套主观交易流程，核心包括全市场筛选、异常持仓、沉淀资金、多空博弈强度，以及交割标的基本面的方向性确认。

The figure below shows the author's personal trading results from **2023-06-01 to 2024-06-01**, recorded through a verified brokerage account at **华安期货 (Huaan Futures)**. These results are independent of this research repository and are provided only as background context for the author's practical market experience.

下图展示了作者个人中国期货账户在 **2023-06-01 至 2024-06-01** 期间的实盘交易表现，该记录来自 **华安期货** 验证账户。该实盘结果独立于本研究仓库，仅作为作者实际市场经验的背景说明。

![Live Trading Performance — 2023-06 to 2024-06](Backtest/real_trading_result.jpg)

Key observations:

核心观察：

- Peak net value: approximately **5×** (~400% return)  
  最高净值约 **5倍**，对应约 **400% 收益**

- Peak cumulative profit: approximately **¥920,000 RMB**  
  最高累计盈利约 **92万元人民币**

- Positive equity drift across multiple Chinese commodity futures regimes  
  在多个中国商品期货市场环境中保持正向权益漂移

This repository attempts to translate that practical trading logic into a structured research system. Instead of relying only on discretionary judgment, the framework separates the trading process into market screening, feature selection, factor direction inference, regime filtering, signal construction, and risk-controlled backtesting.

本仓库的目标，是将上述实盘交易逻辑转化为更加结构化、可回测、可复用的研究系统。与单纯依赖主观判断不同，该框架将交易流程拆分为市场筛选、特征选择、因子方向判断、市场状态过滤、信号构建与风险控制回测等模块。

---

## 2. Research Motivation  
## 2. 研究动机

Chinese commodity futures markets exhibit strong regime switching, high noise-to-signal ratios, highly correlated technical factors, inventory and basis-driven structural behavior, and asymmetric long/short dynamics.

中国商品期货市场具有明显的状态切换、高噪声信号比、高度相关的技术因子、受库存与基差驱动的结构性行为，以及多空不对称特征。

Traditional single-factor testing often struggles under nonlinearity, feature interaction, regime dependence, and factor collinearity. This project therefore focuses on out-of-sample predictive robustness rather than in-sample statistical fit.

传统单因子测试在非线性、特征交互、市场状态依赖以及因子共线性较强的环境下往往表现有限。因此，本项目更关注样本外预测稳健性，而不是单纯追求样本内统计拟合。

The core research question is:

核心研究问题是：

```text
How can discretionary futures trading logic be transformed into a systematic,
leakage-safe, and testable quantitative research framework?
```

```text
如何将主观期货交易逻辑转化为系统化、可防止信息泄露、
并且可测试的量化研究框架？
```

---

## 3. System Architecture  
## 3. 系统架构

The system is organized as a multi-agent trading research framework:

本系统采用多智能体交易研究框架：

```text
Technical Agent
        +
Fundamental Agent
        +
Risk Agent
```

```text
技术面 Agent
        +
基本面 Agent
        +
风险控制 Agent
```

The current implementation focuses on single-contract medium-frequency futures trading with strict anti-leakage design and consistent model-comparison conditions.

当前版本主要聚焦于单合约中频期货交易，并强调严格的信息泄露控制以及统一的模型对比条件。

### 3.1 Technical Agent  
### 3.1 技术面 Agent

The Technical Agent transforms adjusted futures OHLCV data into predictive long/short trading signals. It constructs technical and flow-based features, including momentum, trend, volatility, volume, open interest, oscillator, and candlestick-geometry factors.

技术面 Agent 将复权后的期货 OHLCV 数据转化为多空方向交易信号。特征体系包括动量、趋势、波动率、成交量、持仓量、振荡指标以及 K 线结构等技术与资金行为因子。

The framework uses independent long-side and short-side labels:

框架使用相互独立的多头与空头标签：

```text
y_up   = 1(R[t,t+H] >= TH)
y_down = 1(R[t,t+H] <= -TH)
```

This design explicitly treats long alpha and short alpha as asymmetric instead of forcing them into a single symmetric signal.

这一设计明确假设期货市场中的多头 alpha 与空头 alpha 并不完全对称，因此不强行使用单一对称信号。

### 3.2 Fundamental Agent  
### 3.2 基本面 Agent

The Fundamental Agent acts as a commodity-specific regime filter rather than a direct price predictor. Current implementations focus on warehouse receipt factors, spot basis factors, weekly inventory pressure, and basis regime gating.

基本面 Agent 在系统中主要作为商品特定的市场状态过滤器，而不是直接价格预测器。当前实现重点包括仓单因子、现货基差因子、周度库存压力以及基差状态过滤。

The framework assumes that Chinese commodity futures are strongly influenced by inventory structure, delivery pressure, and basis behavior. Fundamental factors are lagged by one week before merging into daily trading logic to prevent information leakage.

该框架认为，中国商品期货价格高度受到库存结构、交割压力和基差状态的影响。因此，基本面因子在合并进入日频交易逻辑前，都会滞后一周，以避免未来信息泄露。

### 3.3 Risk Agent  
### 3.3 风险控制 Agent

The Risk Agent applies weekly regime-aware filters before trade execution. It includes weekly trend filtering, weekly volatility regime classification, crowded long/short positioning filters, and one-week lagged weekly information to avoid look-ahead bias.

风险控制 Agent 在交易执行前加入周频市场状态过滤，主要包括周线趋势过滤、周度波动率状态识别、拥挤多头/空头过滤，以及周频信息整体滞后一周以避免前视偏差。

The backtesting engine further includes capital-based position sizing, leverage control, stop-loss, take-profit, maximum holding period, weekly gating, and fundamental gating.

回测引擎进一步包含基于资金的仓位控制、杠杆控制、止损、止盈、最大持仓周期、周频过滤以及基本面过滤。

---

## 4. Core Research Pipeline  
## 4. 核心研究流程

The full research pipeline is:

完整研究流程如下：

```text
Adjusted Daily Futures Data
        ↓
Feature Engineering
        ↓
Forward Return Label Construction
        ↓
ML Feature Selection
        ↓
Factor Sign Detection
        ↓
Long / Short Split-Score Construction
        ↓
Weekly / Fundamental Filtering
        ↓
Risk-Controlled Backtesting
```

```text
复权日频期货数据
        ↓
特征工程
        ↓
未来收益标签构建
        ↓
机器学习特征选择
        ↓
因子方向判断
        ↓
多头 / 空头拆分打分
        ↓
周频 / 基本面过滤
        ↓
风险控制回测
```

---

## 5. Machine Learning Feature Selection  
## 5. 机器学习特征选择

The project emphasizes consistent cross-model feature selection under identical research conditions. Supported models include:

本项目重点研究在相同研究条件下，不同机器学习模型对期货因子的筛选差异。当前支持模型包括：

- Elastic Net
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- MLP

All models share the same outer selection logic:

所有模型使用统一的外层筛选逻辑：

```text
Rolling OOS validation
→ OOS permutation importance
→ importance aggregation
→ single-factor OOS AUC
→ final feature selection
```

```text
滚动样本外验证
→ 样本外 permutation importance
→ 重要性聚合
→ 单因子样本外 AUC
→ 最终特征选择
```

This allows different model architectures to be compared under the same preprocessing, validation, feature universe, score construction, filtering logic, and backtesting engine.

这样可以在相同的数据预处理、验证方式、特征空间、打分逻辑、过滤规则和回测引擎下，比较不同模型结构的因子选择行为。

---

## 6. Factor Sign and Score Construction  
## 6. 因子方向与信号构建

After feature selection, each factor undergoes directional sign inference to determine whether it contributes positively or negatively to long-side or short-side prediction.

完成特征选择后，每个因子都会进行方向性判断，用于确认该因子对多头或空头预测是正向贡献、负向贡献，还是应当剔除。

The sign detection process combines quantile bucketing, IC sign, Spearman monotonicity, OLS regression direction, and bootstrap stability, producing a final signal of +1 / −1 / 0.

方向判断结合分位数组、IC 符号、Spearman 单调性、OLS 回归方向和 bootstrap 稳定性，最终输出 +1 / −1 / 0。

Selected factors are then transformed into separate long and short scores through train-only normalization, robust z-score transformation, inferred factor direction, and correlation-filtered aggregation.

通过筛选后的因子会进一步构建为独立的多头分数和空头分数，流程包括仅基于训练集的标准化、稳健 z-score 转换、因子方向加权，以及相关性过滤后的聚合。

This design intentionally avoids a single symmetric score because long-side and short-side futures behavior often differ in both formation mechanism and market structure.

该设计刻意避免使用单一对称分数，因为期货市场中的多头与空头行为在形成机制和市场结构上往往并不相同。

---

## 7. Leakage-Safe Research Design  
## 7. 防信息泄露研究设计

The framework prioritizes out-of-sample robustness over in-sample fit. A major design principle is strict leakage prevention, including:

该框架优先关注样本外稳健性，而不是样本内拟合。核心设计原则之一是严格防止信息泄露，包括：

- Chronological train/test split  
  按时间顺序划分训练集与测试集

- Train-only normalization  
  仅基于训练集进行标准化

- Rolling out-of-sample validation  
  滚动样本外验证

- Shifted weekly and fundamental filters  
  周频与基本面过滤因子滞后处理

- Forward-return-safe labels  
  避免未来收益标签泄露

- OOS permutation importance  
  样本外 permutation importance

- Train-only z-score statistics  
  仅使用训练集统计量构建 z-score

The framework intentionally avoids full-sample normalization, future regime contamination, and merge-asof leakage that are commonly found in retail backtests.

该框架刻意避免散户回测中常见的全样本标准化、未来市场状态污染，以及 merge_asof 使用不当导致的信息泄露问题。

---

## 8. Trading Universe and Data  
## 8. 交易品种与数据

Current research covers major Chinese commodity futures contracts:

当前研究覆盖主要中国商品期货合约：

| Exchange | Contracts |
|---|---|
| CZCE | TA / MA / FG / RM / SR / CF |
| SHFE | AG / AU / CU / HC / RU |

| 交易所 | 合约 |
|---|---|
| 郑商所 | TA / MA / FG / RM / SR / CF |
| 上期所 | AG / AU / CU / HC / RU |

CZCE contracts carry fuller fundamental data coverage, including warehouse receipts, spot basis, and roll yield. SHFE contracts are mainly used for technical-signal research.

郑商所品种具有更完整的基本面数据覆盖，包括仓单、现货基差和展期收益等。上期所品种主要用于技术信号研究。

Contracts were selected based on liquidity, historical data availability, inventory/basis relevance, and representative industrial-chain behavior.

合约选择主要基于流动性、历史数据可得性、库存/基差相关性，以及是否具有代表性的产业链行为。

---

## 9. Project Structure  
## 9. 项目结构

```text
CN-Futures-Multi-Agent-Trading-System/

├── Backtest/
│   ├── backtest_weekly_func.py
│   └── backtest_weekly_fundamental_func.py
│
├── Price & Volume Data/
├── Fundmental Data/
│   └── Processed Data/
├── Macro Data/
│   ├── Train/
│   └── Test/
│
├── Features/
├── Filters/
├── Score/
│
├── Single-Contract Full Pipeline--Technical + Weekly Filters.ipynb
├── Single-Contract Full Pipeline--Technical + Fundamental Factors.ipynb
└── README.md
```

The repository is organized around the full research workflow, including data preparation, feature construction, model-based factor selection, weekly/fundamental filtering, score construction, and backtesting.

该仓库围绕完整研究流程组织，包括数据准备、特征构建、基于模型的因子选择、周频/基本面过滤、信号打分以及回测模块。

---

## 10. Current Focus and Future Directions  
## 10. 当前研究重点与未来方向

Current research focuses on:

当前研究重点包括：

- Cross-model factor comparison  
  跨模型因子选择对比

- Long/short asymmetry analysis  
  多空不对称性分析

- Weekly and fundamental regime filtering  
  周频与基本面市场状态过滤

- Feature robustness analysis  
  特征稳健性分析

- Contract-level transferability  
  合约层面的可迁移性研究

- Fundamental gating effectiveness  
  基本面过滤有效性验证

Potential future extensions include:

未来可能扩展方向包括：

- Multi-contract portfolio optimization  
  多合约组合优化

- Cross-sectional futures selection  
  横截面期货品种筛选

- Portfolio-level risk allocation  
  组合层面风险分配

- Regime clustering  
  市场状态聚类

- Multi-frequency integration  
  多频率信号整合

- Reinforcement-learning overlays  
  强化学习模块叠加

- LLM-assisted macro/fundamental agents  
  LLM 辅助宏观与基本面 Agent

---

## 11. Related Research  
## 11. 相关研究

The project is conceptually related to recent work on multi-agent trading systems, AI-assisted quantitative research pipelines, ML-based factor selection, and regime-aware trading architectures.

本项目在研究方向上与多智能体交易系统、AI 辅助量化研究流程、机器学习因子选择以及市场状态感知交易架构相关。

Related references include:

相关参考包括：

- [TradingAgents](https://arxiv.org/abs/2412.20138)
- [QuantAgents](https://arxiv.org/abs/2510.04643)

These references are used as research-direction inspiration rather than direct reproductions.

这些参考主要用于研究方向启发，并非直接复现。

---

## 12. Disclaimer  
## 12. 免责声明

This repository is intended solely for quantitative research and educational purposes and does not constitute financial advice.

本仓库仅用于量化研究与教育目的，不构成任何投资建议。
