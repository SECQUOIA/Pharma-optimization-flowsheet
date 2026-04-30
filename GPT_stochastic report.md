# Incorporating Uncertainty into the Pharma Site-Selection & Production Model

## Abstract

This report outlines how to extend the existing (deterministic) pharmaceutical manufacturing model to account for three key sources of uncertainty: demand uncertainty, production yield uncertainty, and capacity disruptions. For each source, we describe how it enters the model structure, which distributions are suitable, which modeling tools and approaches to use (stochastic programming, robust optimization, etc.), and how this fits into a scenario-based Pyomo implementation.

## Contents

1. [Modeling Context (Deterministic Baseline)](#1-modeling-context-deterministic-baseline)
2. [Demand Uncertainty](#2-demand-uncertainty)
3. [Production Yield Uncertainty](#3-production-yield-uncertainty)
4. [Capacity Disruptions (Downtime and Regulatory Shocks)](#4-capacity-disruptions-downtime-and-regulatory-shocks)
5. [Summary of Tools and Distributions](#5-summary-of-tools-and-distributions)
6. [Integration Blueprint (High-Level)](#6-integration-blueprint-high-level)

---

## 1 Modeling Context (Deterministic Baseline)

In the baseline model you effectively have:

- **Site/vendor selection**: binary variables deciding where each stage is processed.
- **Production capacities**: parameters representing how much each site can produce if selected.
- **Flows/transport**: variables and costs linking stages and sites.
- **Demand**: a single scalar or a small set of deterministic demand values that must be met.

The deterministic core typically looks like:

$$\min \; \text{Fixed Costs} + \text{Production Costs} + \text{Transport Costs} \tag{1}$$

subject to:

$$\sum_{i,k} \text{capacity}_{ik}\, x_{ik} \;\geq\; D \tag{2}$$

and other capacity, flow-balance, and assignment constraints.

We will turn selected parameters ($D$, yield, capacity) into **random variables**, approximated via **scenarios**.

---

## 2 Demand Uncertainty

### 2.1 Role in the Model

Demand $D$ determines how much capacity the network must cover. In a two-stage setting:

- **First stage**: Choose sites, assign stages to sites (large, strategic decisions).
- **Second stage**: Adjust operational flows and production to meet realized demand in each scenario.

### 2.2 Recommended Distributions

For **total annual demand** (by product or region), reasonable options are:

#### 2.2.1 Truncated Normal (Continuous, Approximately Symmetric)

Let

$$X \sim \mathcal{N}(\mu, \sigma^2), \tag{3}$$

with:

- $\mu$ = forecast demand
- $\sigma = \text{cv} \cdot \mu$, where $\text{cv}$ is a coefficient of variation (e.g., 0.2 for 20% relative standard deviation).

Then define demand as:

$$D = \max(X, d_{\min}), \tag{4}$$

where $d_{\min}$ is a small positive fraction of $\mu$ (e.g., 5%) to avoid negative or implausibly low demand.

*Use this when:*

- Forecast errors are relatively symmetric.
- A central limit approximation is reasonable.

#### 2.2.2 Lognormal (Continuous, Right-Skewed)

Assume

$$\log D \sim \mathcal{N}(\mu_{\log}, \sigma_{\log}^2). \tag{5}$$

This ensures strictly positive demand and allows **right-tail risk** (e.g., surges in uptake) to be captured.

*Use this when:*

- Demand is strictly positive.
- Upside spikes are more likely than symmetric drops.

#### 2.2.3 Discrete Low/Base/High Scenario Set

For interpretability, you can use a small set such as:

- Scenario 1 (low): $D = 0.8\mu$
- Scenario 2 (base): $D = \mu$
- Scenario 3 (high): $D = 1.3\mu$

Scenario probabilities are calibrated from historical data or expert judgment.

*Use this when:*

- You need interpretability and small models.
- Data are limited but expert input is available.

**Practical Default Choice:** Use a **truncated Normal** distribution or a **3–5-point discrete approximation** to a truncated Normal. This keeps coding straightforward and the narrative easy to explain.

### 2.3 Tools and Modeling Approach

#### 2.3.1 Two-Stage Stochastic Programming (Scenario-Based Demand)

- Define a scenario set $\Omega$, each with demand $D(\omega)$ and probability $p_\omega$.
- First-stage decisions (site opening, stage–site assignment) are **scenario-independent**.
- Second-stage decisions (flows, production quantities) are **scenario-dependent**.

#### 2.3.2 Implementation Tools

- **Pyomo** for core MILP formulation.
- Deterministic equivalent formulation:
  - Add a set `SCENARIOS` in Pyomo.
  - Treat demand as `demand[ω]`.
  - Index relevant constraints and cost terms by $\omega$.

#### 2.3.3 Solution Techniques

- Direct MILP solve for modest numbers of scenarios.
- Decomposition methods (e.g., progressive hedging via `mpi-sppy`) when the number of scenarios is large.

#### 2.3.4 Alternative: Robust Optimization

- Model demand in an interval $[\underline{D}, \overline{D}]$.
- Require feasibility under worst-case demand (or under a restricted uncertainty budget).
- This is still implemented in Pyomo but with manually constructed robust counterpart constraints.

---

## 3 Production Yield Uncertainty

### 3.1 Role in the Model

Let $Y_{ik}$ denote the **yield** at site $i$, stage $k$: the fraction of input material that becomes usable output.

Effective capacity becomes:

$$\text{effective\_capacity}_{ik}(\omega) = \text{nominal\_capacity}_{ik} \cdot Y_{ik}(\omega). \tag{6}$$

Lower yields:

- Reduce effective capacity.
- Increase required input amounts.
- May create bottlenecks in specific stages/sites.

### 3.2 Recommended Distributions

Yields are fractions in $(0, 1)$ (or near 1 with rare over-yield events). Natural distributions are:

#### 3.2.1 Beta Distribution (Bounded on $[0, 1]$)

$$Y_{ik} \sim \text{Beta}(\alpha_{ik}, \beta_{ik}), \tag{7}$$

with

$$\mathbb{E}[Y_{ik}] = \frac{\alpha_{ik}}{\alpha_{ik} + \beta_{ik}}. \tag{8}$$

Variance is controlled via $(\alpha_{ik} + \beta_{ik})$.

*Use this when:*

- Yields are bounded between 0 and 1.
- Historical data or expert information allows calibration of mean and variance.

#### 3.2.2 Triangular Distribution (Min, Mode, Max)

Specify:

- $\underline{y}$: minimum feasible yield
- $y_{\text{mode}}$: most likely or typical yield
- $\overline{y}$: maximum feasible yield

*Use this when:*

- Data are sparse.
- Process engineers can provide min/mode/max yield estimates.

#### 3.2.3 Lognormal for Defect Rate, Then Yield = 1 − Defect

Model a **defect fraction** $D_{ik}$ via a lognormal distribution and set:

$$Y_{ik} = 1 - D_{ik}. \tag{9}$$

*Use this when:*

- Defect rates are small but skewed, reflecting multiplicative causes of defects.

**Practical Default Choice:** Use a **Beta distribution** for yield, calibrated from mean yield and coefficient of variation; then discretize to a finite set of scenarios.

### 3.3 Tools and Modeling Approach

#### 3.3.1 Scenario Generation

- For each scenario $\omega$, sample $Y_{ik}(\omega)$ from the chosen distribution (Beta or triangular).
- Compute effective capacities:

$$\text{capacity}_{ik}(\omega) = \text{nominal\_capacity}_{ik} \cdot Y_{ik}(\omega). \tag{10}$$

#### 3.3.2 Stochastic Programming

Capacity constraints become scenario-dependent:

$$\sum_{j} q_{ijk}(\omega) \;\leq\; \text{capacity}_{ik}(\omega) \cdot x_{ik}, \quad \forall\, i, k, \omega, \tag{11}$$

where $x_{ik}$ (site/stage assignment) is first-stage and $q_{ijk}(\omega)$ are second-stage outputs.

#### 3.3.3 Chance-Constrained Optimization

If you prefer probabilistic guarantees rather than scenario-by-scenario feasibility, you can require, for example:

$$\mathbb{P}\left(\text{Total output} \geq D\right) \geq 1 - \alpha. \tag{12}$$

This can be approximated by:

- Enforcing constraints on all but a small fraction of sampled scenarios.
- Using analytic approximations where distributions are simple and independent.

#### 3.3.4 Tools

- **Pyomo** for formulating scenario-based or chance-constrained models.
- **NumPy/SciPy** (or similar) for sampling yields from the specified distributions.

---

## 4 Capacity Disruptions (Downtime and Regulatory Shocks)

### 4.1 Role in the Model

Capacity disruptions capture effects such as:

- Unexpected **shutdowns** (e.g., quality issues or regulatory holds).
- **Partial operation** (e.g., derated capacity during maintenance or staffing constraints).

Let $Z_i(\omega)$ denote the **availability factor** of site $i$ in scenario $\omega$:

- $Z_i(\omega) = 1$: fully available
- $Z_i(\omega) \in (0, 1)$: partial capacity (e.g., 0.5 for half capacity)
- $Z_i(\omega) = 0$: fully down

Effective site capacity becomes:

$$\text{site\_capacity}_i(\omega) = Z_i(\omega) \cdot \text{nominal\_capacity}_i. \tag{13}$$

If you combine this with yield uncertainty:

$$\text{capacity}_{ik}(\omega) = \text{nominal\_capacity}_{ik} \cdot Y_{ik}(\omega) \cdot Z_i(\omega). \tag{14}$$

### 4.2 Recommended Distributions

#### 4.2.1 Bernoulli (On/Off) Site Availability

Model a single-period availability decision:

$$Z_i \sim \text{Bernoulli}(p_i), \tag{15}$$

where $p_i = \mathbb{P}(\text{site } i \text{ is up})$.

- $Z_i = 1$ with probability $p_i$.
- $Z_i = 0$ with probability $1 - p_i$.

*Use this when:*

- You focus on whether a site is fully operational or fully down over the planning horizon.

#### 4.2.2 Discrete Distribution with Multiple Availability Levels

For more nuance, specify discrete levels:

- $Z_i = 1$ (full capacity) with probability $p_{i1}$
- $Z_i = 0.5$ (half capacity) with probability $p_{i2}$
- $Z_i = 0$ (shutdown) with probability $p_{i3}$

*Use this when:*

- Reliability data indicate frequent partial deratings (e.g., capacity reduced for maintenance).
- You want an interpretable model of partial disruptions.

#### 4.2.3 Two-State Markov Chain (Multi-Period Horizon)

For a multi-period planning problem, the capacity state may follow a Markov process:

- States: Up (U), Down (D).
- Transition matrix:

$$P = \begin{pmatrix} p_{UU} & p_{UD} \\ p_{DU} & p_{DD} \end{pmatrix}. \tag{16}$$

*Use this when:*

- Capacity status persistence matters over multiple time periods.
- You require time-correlated disruptions.

**Practical Default Choice (Single-Period or Annual Planning):** Use a **Bernoulli** or a small **discrete distribution over availability factors** $Z_i$ based on historical uptime and maintenance records.

### 4.3 Tools and Modeling Approach

#### 4.3.1 Scenario Generation

- For each scenario $\omega$, sample $Z_i(\omega)$ from Bernoulli or the discrete distribution.
- Combine with yields $Y_{ik}(\omega)$ (if included) to define:

$$\text{capacity}_{ik}(\omega) = \text{nominal\_capacity}_{ik} \cdot Y_{ik}(\omega) \cdot Z_i(\omega). \tag{17}$$

#### 4.3.2 Stochastic Programming

Capacity constraints are adapted as:

$$\sum_{j} q_{ijk}(\omega) \;\leq\; \text{capacity}_{ik}(\omega) \cdot x_{ik}, \quad \forall\, i, k, \omega. \tag{18}$$

Note that $x_{ik}$ remains a first-stage decision, while $Z_i(\omega)$ reflects whether that investment is usable in each scenario.

#### 4.3.3 Robust Optimization

- Define an uncertainty set for availability, e.g.:

$$Z_i \in [\underline{z}_i, 1] \tag{19}$$

- Enforce feasibility against the worst-case $\underline{z}_i$ or against a bounded number of simultaneous site outages.
- Pyomo can handle this with explicit robust counterpart constraints.

---

## 5 Summary of Tools and Distributions

### 5.1 Modeling Tools and Approaches

#### 5.1.1 Core Optimization

- **Pyomo** (MILP model formulation).
- **MILP solvers**: Gurobi, CPLEX, or open-source solvers such as CBC or HiGHS.

#### 5.1.2 Uncertainty Modeling and Solution

- **Scenario-based stochastic programming**:
  - Represent uncertainty via a finite scenario set `SCENARIOS`.
  - Formulate the deterministic equivalent and solve directly.
- **Decomposition methods**:
  - Progressive hedging (e.g., via `mpi-sppy`) for large scenario sets.
- **Robust optimization**:
  - Construct robust counterparts for demand and capacity intervals.
- **Chance-constraints / risk measures**:
  - Approximate probabilistic constraints via sampled scenarios.
  - Incorporate **CVaR** (Conditional Value-at-Risk) of cost into the objective to penalize extreme high-cost outcomes.

#### 5.1.3 Scenario Generation and Validation

- **NumPy/SciPy** (or equivalent) for sampling from Normal, Lognormal, Beta, and discrete distributions.
- **Monte Carlo simulation** to evaluate how well a given design performs under many out-of-sample random draws.

### 5.2 Distribution Choices (Recommended Defaults)

| Uncertainty Type     | Variable                            | Suggested Default Distribution                                  | Notes                                                            |
| -------------------- | ----------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| Demand               | $D$                                 | Truncated Normal or 3–5-point discrete approximation            | Mean = forecast; CV based on historical volatility.              |
| Production yield     | $Y_{ik} \in [0, 1]$                 | Beta (or triangular)                                            | Calibrate from mean yield and variability.                       |
| Capacity disruptions | $Z_i \in \{0, 0.5, 1\}$ or $[0, 1]$ | Bernoulli (on/off) or small discrete set of availability levels | Probabilities from uptime, maintenance, and reliability records. |

*Table 1: Recommended default distributions for each uncertainty type.*

---

## 6 Integration Blueprint (High-Level)

1. **Define a scenario set** $\Omega$ (e.g., 20–100 scenarios for planning-scale models).

2. For each scenario $\omega \in \Omega$, draw:
   - Demand $D(\omega)$ from the chosen demand distribution (truncated Normal, lognormal, or discrete).
   - Yields $Y_{ik}(\omega)$ from the Beta (or triangular) distribution.
   - Availability $Z_i(\omega)$ from Bernoulli or a discrete availability distribution.

3. Compute scenario-dependent capacities:

$$\text{capacity}_{ik}(\omega) = \text{nominal\_capacity}_{ik} \cdot Y_{ik}(\omega) \cdot Z_i(\omega). \tag{20}$$

4. Build a Pyomo model with:
   - A `SCENARIOS` set.
   - Scenario-dependent parameters:
     - `demand[ω]`
     - `capacity[i, k, ω]` (or derived within constraint rules).
   - **First-stage decision variables** (e.g., site opening, stage assignment) without index $\omega$.
   - **Second-stage decision variables** (e.g., production quantities, shipment flows) indexed by $\omega$.

5. Define the objective as minimization of **expected total cost**, optionally augmented with a risk term:

$$\min \; \sum_{\omega \in \Omega} p_\omega\, \text{Cost}(\omega) \;+\; \lambda \cdot \text{RiskMeasure}\bigl(\{\text{Cost}(\omega)\}\bigr), \tag{21}$$

where $\lambda$ is a risk-aversion parameter and the risk measure can be, for example, CVaR.

---

This framework allows systematic incorporation of **demand uncertainty**, **production yield uncertainty**, and **capacity disruptions** into the pharmaceutical site-selection and production model while remaining aligned with a standard scenario-based MILP implementation in Pyomo.
