"""Stochastic optimization for pharmaceutical manufacturing with demand uncertainty.

This module provides:
- DemandScenarioGenerator: Generate demand scenarios for uncertainty modeling
- pharma_scenario_creator: mpi-sppy-compatible scenario creator function
- StochasticProductionOptimizer: Two-stage stochastic programming model (via mpi-sppy)

mpi-sppy is used for:
- Extensive Form (EF) assembly and solving
- Progressive Hedging (PH) decomposition for large-scale problems
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional

import pyomo.environ as pyo
import pandas as pd

try:
    import mpisppy.utils.sputils as sputils
    from mpisppy.opt.ef import ExtensiveForm
    from mpisppy.opt.ph import PH
    _MPISPPY_AVAILABLE = True
except ImportError:
    _MPISPPY_AVAILABLE = False


class DemandScenarioGenerator:
    """Generates demand scenarios for stochastic optimization.

    Supports multiple distribution types for modeling demand uncertainty:
    - Discrete Low/Base/High scenarios
    - Truncated Normal distribution
    - Non-Stationary (time-varying mean with trend and seasonality)
    - Lumpy / Intermittent (sporadic large orders, many zero-demand periods)
    - Slow-Moving (low-volume Poisson demand)
    - Binomial (fixed population, each with probability of needing product)
    - Custom scenario sets

    Based on Section 2 of the uncertainty report:
    - Mean (mu) = forecast demand
    - sigma = cv * mu, where cv is coefficient of variation (e.g., 0.2 for 20%)
    - D = max(X, d_min) where d_min is minimum demand (e.g., 5% of mu)
    """

    def __init__(self, mean_demand: float, cv: float = 0.2, seed: Optional[int] = None):
        """Initialize the scenario generator.

        Args:
            mean_demand: Expected/forecast demand (mu)
            cv: Coefficient of variation (sigma/mu), default 0.2 (20% variability)
            seed: Random seed for reproducibility
        """
        self.mean_demand = mean_demand
        self.cv = cv
        self.std_demand = cv * mean_demand
        self.min_demand = max(0.05 * mean_demand, 1)  # Minimum 5% of mean or 1

        if seed is not None:
            np.random.seed(seed)

    def generate_discrete_scenarios(self,
                                    levels: List[float] = [0.8, 1.0, 1.3],
                                    probabilities: List[float] = [0.2, 0.5, 0.3]
                                   ) -> Dict[str, Tuple[float, float]]:
        """Generate discrete Low/Base/High demand scenarios.

        As per report Section 2.2.3:
        - Scenario 1 (low): D = 0.8*mu
        - Scenario 2 (base): D = mu
        - Scenario 3 (high): D = 1.3*mu

        Args:
            levels: Multipliers for mean demand [low, base, high]
            probabilities: Probability of each scenario (must sum to 1)

        Returns:
            Dictionary mapping scenario name to (demand, probability)
        """
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1")

        scenario_names = ['low', 'base', 'high'] if len(levels) == 3 else [f's{i+1}' for i in range(len(levels))]

        scenarios = {}
        for name, level, prob in zip(scenario_names, levels, probabilities):
            scenarios[name] = (level * self.mean_demand, prob)

        return scenarios

    def generate_normal_scenarios(self, num_scenarios: int = 5) -> Dict[str, Tuple[float, float]]:
        """Generate scenarios by sampling from truncated normal distribution.

        As per report Section 2.2.1:
        - X ~ N(mu, sigma^2) with D = max(X, d_min)

        Uses equiprobable discretization of the truncated normal distribution.

        Args:
            num_scenarios: Number of scenarios to generate (3-5 recommended)

        Returns:
            Dictionary mapping scenario name to (demand, probability)
        """
        # Use truncated normal: truncate at min_demand on left, no upper bound
        a = (self.min_demand - self.mean_demand) / self.std_demand  # Lower bound in standard units
        b = np.inf  # No upper bound

        # Generate equiprobable quantiles
        prob = 1.0 / num_scenarios
        quantiles = [(i + 0.5) / num_scenarios for i in range(num_scenarios)]

        # Sample at quantile midpoints from truncated normal
        scenarios = {}
        for i, q in enumerate(quantiles):
            demand = stats.truncnorm.ppf(q, a, b, loc=self.mean_demand, scale=self.std_demand)
            scenarios[f'scenario_{i+1}'] = (demand, prob)

        return scenarios

    def generate_nonstationary_scenarios(self,
                                        time_point: float,
                                        trend_rate: float = 0.05,
                                        amplitude: float = 0.0,
                                        period: float = 12.0,
                                        num_scenarios: int = 5
                                       ) -> Dict[str, Tuple[float, float]]:
        """Generate scenarios from a non-stationary demand distribution.

        Models demand where the mean shifts over time due to trend and/or
        seasonality. At a given time point t, demand is drawn from a
        truncated normal with time-varying mean:

            mu(t) = mean_demand * (1 + trend_rate * t)
                    + amplitude * sin(2 * pi * t / period)

        Args:
            time_point: The time index at which to generate scenarios
            trend_rate: Linear growth rate per time unit (e.g., 0.05 = 5% growth)
            amplitude: Amplitude of seasonal component (in demand units)
            period: Period of seasonal cycle (e.g., 12 for monthly data with yearly cycle)
            num_scenarios: Number of equiprobable scenarios to generate

        Returns:
            Dictionary mapping scenario name to (demand, probability)
        """
        # Time-varying mean
        mu_t = (self.mean_demand * (1 + trend_rate * time_point)
                + amplitude * np.sin(2 * np.pi * time_point / period))
        mu_t = max(mu_t, self.min_demand)
        sigma_t = self.cv * mu_t

        # Truncated normal at this time point
        lower = max(self.min_demand, 0)
        a = (lower - mu_t) / sigma_t if sigma_t > 0 else 0
        b = np.inf

        prob = 1.0 / num_scenarios
        quantiles = [(i + 0.5) / num_scenarios for i in range(num_scenarios)]

        scenarios = {}
        for i, q in enumerate(quantiles):
            demand = stats.truncnorm.ppf(q, a, b, loc=mu_t, scale=sigma_t)
            scenarios[f'ns_t{time_point}_{i+1}'] = (max(demand, lower), prob)

        return scenarios

    def generate_lumpy_scenarios(self,
                                 occurrence_prob: float = 0.3,
                                 size_mean: Optional[float] = None,
                                 size_cv: float = 0.5,
                                 num_scenarios: int = 5
                                ) -> Dict[str, Tuple[float, float]]:
        """Generate scenarios from a lumpy/intermittent demand distribution.

        Models demand that occurs sporadically with large quantities.
        Characterized by high inter-demand intervals and high variance
        when demand occurs (ADI > 1.32, CV^2 > 0.49).

        Uses a compound distribution:
            D = B * X
        where B ~ Bernoulli(occurrence_prob) and X ~ LogNormal(mu_ln, sigma_ln)

        One scenario always represents zero demand (no occurrence).
        The remaining scenarios discretize the non-zero demand distribution.

        Args:
            occurrence_prob: Probability that any demand occurs in a period (0, 1)
            size_mean: Mean demand size when it occurs (defaults to mean_demand)
            size_cv: CV of demand size when it occurs (default 0.5)
            num_scenarios: Total number of scenarios (>= 2, one is always zero-demand)

        Returns:
            Dictionary mapping scenario name to (demand, probability)
        """
        if num_scenarios < 2:
            raise ValueError("num_scenarios must be >= 2 for lumpy demand")
        if not 0 < occurrence_prob < 1:
            raise ValueError("occurrence_prob must be in (0, 1)")

        if size_mean is None:
            size_mean = self.mean_demand

        # LogNormal parameters from mean and cv of the size distribution
        sigma_ln = np.sqrt(np.log(1 + size_cv**2))
        mu_ln = np.log(size_mean) - 0.5 * sigma_ln**2

        scenarios = {}

        # Zero-demand scenario
        scenarios['no_demand'] = (0.0, 1.0 - occurrence_prob)

        # Non-zero scenarios: equiprobable discretization of LogNormal
        n_nonzero = num_scenarios - 1
        prob_each = occurrence_prob / n_nonzero
        quantiles = [(i + 0.5) / n_nonzero for i in range(n_nonzero)]

        for i, q in enumerate(quantiles):
            demand = stats.lognorm.ppf(q, s=sigma_ln, scale=np.exp(mu_ln))
            scenarios[f'lumpy_{i+1}'] = (demand, prob_each)

        return scenarios

    def generate_slow_moving_scenarios(self,
                                        mean_rate: Optional[float] = None,
                                        num_scenarios: int = 5
                                       ) -> Dict[str, Tuple[float, float]]:
        """Generate scenarios from a slow-moving (Poisson) demand distribution.

        Models low-volume demand with many zero periods and small order sizes.
        Characterized by high inter-demand intervals but low variance when
        demand occurs (ADI > 1.32, CV^2 <= 0.49).

        D ~ Poisson(lambda) where lambda = mean_rate.

        Selects the most probable demand values from the Poisson PMF and
        groups remaining tail probability into boundary scenarios.

        Args:
            mean_rate: Poisson rate parameter lambda (defaults to mean_demand)
            num_scenarios: Number of scenarios to generate

        Returns:
            Dictionary mapping scenario name to (demand, probability)
        """
        if mean_rate is None:
            mean_rate = self.mean_demand

        # Compute PMF for values covering 99.9% of probability mass
        max_k = int(stats.poisson.ppf(0.999, mean_rate)) + 1
        k_values = np.arange(0, max_k + 1)
        pmf_values = stats.poisson.pmf(k_values, mean_rate)

        if num_scenarios >= len(k_values):
            # Fewer unique values than scenarios — use exact PMF
            scenarios = {}
            for k, p in zip(k_values, pmf_values):
                if p > 1e-10:
                    scenarios[f'poisson_{int(k)}'] = (float(k), float(p))
            # Normalize for any tail truncation
            total_p = sum(p for _, p in scenarios.values())
            scenarios = {name: (d, p / total_p) for name, (d, p) in scenarios.items()}
            return scenarios

        # Group into num_scenarios equiprobable bins via quantile boundaries
        prob = 1.0 / num_scenarios
        scenarios = {}
        for i in range(num_scenarios):
            q_lo = i / num_scenarios
            q_hi = (i + 1) / num_scenarios
            q_mid = (q_lo + q_hi) / 2

            demand = stats.poisson.ppf(q_mid, mean_rate)
            scenarios[f'slow_{i+1}'] = (float(demand), prob)

        return scenarios

    def generate_binomial_scenarios(self,
                                     num_trials: int,
                                     prob_success: float,
                                     num_scenarios: int = 5
                                    ) -> Dict[str, Tuple[float, float]]:
        """Generate scenarios from a binomial demand distribution.

        Models demand as the number of successes from a fixed population,
        e.g., a patient population of size n where each patient independently
        needs the drug with probability p.

        D ~ Binomial(n, p)  with mean = n*p, variance = n*p*(1-p).

        Args:
            num_trials: Population size n (number of independent trials)
            prob_success: Probability each trial generates demand (0, 1)
            num_scenarios: Number of scenarios to generate

        Returns:
            Dictionary mapping scenario name to (demand, probability)
        """
        if not 0 < prob_success < 1:
            raise ValueError("prob_success must be in (0, 1)")

        # Compute PMF for the full support [0, n]
        k_values = np.arange(0, num_trials + 1)
        pmf_values = stats.binom.pmf(k_values, num_trials, prob_success)

        # Filter to values with non-negligible probability
        mask = pmf_values > 1e-10
        k_values = k_values[mask]
        pmf_values = pmf_values[mask]

        if num_scenarios >= len(k_values):
            # Fewer unique values than scenarios — use exact PMF
            scenarios = {}
            total_p = pmf_values.sum()
            for k, p in zip(k_values, pmf_values):
                scenarios[f'binom_{int(k)}'] = (float(k), float(p / total_p))
            return scenarios

        # Group into num_scenarios equiprobable bins via quantile discretization
        prob = 1.0 / num_scenarios
        scenarios = {}
        for i in range(num_scenarios):
            q_mid = (i + 0.5) / num_scenarios
            demand = stats.binom.ppf(q_mid, num_trials, prob_success)
            scenarios[f'binom_{i+1}'] = (float(demand), prob)

        return scenarios

    def generate_custom_scenarios(self,
                                  demands: List[float],
                                  probabilities: List[float]
                                 ) -> Dict[str, Tuple[float, float]]:
        """Generate scenarios from custom demand values and probabilities.

        Args:
            demands: List of demand values
            probabilities: List of probabilities (must sum to 1)

        Returns:
            Dictionary mapping scenario name to (demand, probability)
        """
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1")
        if len(demands) != len(probabilities):
            raise ValueError("demands and probabilities must have same length")

        scenarios = {}
        for i, (d, p) in enumerate(zip(demands, probabilities)):
            scenarios[f'scenario_{i+1}'] = (d, p)

        return scenarios

    def summary(self, scenarios: Dict[str, Tuple[float, float]]):
        """Print summary statistics for generated scenarios.

        Args:
            scenarios: Dictionary of scenarios from generate_* methods
        """
        demands = [s[0] for s in scenarios.values()]
        probs = [s[1] for s in scenarios.values()]

        expected = sum(d * p for d, p in zip(demands, probs))
        variance = sum(p * (d - expected)**2 for d, p in zip(demands, probs))
        std = np.sqrt(variance)

        print("=== DEMAND SCENARIOS ===")
        print(f"{'Scenario':<15} {'Demand':>12} {'Probability':>12}")
        print("-" * 40)
        for name, (demand, prob) in scenarios.items():
            print(f"{name:<15} {demand:>12.2f} {prob:>12.2%}")
        print("-" * 40)
        print(f"Expected demand: {expected:.2f}")
        print(f"Std deviation:   {std:.2f}")
        print(f"CV:              {std/expected:.2%}")


def pharma_scenario_creator(
    scenario_name: str,
    scenarios: Dict[str, Tuple[float, float]],
    vendors: Dict,
    pairs: List[Tuple],
    prod_dict: Dict,
    cost_dict: Dict,
    transport_dict: Dict,
    transport_routes: List[Tuple],
    ordered_steps: List,
    shortage_penalty: float,
) -> pyo.ConcreteModel:
    """mpi-sppy scenario creator for the pharmaceutical stochastic program.

    Builds a Pyomo ConcreteModel for a single demand scenario. First-stage
    variables (vendor and route selection) are marked as non-anticipativity
    variables so mpi-sppy can enforce consistency across scenarios in the
    Extensive Form or Progressive Hedging decomposition.

    Args:
        scenario_name: Key into `scenarios` dict identifying this scenario.
        scenarios: Full dict mapping scenario_name -> (demand, probability).
        vendors: Dict mapping step -> list of vendor names.
        pairs: List of (step, vendor) tuples.
        prod_dict: Dict mapping (step, vendor) -> production capacity.
        cost_dict: Dict mapping (step, vendor) -> production cost.
        transport_dict: Dict mapping route tuple -> transport cost.
        transport_routes: List of (src_step, src_vendor, dst_step, dst_vendor) tuples.
        ordered_steps: Sorted list of manufacturing steps.
        shortage_penalty: Cost per unit of unmet demand.

    Returns:
        Pyomo ConcreteModel annotated with mpi-sppy non-anticipativity metadata.
    """
    if not _MPISPPY_AVAILABLE:
        raise ImportError(
            "mpi-sppy is required. Install with: pip install mpi-sppy"
        )

    demand, prob = scenarios[scenario_name]
    first_step = ordered_steps[0]
    last_step = ordered_steps[-1]

    m = pyo.ConcreteModel()

    # ============ SETS ============
    m.steps = pyo.Set(initialize=vendors.keys())
    m.step_option = pyo.Set(dimen=2, initialize=pairs)
    m.transport_routes = pyo.Set(dimen=4, initialize=transport_routes)

    # ============ PARAMETERS ============
    m.production = pyo.Param(
        m.step_option,
        initialize=prod_dict,
        within=pyo.NonNegativeReals
    )
    m.prod_cost = pyo.Param(
        m.step_option,
        initialize=cost_dict,
        within=pyo.NonNegativeReals
    )
    m.transport_cost = pyo.Param(
        m.transport_routes,
        initialize=transport_dict,
        within=pyo.NonNegativeReals
    )
    m.demand = pyo.Param(initialize=demand, within=pyo.NonNegativeReals)
    m.shortage_penalty = pyo.Param(initialize=shortage_penalty, within=pyo.NonNegativeReals)

    # ============ FIRST-STAGE VARIABLES (non-anticipativity) ============
    m.x = pyo.Var(m.step_option, domain=pyo.Binary)       # Vendor selection
    m.y = pyo.Var(m.transport_routes, domain=pyo.Binary)  # Transport route selection

    # ============ SECOND-STAGE VARIABLES (scenario-specific) ============
    m.flow = pyo.Var(m.steps, domain=pyo.NonNegativeReals)  # Material flow at each echelon
    m.unmet = pyo.Var(domain=pyo.NonNegativeReals)           # Unmet demand (recourse)

    # ============ FIRST-STAGE CONSTRAINTS ============

    # One vendor per step
    def one_per_step(mdl, step):
        return sum(mdl.x[step, v] for v in vendors[step]) == 1
    m.one_per_step = pyo.Constraint(m.steps, rule=one_per_step)

    # One transport route per consecutive step pair
    for i in range(len(ordered_steps) - 1):
        step1, step2 = ordered_steps[i], ordered_steps[i + 1]
        valid_routes = [
            (r1, v1, r2, v2) for (r1, v1, r2, v2) in transport_routes
            if r1 == step1 and r2 == step2
        ]
        if valid_routes:
            setattr(m, f'transport_selection_{step1}_{step2}',
                    pyo.Constraint(expr=sum(m.y[r] for r in valid_routes) == 1))

    # Link transport routes to vendor selection
    def link_transport_source(mdl, s1, v1, s2, v2):
        return mdl.y[s1, v1, s2, v2] <= mdl.x[s1, v1]
    m.link_transport_source = pyo.Constraint(m.transport_routes, rule=link_transport_source)

    def link_transport_dest(mdl, s1, v1, s2, v2):
        return mdl.y[s1, v1, s2, v2] <= mdl.x[s2, v2]
    m.link_transport_dest = pyo.Constraint(m.transport_routes, rule=link_transport_dest)

    # ============ SECOND-STAGE CONSTRAINTS (multi-echelon flow) ============

    # Per-step capacity: flow cannot exceed selected vendor's capacity
    def step_capacity(mdl, step):
        selected_capacity = sum(
            mdl.production[step, v] * mdl.x[step, v]
            for v in vendors[step]
        )
        return mdl.flow[step] <= selected_capacity
    m.step_capacity = pyo.Constraint(m.steps, rule=step_capacity)

    # Flow conservation: flow at step s cannot exceed flow from previous step
    def flow_conservation(mdl, step):
        if step == first_step:
            return pyo.Constraint.Skip
        prev_step = ordered_steps[ordered_steps.index(step) - 1]
        return mdl.flow[step] <= mdl.flow[prev_step]
    m.flow_conservation = pyo.Constraint(m.steps, rule=flow_conservation)

    # Demand satisfaction at final echelon only
    def demand_satisfaction(mdl):
        return mdl.flow[last_step] + mdl.unmet >= mdl.demand
    m.demand_satisfaction = pyo.Constraint(rule=demand_satisfaction)

    # ============ COST EXPRESSIONS ============
    # First-stage cost: paid regardless of scenario (vendor + transport selection)
    m.FirstStageCost = pyo.Expression(
        expr=sum(m.prod_cost[s, v] * m.x[s, v] for s, v in m.step_option)
             + sum(m.transport_cost[r] * m.y[r] for r in m.transport_routes)
    )

    # Second-stage cost: scenario-specific shortage recourse
    m.SecondStageCost = pyo.Expression(expr=m.shortage_penalty * m.unmet)

    # ============ OBJECTIVE (per-scenario total cost, NOT probability-weighted) ============
    # mpi-sppy weights by _mpisppy_probability when assembling the EF objective:
    #   EF obj = sum_omega [ prob_omega * (FirstStageCost + SecondStageCost_omega) ]
    #          = FirstStageCost + E[SecondStageCost]   (since sum prob_omega = 1)
    m.obj = pyo.Objective(
        expr=m.FirstStageCost + m.SecondStageCost,
        sense=pyo.minimize
    )

    # ============ mpi-sppy NON-ANTICIPATIVITY ANNOTATION ============
    # Set scenario probability before attach_root_node so it is not overridden.
    m._mpisppy_probability = prob

    # attach_root_node registers:
    #   - the first-stage cost expression (used by PH proximal term)
    #   - the list of first-stage (non-anticipative) variables
    # varlist takes Pyomo Var objects; build_vardatalist expands indexed vars.
    sputils.attach_root_node(m, m.FirstStageCost, [m.x, m.y])

    return m


class StochasticProductionOptimizer:
    """
    Static-dynamic two-stage stochastic production optimizer with demand uncertainty
    and multi-echelon flow constraints. Uses mpi-sppy for scenario management and solving.

    Models a serial pharmaceutical supply chain where material flows sequentially
    through manufacturing steps (e.g., API synthesis -> formulation -> fill/finish).
    The bottleneck step governs throughput, not the sum of capacities.

    First Stage (static, scenario-independent):
        - Vendor selection at each step: x[s,v] in {0,1}
        - Transport route selection: y[route] in {0,1}

    Second Stage (dynamic, scenario-dependent):
        - Material flow at each echelon: flow[s] >= 0  (per scenario)
        - Unmet demand (shortage): unmet >= 0           (per scenario)

    Solved via mpi-sppy:
        - Extensive Form (EF): assembles and solves the full deterministic
          equivalent in a single pass. Exact, scales to ~hundreds of scenarios.
        - Progressive Hedging (PH): Lagrangian decomposition by scenario.
          Heuristic for MIPs, exact for LPs. Scales to thousands of scenarios.

    Objective: Minimize expected total cost including shortage penalty
        min sum(prod_cost * x) + sum(transport_cost * y)
            + E[penalty * unmet[omega]]
    """

    def __init__(self,
                 prod_cost_csv: str,
                 transport_cost_csv: str,
                 scenarios: Dict[str, Tuple[float, float]],
                 shortage_penalty: float = 1000.0):
        """
        Initialize the stochastic optimizer.

        Args:
            prod_cost_csv: Path to production/manufacturing CSV
            transport_cost_csv: Path to transportation cost CSV
            scenarios: Dict mapping scenario_name -> (demand, probability)
            shortage_penalty: Penalty cost per unit of unmet demand
        """
        if not _MPISPPY_AVAILABLE:
            raise ImportError(
                "mpi-sppy is required. Install with: pip install mpi-sppy"
            )

        # Load production data
        self.prod_df = pd.read_csv(prod_cost_csv)
        self.transport_df = pd.read_csv(transport_cost_csv)
        self.scenarios = scenarios
        self.shortage_penalty = shortage_penalty

        # Process production data
        self.vendors = (
            self.prod_df
            .groupby("step")["Vendor"]
            .apply(list)
            .to_dict()
        )

        # (step, vendor) pairs
        self.pairs = list(self.prod_df[["step", "Vendor"]]
                          .itertuples(index=False, name=None))

        # Production lookup dicts
        self.prod_dict = {
            (r.step, r.Vendor): r.Production
            for r in self.prod_df.itertuples()
        }
        self.cost_dict = {
            (r.step, r.Vendor): r.Cost
            for r in self.prod_df.itertuples()
        }

        # Process transport data
        self.transport_dict = {
            (r.SourceStep, r.SourceVendor, r.DestinationStep, r.DestinationVendor): r.TransportationCost
            for r in self.transport_df.itertuples()
        }

        self.transport_routes = list(self.transport_dict.keys())
        self.ordered_steps = sorted(self.vendors.keys())

        # Populated after solve()
        self._ef: Optional[ExtensiveForm] = None
        self._ph: Optional[PH] = None

    def _creator_kwargs(self) -> dict:
        """Build the kwargs dict for pharma_scenario_creator."""
        return {
            'scenarios': self.scenarios,
            'vendors': self.vendors,
            'pairs': self.pairs,
            'prod_dict': self.prod_dict,
            'cost_dict': self.cost_dict,
            'transport_dict': self.transport_dict,
            'transport_routes': self.transport_routes,
            'ordered_steps': self.ordered_steps,
            'shortage_penalty': self.shortage_penalty,
        }

    def solve(
        self,
        solver_name: str = 'glpk',
        tee: bool = False,
        method: str = 'ef',
        ph_options: Optional[dict] = None,
    ):
        """Solve the two-stage stochastic program via mpi-sppy.

        Args:
            solver_name: MIP solver name recognised by Pyomo (e.g. 'glpk', 'cplex', 'gurobi').
            tee: Stream solver output to stdout.
            method: 'ef' for Extensive Form (exact, default) or 'ph' for Progressive Hedging.
            ph_options: Override options for PH. Relevant keys:
                - 'PHIterLimit' (default 50)
                - 'defaultPHrho' (default 1.0)
                - 'defaultPHp'   (default 2)

        Returns:
            For 'ef': the Pyomo solver results object from ExtensiveForm.solve_extensive_form().
            For 'ph': None (convergence info available via self._ph).
        """
        scenario_names = list(self.scenarios.keys())
        kwargs = self._creator_kwargs()

        if method == 'ph':
            default_ph_opts = {
                'solvername': solver_name,
                'PHIterLimit': 50,
                'defaultPHrho': 1.0,
                'defaultPHp': 2,
                'convthresh': 1e-4,
                'verbose': False,
                'display_progress': False,
                'display_convergence_detail': False,
                'linearize_proximal_terms': False,
            }
            if ph_options:
                default_ph_opts.update(ph_options)

            self._ph = PH(
                default_ph_opts,
                scenario_names,
                pharma_scenario_creator,
                scenario_creator_kwargs=kwargs,
            )
            self._ph.ph_main()
            self._ef = None
            return None

        # Default: Extensive Form
        # options dict must contain 'solver' key for ExtensiveForm
        self._ef = ExtensiveForm(
            {'solver': solver_name},
            scenario_names,
            pharma_scenario_creator,
            scenario_creator_kwargs=kwargs,
        )
        results = self._ef.solve_extensive_form(tee=tee)
        self._ph = None
        return results

    def _active_scenarios(self) -> Dict[str, pyo.ConcreteModel]:
        """Return the dict of solved scenario submodels."""
        if self._ef is not None:
            return self._ef.local_scenarios
        if self._ph is not None:
            return self._ph.local_scenarios
        raise RuntimeError("Model has not been solved yet. Call solve() first.")

    def _first_stage_model(self) -> pyo.ConcreteModel:
        """Return any one scenario model to read first-stage decisions from."""
        return next(iter(self._active_scenarios().values()))

    def get_selected_options(self) -> Dict[int, str]:
        """Get selected vendor for each step."""
        m = self._first_stage_model()
        selections = {}
        for (s, v) in m.step_option:
            if pyo.value(m.x[s, v]) > 0.5:
                selections[s] = v
        return selections

    def get_selected_transport_routes(self) -> List[Tuple]:
        """Get selected transport routes."""
        m = self._first_stage_model()
        return [
            route for route in m.transport_routes
            if pyo.value(m.y[route]) > 0.5
        ]

    def get_throughput(self) -> Dict[str, Dict[int, float]]:
        """Get per-step flow values for each scenario.

        Returns:
            Dict mapping scenario name to {step: flow_value}
        """
        return {
            sname: {s: pyo.value(smodel.flow[s]) for s in self.ordered_steps}
            for sname, smodel in self._active_scenarios().items()
        }

    def get_scenario_results(self) -> pd.DataFrame:
        """Get detailed results for each scenario including multi-echelon throughput."""
        last_step = self.ordered_steps[-1]
        m_fs = self._first_stage_model()

        # Per-step capacities from selected vendors (same across scenarios)
        step_capacities = {
            step: sum(
                pyo.value(m_fs.production[step, v] * m_fs.x[step, v])
                for v in self.vendors[step]
            )
            for step in self.ordered_steps
        }
        bottleneck_step = min(step_capacities, key=step_capacities.get)
        bottleneck_capacity = step_capacities[bottleneck_step]

        prod_cost = sum(
            pyo.value(m_fs.prod_cost[s, v] * m_fs.x[s, v])
            for s, v in m_fs.step_option
        )
        transport_cost = sum(
            pyo.value(m_fs.transport_cost[r] * m_fs.y[r])
            for r in m_fs.transport_routes
        )

        results = []
        for sname, smodel in self._active_scenarios().items():
            demand = pyo.value(smodel.demand)
            prob = self.scenarios[sname][1]
            throughput = pyo.value(smodel.flow[last_step])
            unmet = pyo.value(smodel.unmet)
            shortage_cost = self.shortage_penalty * unmet
            total_cost = prod_cost + transport_cost + shortage_cost

            results.append({
                'scenario': sname,
                'demand': demand,
                'probability': prob,
                'bottleneck_capacity': bottleneck_capacity,
                'throughput': throughput,
                'unmet_demand': unmet,
                'prod_cost': prod_cost,
                'transport_cost': transport_cost,
                'shortage_cost': shortage_cost,
                'total_cost': total_cost,
            })

        return pd.DataFrame(results)

    def get_expected_cost(self) -> float:
        """Get expected total cost across all scenarios."""
        if self._ef is not None:
            return self._ef.get_objective_value()
        # For PH, compute from scenario results
        df = self.get_scenario_results()
        return (df['total_cost'] * df['probability']).sum()

    def get_cost_statistics(self) -> Dict[str, float]:
        """Calculate cost statistics across scenarios."""
        scenario_df = self.get_scenario_results()

        expected = (scenario_df['total_cost'] * scenario_df['probability']).sum()
        variance = (scenario_df['probability'] * (scenario_df['total_cost'] - expected)**2).sum()
        std_dev = np.sqrt(variance)

        return {
            'expected_cost': expected,
            'std_dev': std_dev,
            'min_cost': scenario_df['total_cost'].min(),
            'max_cost': scenario_df['total_cost'].max()
        }

    def summary(self):
        """Print comprehensive summary of stochastic solution."""
        selections = self.get_selected_options()
        transport_routes = self.get_selected_transport_routes()
        scenario_df = self.get_scenario_results()
        cost_stats = self.get_cost_statistics()

        print("=" * 60)
        print("STATIC-DYNAMIC STOCHASTIC OPTIMIZATION RESULTS")
        print("(Multi-Echelon with Demand Uncertainty — via mpi-sppy)")
        print("=" * 60)

        print("\n--- FIRST-STAGE DECISIONS (Static) ---")
        print("\nSelected vendors by step:")
        step_capacities = {}
        for step in sorted(selections.keys()):
            vendor = selections[step]
            prod_cost = self.cost_dict[(step, vendor)]
            production = self.prod_dict[(step, vendor)]
            step_capacities[step] = production
            print(f"  Step {step}: {vendor} (Cost: ${prod_cost:,}, Capacity: {production:,})")

        print("\nSelected transport routes:")
        for s1, v1, s2, v2 in sorted(transport_routes):
            trans_cost = self.transport_dict[(s1, v1, s2, v2)]
            print(f"  Step {s1}({v1}) -> Step {s2}({v2}): ${trans_cost}")

        path_str = " -> ".join([f"Step {s}({selections[s]})" for s in sorted(selections.keys())])
        print(f"\nOptimal Path: {path_str}")

        # Multi-echelon throughput analysis
        bottleneck_step = min(step_capacities, key=step_capacities.get)
        bottleneck_cap = step_capacities[bottleneck_step]
        print("\n--- MULTI-ECHELON THROUGHPUT ---")
        print("Per-step capacity of selected vendors:")
        for step in sorted(step_capacities.keys()):
            marker = " <-- BOTTLENECK" if step == bottleneck_step else ""
            print(f"  Step {step}: {step_capacities[step]:,}{marker}")
        print(f"\nEffective throughput (bottleneck): {bottleneck_cap:,}")

        print("\n--- SECOND-STAGE RESULTS (Dynamic per Scenario) ---")
        print(scenario_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

        print("\n--- COST STATISTICS ---")
        print(f"Expected Cost:     ${cost_stats['expected_cost']:,.2f}")
        print(f"Std Deviation:     ${cost_stats['std_dev']:,.2f}")
        print(f"Minimum Cost:      ${cost_stats['min_cost']:,.2f}")
        print(f"Maximum Cost:      ${cost_stats['max_cost']:,.2f}")
        print("=" * 60)
