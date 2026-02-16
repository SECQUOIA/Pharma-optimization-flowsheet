"""Stochastic optimization for pharmaceutical manufacturing with demand uncertainty.

This module provides classes for:
- DemandScenarioGenerator: Generate demand scenarios for uncertainty modeling
- StochasticProductionOptimizer: Two-stage stochastic programming model
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional

import pyomo.environ as pyo
import pandas as pd


class DemandScenarioGenerator:
    """Generates demand scenarios for stochastic optimization.

    Supports multiple distribution types for modeling demand uncertainty:
    - Discrete Low/Base/High scenarios
    - Truncated Normal distribution
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


class StochasticProductionOptimizer:
    """
    Static-dynamic two-stage stochastic production optimizer with demand uncertainty
    and multi-echelon flow constraints.

    Models a serial pharmaceutical supply chain where material flows sequentially
    through manufacturing steps (e.g., API synthesis -> formulation -> fill/finish).
    The bottleneck step governs throughput, not the sum of capacities.

    First Stage (static, scenario-independent):
        - Vendor selection at each step: x[s,v] in {0,1}
        - Transport route selection: y[route] in {0,1}

    Second Stage (dynamic, scenario-dependent):
        - Material flow at each echelon: flow[s, omega] >= 0
        - Unmet demand (shortage): unmet[omega] >= 0

    Multi-echelon constraints:
        - Per-step capacity: flow[s,w] <= capacity of selected vendor at step s
        - Flow conservation: flow[s,w] <= flow[s-1,w] (serial chain)
        - Demand satisfaction at final echelon only

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

        # Build stochastic model
        self._build_model()

    def _build_model(self):
        """Build the two-stage stochastic Pyomo model."""
        m = pyo.ConcreteModel()

        # ============ SETS ============
        m.steps = pyo.Set(initialize=self.vendors.keys())
        m.step_option = pyo.Set(dimen=2, initialize=self.pairs)
        m.transport_routes = pyo.Set(dimen=4, initialize=self.transport_routes)
        m.scenarios = pyo.Set(initialize=self.scenarios.keys())

        # ============ PARAMETERS ============
        m.production = pyo.Param(
            m.step_option,
            initialize=self.prod_dict,
            within=pyo.NonNegativeReals
        )
        m.prod_cost = pyo.Param(
            m.step_option,
            initialize=self.cost_dict,
            within=pyo.NonNegativeReals
        )
        m.transport_cost = pyo.Param(
            m.transport_routes,
            initialize=self.transport_dict,
            within=pyo.NonNegativeReals
        )

        # Scenario-dependent parameters
        m.demand = pyo.Param(
            m.scenarios,
            initialize={s: self.scenarios[s][0] for s in self.scenarios},
            within=pyo.NonNegativeReals
        )
        m.probability = pyo.Param(
            m.scenarios,
            initialize={s: self.scenarios[s][1] for s in self.scenarios},
            within=pyo.NonNegativeReals
        )
        m.shortage_penalty = pyo.Param(
            initialize=self.shortage_penalty,
            within=pyo.NonNegativeReals
        )

        # ============ FIRST-STAGE VARIABLES (scenario-independent) ============
        m.x = pyo.Var(m.step_option, domain=pyo.Binary)  # Vendor selection
        m.y = pyo.Var(m.transport_routes, domain=pyo.Binary)  # Transport route selection

        # ============ SECOND-STAGE VARIABLES (scenario-dependent) ============
        m.flow = pyo.Var(m.steps, m.scenarios, domain=pyo.NonNegativeReals)  # Material flow at each echelon
        m.unmet = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)  # Unmet demand

        # ============ FIRST-STAGE CONSTRAINTS ============

        # One vendor per step
        def one_per_step(mdl, step):
            return sum(mdl.x[step, v] for v in self.vendors[step]) == 1
        m.one_per_step = pyo.Constraint(m.steps, rule=one_per_step)

        # One transport route per consecutive step pair
        for i in range(len(self.ordered_steps) - 1):
            step1, step2 = self.ordered_steps[i], self.ordered_steps[i + 1]

            def transport_selection_rule(mdl, s1=step1, s2=step2):
                valid_routes = [
                    (r1, v1, r2, v2) for (r1, v1, r2, v2) in self.transport_routes
                    if r1 == s1 and r2 == s2
                ]
                if valid_routes:
                    return sum(mdl.y[route] for route in valid_routes) == 1
                return pyo.Constraint.Skip

            setattr(m, f'transport_selection_{step1}_{step2}',
                   pyo.Constraint(rule=transport_selection_rule))

        # Link transport routes to vendor selection
        def link_transport_source(mdl, s1, v1, s2, v2):
            return mdl.y[s1, v1, s2, v2] <= mdl.x[s1, v1]
        m.link_transport_source = pyo.Constraint(m.transport_routes, rule=link_transport_source)

        def link_transport_dest(mdl, s1, v1, s2, v2):
            return mdl.y[s1, v1, s2, v2] <= mdl.x[s2, v2]
        m.link_transport_dest = pyo.Constraint(m.transport_routes, rule=link_transport_dest)

        # ============ SECOND-STAGE CONSTRAINTS (multi-echelon flow) ============

        first_step = self.ordered_steps[0]
        last_step = self.ordered_steps[-1]

        # Per-step capacity: flow cannot exceed selected vendor's capacity
        def step_capacity(mdl, step, omega):
            selected_capacity = sum(
                mdl.production[step, v] * mdl.x[step, v]
                for v in self.vendors[step]
            )
            return mdl.flow[step, omega] <= selected_capacity
        m.step_capacity = pyo.Constraint(m.steps, m.scenarios, rule=step_capacity)

        # Flow conservation: flow at step s cannot exceed flow from previous step
        def flow_conservation(mdl, step, omega):
            if step == first_step:
                return pyo.Constraint.Skip
            prev_step = self.ordered_steps[self.ordered_steps.index(step) - 1]
            return mdl.flow[step, omega] <= mdl.flow[prev_step, omega]
        m.flow_conservation = pyo.Constraint(m.steps, m.scenarios, rule=flow_conservation)

        # Demand satisfaction at final echelon only
        def demand_satisfaction(mdl, omega):
            return mdl.flow[last_step, omega] + mdl.unmet[omega] >= mdl.demand[omega]
        m.demand_satisfaction = pyo.Constraint(m.scenarios, rule=demand_satisfaction)

        # ============ OBJECTIVE ============
        # Expected cost = production cost + transport cost + E[shortage penalty]
        def expected_cost(mdl):
            # First-stage costs (deterministic, paid regardless of scenario)
            prod_cost_expr = sum(mdl.prod_cost[s, v] * mdl.x[s, v] for s, v in mdl.step_option)
            transport_cost_expr = sum(mdl.transport_cost[r] * mdl.y[r] for r in mdl.transport_routes)

            # Second-stage expected shortage penalty
            shortage_cost_expr = sum(
                mdl.probability[omega] * mdl.shortage_penalty * mdl.unmet[omega]
                for omega in mdl.scenarios
            )

            return prod_cost_expr + transport_cost_expr + shortage_cost_expr

        m.obj = pyo.Objective(rule=expected_cost, sense=pyo.minimize)

        self.model = m

    def solve(self, solver_name: str = 'glpk', tee: bool = False):
        """Solve the stochastic optimization model."""
        opt = pyo.SolverFactory(solver_name)
        result = opt.solve(self.model, tee=tee)
        return result

    def get_selected_options(self) -> Dict[int, str]:
        """Get selected vendor for each step."""
        selections = {}
        for (s, v) in self.model.step_option:
            if pyo.value(self.model.x[s, v]) > 0.5:
                selections[s] = v
        return selections

    def get_selected_transport_routes(self) -> List[Tuple]:
        """Get selected transport routes."""
        selected = []
        for route in self.model.transport_routes:
            if pyo.value(self.model.y[route]) > 0.5:
                selected.append(route)
        return selected

    def get_throughput(self) -> Dict[str, Dict[int, float]]:
        """Get per-step flow values for each scenario.

        Returns:
            Dict mapping scenario name to {step: flow_value}
        """
        throughput = {}
        for omega in self.model.scenarios:
            throughput[omega] = {
                s: pyo.value(self.model.flow[s, omega])
                for s in self.ordered_steps
            }
        return throughput

    def get_scenario_results(self) -> pd.DataFrame:
        """Get detailed results for each scenario including multi-echelon throughput."""
        results = []
        last_step = self.ordered_steps[-1]

        # Per-step capacities from selected vendors (same across scenarios)
        step_capacities = {}
        for step in self.ordered_steps:
            step_capacities[step] = sum(
                pyo.value(self.model.production[step, v] * self.model.x[step, v])
                for v in self.vendors[step]
            )
        bottleneck_step = min(step_capacities, key=step_capacities.get)
        bottleneck_capacity = step_capacities[bottleneck_step]

        for omega in self.model.scenarios:
            demand = pyo.value(self.model.demand[omega])
            prob = pyo.value(self.model.probability[omega])

            # Actual throughput from flow variable at final echelon
            throughput = pyo.value(self.model.flow[last_step, omega])
            unmet = pyo.value(self.model.unmet[omega])

            # Cost breakdown
            prod_cost = sum(
                pyo.value(self.model.prod_cost[s, v] * self.model.x[s, v])
                for s, v in self.model.step_option
            )
            transport_cost = sum(
                pyo.value(self.model.transport_cost[r] * self.model.y[r])
                for r in self.model.transport_routes
            )
            shortage_cost = self.shortage_penalty * unmet
            total_cost = prod_cost + transport_cost + shortage_cost

            results.append({
                'scenario': omega,
                'demand': demand,
                'probability': prob,
                'bottleneck_capacity': bottleneck_capacity,
                'throughput': throughput,
                'unmet_demand': unmet,
                'prod_cost': prod_cost,
                'transport_cost': transport_cost,
                'shortage_cost': shortage_cost,
                'total_cost': total_cost
            })

        return pd.DataFrame(results)

    def get_expected_cost(self) -> float:
        """Get expected total cost across all scenarios."""
        return pyo.value(self.model.obj)

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
        print("(Multi-Echelon with Demand Uncertainty)")
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
