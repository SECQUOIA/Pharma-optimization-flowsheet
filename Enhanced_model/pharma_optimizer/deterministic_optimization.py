"""Deterministic production optimizer for pharmaceutical manufacturing.

This module provides the EnhancedProductionOptimizer class for solving deterministic
vendor selection and transportation routing problems in pharmaceutical manufacturing.
"""

import pyomo.environ as pyo
import pandas as pd


class EnhancedProductionOptimizer:
    """
    Enhanced production optimizer that incorporates both production costs and transport costs
    between vendors across sequential steps.

    Production CSV must have columns: [step, Vendor, Production, Cost]
    Transport CSV must have columns: [SourceStep, DestinationStep, SourceVendor, DestinationVendor, TransportationCost]
    """

    def __init__(self, prod_cost_csv: str, transport_cost_csv: str, demand: float):
        """
        Initialize by loading data and constructing the enhanced Pyomo model.

        :param prod_cost_csv: Path to CSV with columns [step, Vendor, Production, Cost]
        :param transport_cost_csv: Path to CSV with columns [SourceStep, DestinationStep, SourceVendor, DestinationVendor, TransportationCost]
        :param demand: Total demand that must be satisfied
        """
        # Load production data
        self.prod_df = pd.read_csv(prod_cost_csv)
        self.transport_df = pd.read_csv(transport_cost_csv)
        self.demand = demand

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

        # Get all transport routes (step1, vendor1, step2, vendor2)
        self.transport_routes = list(self.transport_dict.keys())

        # Get ordered steps
        self.ordered_steps = sorted(self.vendors.keys())

        # Build model
        self._build_model()

    def _build_model(self):
        """
        Build the deterministic Pyomo optimization model.
        """
        m = pyo.ConcreteModel()

        # Sets
        m.steps = pyo.Set(initialize=self.vendors.keys())
        m.step_option = pyo.Set(dimen=2, initialize=self.pairs)
        m.transport_routes = pyo.Set(dimen=4, initialize=self.transport_routes)

        # Parameters
        m.production = pyo.Param(
            m.step_option, initialize=self.prod_dict,
            within=pyo.NonNegativeReals
        )
        m.prod_cost = pyo.Param(
            m.step_option, initialize=self.cost_dict,
            within=pyo.NonNegativeReals
        )
        m.transport_cost = pyo.Param(
            m.transport_routes, initialize=self.transport_dict,
            within=pyo.NonNegativeReals
        )
        m.demand = pyo.Param(initialize=self.demand,
                             within=pyo.NonNegativeReals)

        # Decision variables
        m.x = pyo.Var(m.step_option, domain=pyo.Binary)  # vendor selection
        m.y = pyo.Var(m.transport_routes, domain=pyo.Binary)  # transport route selection

        # Constraints

        # One vendor per step
        def one_per_step(mdl, step):
            return sum(mdl.x[step, v] for v in self.vendors[step]) == 1
        m.one_per_step = pyo.Constraint(m.steps, rule=one_per_step)

        # Production level constraint
        def prod_level(mdl):
            return sum(
                mdl.production[s, o] * mdl.x[s, o]
                for s, o in mdl.step_option
            ) >= mdl.demand
        m.prod_level = pyo.Constraint(rule=prod_level)

        # Only one transport route per consecutive step pair
        for i in range(len(self.ordered_steps) - 1):
            step1, step2 = self.ordered_steps[i], self.ordered_steps[i + 1]

            def transport_selection_rule(mdl):
                valid_routes = [
                    (s1, v1, s2, v2) for (s1, v1, s2, v2) in self.transport_routes
                    if s1 == step1 and s2 == step2
                ]
                if valid_routes:
                    return sum(mdl.y[route] for route in valid_routes) == 1
                else:
                    return pyo.Constraint.Skip

            setattr(m, f'transport_selection_{step1}_{step2}',
                    pyo.Constraint(rule=transport_selection_rule))

        # Link transport routes to vendor selection
        for step1, vendor1, step2, vendor2 in self.transport_routes:
            def link_transport_rule(mdl):
                return mdl.y[step1, vendor1, step2, vendor2] <= mdl.x[step1, vendor1]

            def link_transport_rule2(mdl):
                return mdl.y[step1, vendor1, step2, vendor2] <= mdl.x[step2, vendor2]

            setattr(m, f'link_transport_{step1}_{vendor1}_{step2}_{vendor2}_1',
                    pyo.Constraint(rule=link_transport_rule))
            setattr(m, f'link_transport_{step1}_{vendor1}_{step2}_{vendor2}_2',
                    pyo.Constraint(rule=link_transport_rule2))

        # Objective: minimize production costs + transport costs
        m.obj = pyo.Objective(
            expr=sum(m.prod_cost[s, o] * m.x[s, o] for s, o in m.step_option) +
                 sum(m.transport_cost[route] * m.y[route] for route in m.transport_routes),
            sense=pyo.minimize
        )

        self.model = m

    def solve(self, solver_name: str = 'glpk', tee: bool = False):
        """Solve the optimization model.

        Args:
            solver_name: Name of the solver to use (default: 'glpk')
            tee: Whether to display solver output (default: False)

        Returns:
            Solver result object
        """
        opt = pyo.SolverFactory(solver_name)
        result = opt.solve(self.model, tee=tee)
        return result

    def get_selected_options(self):
        """
        Retrieve the chosen vendor for each step.

        :return: dict mapping step -> selected vendor
        """
        selections = {}
        for (s, v) in self.model.step_option:
            if pyo.value(self.model.x[s, v]) > 0.5:
                selections[s] = v
        return selections

    def get_selected_transport_routes(self):
        """
        Retrieve the selected transport routes.

        :return: list of selected transport routes
        """
        selected_routes = []
        for route in self.model.transport_routes:
            if pyo.value(self.model.y[route]) > 0.5:
                selected_routes.append(route)
        return selected_routes

    def get_total_production_cost(self):
        """
        Compute the total production cost of the solution.

        :return: float
        """
        return sum(
            pyo.value(self.model.prod_cost[s, o] * self.model.x[s, o])
            for s, o in self.model.step_option
        )

    def get_total_transport_cost(self):
        """
        Compute the total transport cost of the solution.

        :return: float
        """
        return sum(
            pyo.value(self.model.transport_cost[route] * self.model.y[route])
            for route in self.model.transport_routes
        )

    def get_total_cost(self):
        """
        Compute the total cost (production + transport) of the solution.

        :return: float
        """
        return pyo.value(self.model.obj)

    def summary(self):
        """
        Print a detailed summary of results including transport routes.
        """
        selections = self.get_selected_options()
        transport_routes = self.get_selected_transport_routes()

        print("=== OPTIMIZATION RESULTS ===")
        print("\nSelected vendors by step:")
        for step in sorted(selections.keys()):
            vendor = selections[step]
            prod_cost = self.cost_dict[(step, vendor)]
            production = self.prod_dict[(step, vendor)]
            print(f"  Step {step}: {vendor} (Cost: ${prod_cost}, Production: {production})")

        print("\nSelected transport routes:")
        for step1, vendor1, step2, vendor2 in sorted(transport_routes):
            trans_cost = self.transport_dict[(step1, vendor1, step2, vendor2)]
            print(f"  Step {step1}({vendor1}) -> Step {step2}({vendor2}): ${trans_cost}")

        print(f"\n=== COST BREAKDOWN ===")
        print(f"Total production cost: ${self.get_total_production_cost():.2f}")
        print(f"Total transport cost: ${self.get_total_transport_cost():.2f}")
        print(f"TOTAL COST: ${self.get_total_cost():.2f}")

        # Show the complete path
        print(f"\n=== OPTIMAL PATH ===")
        path_str = " -> ".join([f"Step {s}({selections[s]})" for s in sorted(selections.keys())])
        print(f"Path: {path_str}")
