"""Pharmaceutical Manufacturing Optimization Package.

This package provides tools for optimizing pharmaceutical manufacturing networks,
including vendor selection and transportation routing.

Classes:
    Generator: Generates random test instances
    EnhancedProductionOptimizer: Deterministic optimization model
    DemandScenarioGenerator: Generates demand scenarios for stochastic optimization
    StochasticProductionOptimizer: Two-stage stochastic optimization model (mpi-sppy)

Functions:
    pharma_scenario_creator: mpi-sppy scenario creator for direct use with EF/PH
    visualize_solution_from_optimizer: Visualize optimization solutions
"""

from .instance_generator import Generator
from .deterministic_optimization import EnhancedProductionOptimizer
from .stochastic import DemandScenarioGenerator, StochasticProductionOptimizer, pharma_scenario_creator
from .visualization import visualize_solution_from_optimizer, visualize_scenarios

__all__ = [
    'Generator',
    'EnhancedProductionOptimizer',
    'DemandScenarioGenerator',
    'StochasticProductionOptimizer',
    'pharma_scenario_creator',
    'visualize_solution_from_optimizer',
    'visualize_scenarios'
]
