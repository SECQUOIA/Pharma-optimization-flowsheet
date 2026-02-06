"""Pharmaceutical Manufacturing Optimization Package.

This package provides tools for optimizing pharmaceutical manufacturing networks,
including vendor selection and transportation routing.

Classes:
    Generator: Generates random test instances
    EnhancedProductionOptimizer: Deterministic optimization model
    DemandScenarioGenerator: Generates demand scenarios for stochastic optimization
    StochasticProductionOptimizer: Two-stage stochastic optimization model

Functions:
    visualize_solution_from_optimizer: Visualize optimization solutions
"""

from .generator import Generator
from .deterministic import EnhancedProductionOptimizer
from .stochastic import DemandScenarioGenerator, StochasticProductionOptimizer
from .visualization import visualize_solution_from_optimizer

__all__ = [
    'Generator',
    'EnhancedProductionOptimizer',
    'DemandScenarioGenerator',
    'StochasticProductionOptimizer',
    'visualize_solution_from_optimizer',
]
