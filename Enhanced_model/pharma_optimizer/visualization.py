"""Visualization functions for pharmaceutical optimization solutions.

This module provides functions to visualize optimization solutions
as network diagrams showing the manufacturing process flow.
"""

import networkx as nx
import matplotlib.pyplot as plt


def visualize_solution_from_optimizer(optimizer, title="Optimization Solution Path"):
    """Create a visual representation of the pharmaceutical optimization solution.

    Generates a directed graph showing the manufacturing process flow,
    highlighting the selected vendors/options at each step.

    Args:
        optimizer: The optimizer object with solved model (EnhancedProductionOptimizer
                   or StochasticProductionOptimizer)
        title (str): Title for the visualization plot

    Returns:
        tuple: Contains (G, pos, chosen) where:
            - G (nx.DiGraph): NetworkX directed graph object
            - pos (dict): Node positions for visualization
            - chosen (list): List of (step, option) tuples representing the solution path
    """

    # Get solution data from optimizer
    vendors = optimizer.vendors
    selections = optimizer.get_selected_options()

    # Sort manufacturing steps in ascending order
    steps = sorted(vendors.keys())

    # Create directed graph for the manufacturing process
    G = nx.DiGraph()
    START = 'Start'
    G.add_node(START)

    # Add nodes for each vendor option at each manufacturing step
    for step in steps:
        for opt in vendors[step]:
            G.add_node((step, opt))

    # Extract the selected solution path
    chosen = [(step, vendor) for step, vendor in sorted(selections.items())]

    # Build the actual solution path edges
    real_edges = []

    if chosen:
        real_edges.append((START, chosen[0]))

    for a, b in zip(chosen, chosen[1:]):
        real_edges.append((a, b))

    # Calculate node positions using hierarchical layout
    # Assign each node to a layer based on its manufacturing step
    pos = {}

    # Position START node
    pos[START] = (0, 0)

    # Position nodes in vertical columns by manufacturing step
    horizontal_spacing = 3.0
    vertical_spacing = 2.0

    for step_idx, step in enumerate(steps):
        x = (step_idx + 1) * horizontal_spacing  # Horizontal position (column)
        num_options = len(vendors[step])

        # Center the options vertically
        for option_idx, option in enumerate(vendors[step]):
            y = (option_idx - (num_options - 1) / 2) * vertical_spacing
            pos[(step, option)] = (x, y)

    # Initialize matplotlib figure
    plt.figure(figsize=(12, 8))

    # Draw all possible edges (faint background connections)
    all_edges = []
    all_edges.extend([(START, (steps[0], opt)) for opt in vendors[steps[0]]])
    for i in range(len(steps) - 1):
        current_step = steps[i]
        next_step = steps[i + 1]
        for opt1 in vendors[current_step]:
            for opt2 in vendors[next_step]:
                all_edges.append(((current_step, opt1), (next_step, opt2)))

    nx.draw_networkx_edges(
        G, pos,
        edgelist=all_edges,
        edge_color='lightgray',
        width=0.5,
        alpha=0.3,
        arrows=False
    )

    # Draw the selected solution path (highlighted)
    if real_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=real_edges,
            arrowstyle='-|>',
            arrowsize=16,
            edge_color='red',
            width=3,
            alpha=0.8
        )

    # Color palette for different manufacturing steps
    colors = ['skyblue', 'lightgreen', 'orange', 'violet', 'gold', 'pink', 'lightcoral']

    # Draw nodes for each manufacturing step
    for idx, step in enumerate(steps):
        step_nodes = [(step, o) for o in vendors[step]]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=step_nodes,
            node_color=colors[idx % len(colors)],
            node_shape='s',
            node_size=1200,
            alpha=0.8
        )

    # Draw the start node
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[START],
        node_color='darkgreen',
        node_shape='s',
        node_size=1500,
        alpha=0.9
    )

    # Create node labels
    labels = {START: 'Start'}
    labels.update({(step, o): f"Step {step}\n{o}" for step in steps for o in vendors[step]})

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels,
                           font_color='white',
                           font_weight='bold',
                           font_size=9)

    # Configure plot appearance
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    return G, pos, chosen
