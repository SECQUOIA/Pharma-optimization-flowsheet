"""Visualization functions for pharmaceutical optimization solutions.

This module provides functions to visualize optimization solutions
as network diagrams showing the manufacturing process flow.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyomo.environ as pyo


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

def visualize_scenarios(optimizer, scenarios_to_show=None, figsize=(20, 5)):
    """
    Draw the selected vendor path for each scenario side-by-side,
    showing per-step flow, capacity utilisation, and unmet demand.
    """
    selections  = optimizer.get_selected_options()
    steps       = optimizer.ordered_steps
    local_scens = optimizer._ef.local_scenarios

    if scenarios_to_show is None:
        scenarios_to_show = list(local_scens.keys())

    n = len(scenarios_to_show)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    # ── layout constants ─────────────────────────────────────────────
    NODE_W, NODE_H = 2.2, 1.6
    X_START  = 0.0
    X_STEP   = 4.0
    Y_NODE   = 2.5
    Y_BAR    = 0.4
    BAR_H    = 0.35
    BAR_W    = NODE_W * 1.1

    C_BOT   = '#d9534f'
    C_STEP  = '#4a90d9'
    C_START = '#555555'

    def box(ax, cx, cy, w, h, color, lines, fs=8):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.08", lw=1.2,
            edgecolor='white', facecolor=color, alpha=0.93, zorder=3))
        ax.text(cx, cy, "\n".join(lines),
                ha='center', va='center', fontsize=fs,
                color='white', fontweight='bold', zorder=4, linespacing=1.45)

    def arrow(ax, x1, x2, y):
        ax.annotate("",
            xy=(x2 - NODE_W/2 - 0.05, y),
            xytext=(x1 + NODE_W/2 + 0.05, y),
            arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.8,
                            mutation_scale=16), zorder=2)

    for ax, sname in zip(axes, scenarios_to_show):
        sm         = local_scens[sname]
        demand     = pyo.value(sm.demand)
        unmet      = pyo.value(sm.unmet)
        throughput = demand - unmet
        flows      = {s: pyo.value(sm.flow[s]) for s in steps}
        caps       = {s: optimizer.prod_dict[(s, selections[s])] for s in steps}
        bottleneck = min(steps, key=lambda s: caps[s])

        xs    = {s: X_START + (i + 1) * X_STEP for i, s in enumerate(steps)}
        x_max = X_START + (len(steps) + 0.5) * X_STEP

        # Start node
        box(ax, X_START, Y_NODE, NODE_W, NODE_H, C_START, ["Start"], fs=9)

        # Step nodes + arrows
        prev_x = X_START
        for s in steps:
            cx, v   = xs[s], selections[s]
            fl, cap = flows[s], caps[s]
            util    = 100 * fl / cap if cap > 0 else 0
            is_bot  = (s == bottleneck)
            color   = C_BOT if is_bot else C_STEP
            h_extra = 0.28 if is_bot else 0
            lines   = [f"Step {s} · {v}",
                       f"flow  {fl:,.0f}",
                       f"cap   {cap:,.0f}",
                       f"util  {util:.0f}%"]
            if is_bot:
                lines.append("⚡ bottleneck")
            box(ax, cx, Y_NODE, NODE_W, NODE_H + h_extra, color, lines, fs=7.5)
            arrow(ax, prev_x, cx, Y_NODE)
            prev_x = cx

        # Demand bar under last step node
        last_x = xs[steps[-1]]
        x0     = last_x - BAR_W / 2
        met_w  = BAR_W * (throughput / demand) if demand > 0 else 0
        unm_w  = BAR_W * (unmet / demand)      if demand > 0 else 0

        ax.add_patch(mpatches.Rectangle(
            (x0, Y_BAR - BAR_H/2), met_w, BAR_H,
            facecolor=C_STEP, alpha=0.85, zorder=3))
        ax.add_patch(mpatches.Rectangle(
            (x0 + met_w, Y_BAR - BAR_H/2), unm_w, BAR_H,
            facecolor=C_BOT, alpha=0.85, zorder=3))
        ax.add_patch(mpatches.Rectangle(
            (x0, Y_BAR - BAR_H/2), BAR_W, BAR_H,
            lw=0.8, edgecolor='#888', facecolor='none', zorder=4))

        ax.plot([last_x, last_x],
                [Y_NODE - NODE_H/2 - 0.05, Y_BAR + BAR_H/2 + 0.05],
                '--', color='#bbb', lw=1, zorder=1)
        ax.text(last_x, Y_BAR - BAR_H/2 - 0.12,
                f"demand {demand:,.0f}   met {throughput:,.0f}   unmet {unmet:,.0f}",
                ha='center', va='top', fontsize=7.5, color='#333')

        # Title
        prob = optimizer.scenarios[sname][1]
        sc   = optimizer.shortage_penalty * unmet
        ax.set_title(
            f"Scenario: {sname.upper()}   (p = {prob:.0%})\n"
            f"shortage cost = ${sc:,.0f}",
            fontsize=11, fontweight='bold', pad=8)

        ax.set_xlim(-NODE_W * 0.8, x_max)
        ax.set_ylim(Y_BAR - 0.85, Y_NODE + NODE_H * 0.85)
        ax.axis('off')

    # Legend
    fig.legend(handles=[
        mpatches.Patch(color=C_STEP,  label='Selected vendor'),
        mpatches.Patch(color=C_BOT,   label='Bottleneck step / unmet demand'),
        mpatches.Patch(color=C_START, label='Start'),
    ], loc='lower center', ncol=3, fontsize=9,
       framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Per-Scenario Throughput Along Selected Vendor Path",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
