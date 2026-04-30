# TODO

## 1. Implement cost based on production volume

Currently the overall cost depends only on which vendors are chosen (binary $x$), not on how much is actually produced. Update the objective so production cost scales with produced quantity.

![Yirang's Slack comment on cost](images/yirang_cost_comment.png)

> **Yirang Park — Monday, April 20th, 10:47 AM:**
> Hey Sai, there's a comment that I left a few days ago on that as well. Right now the overall cost only depends on what vendors are chosen, rather than how much is actually produced because if you look at the first term in the objective function the decision variables x are binary. So the cost is incurred only when the x is chosen and the corresponding coefficient cost for that choice regardless of how much we produce. Which is why when we don't have "unmet demands" we have the same cost as you have written in one of the markdowns.

## 2. Perform a scaling study

- Vary solvers (e.g., Gurobi, CPLEX, CBC, HiGHS).
- Vary overall network size (sites, stages, scenarios).
- Record solve time, optimality gap, and memory usage.

## 3. Implement a few more demand distributions

Extend `Enhanced_model/pharma_optimizer/stochastic.py` to support additional demand distributions beyond the current set (e.g., lognormal, truncated Normal variants, discrete low/base/high scenarios — see `GPT_stochastic report.md` §2.2).
