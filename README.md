# Pharmaceutical Manufacturing Outsourcing Optimization
## Background
This project seeks to address a common pharmaceutical industry challenge of scaling a drug to production after FDA approval. 
There is urgency in the process of scaling to production after filing, because the patent for the drug is only held by the company for 10 years 
and vertically integrating is no longer the most economically viable option. Third party vendors must be employed to sequentially 
produce the active pharmaceutical ingredient (API) over many steps. Selection of these vendors at each step can be 
a strenuous and inefficient process to work out as dozens of steps is common to build an API. Computational optimization is the perfect tool to rapidly determine the third party vendor sequence for API production that allows for the pharmaceutical company to maximize profit.

# Problem Description

![image-4.png](attachment:image-4.png)

Consider the above flowsheet, in which raw materials are input to a sequence of steps, each step representing a unit operation type such as a reaction or separation. The final product of the overall process is an Active Pharmaceutical Ingredient (API). For a given step, there are various types of operations such as batch, continuous, or hybrid modes that can complete the unit operation, which may also vary with the vendors. The economic implications of selecting among these operation modes and vendor options can be systematically evaluated using optimization.

![image-3.png](attachment:image-3.png)

A pharmaceutical company that has just patented an API and wishes to produce it via this synthesis pathway often outsources the steps to third party vendors to retain a competitive advantage. Third party vendors are individually capable of some steps via some synthesis pathways as displayed below. 

![image-5.png](attachment:image-5.png)

The challenge is determining the best vendors to assign to each step. Computational optimization provides a modern tool to efficiently determine the best vendor for each step by minimizing the cost necessary to produce the products from the reactants. For this problem considering all possibilities in a superstructure results in the following flowsheet.

![image-2.png](attachment:image-2.png)

## Solution
### Solver
In this Jupyter notebook, we utilize the Gurobi solver to identify the optimal site for each step of the process. The class reads in the CSVs. The first contains the production information for vendors in 4 columns: the step number, the vendor name, the production amount, and the production cost. The second CSV contains the transport costs between the production sites for each step of the process. This CSV had 5 columns: the starting step, the ending step, the starting vendor, the ending vendor, and how much it cost to go between them. The optimizer class that was built ingests these CSVs and inputs them into the solver. 

### Visualization
The second part of this notebook takes the output of the Gurobi solver and uses it to construct a graph showing the optimal solution. It then will highlight the optimal path for between steps of the process. 
