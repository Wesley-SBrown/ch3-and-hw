import cvxpy as cp

import cvxpy as cp
import numpy as np

# Parameters
plants = ['Seattle', 'SanDiego']
customers = ['NewYork', 'Chicago', 'Topeka']

supply = {'Seattle': 350, 'SanDiego': 600}
demand = {'NewYork': 325, 'Chicago': 300, 'Topeka': 275}

distances = {
    ('Seattle', 'NewYork'): 2500,
    ('Seattle', 'Chicago'): 1700,
    ('Seattle', 'Topeka'): 1800,
    ('SanDiego', 'NewYork'): 2500,
    ('SanDiego', 'Chicago'): 1800,
    ('SanDiego', 'Topeka'): 1400
}

cost_per_mile = 90  # cost per case per thousand miles

# Create the cost matrix
cost_matrix = np.zeros((len(plants), len(customers)))

for i, plant in enumerate(plants):
    for j, customer in enumerate(customers):
        cost_matrix[i, j] = distances[(plant, customer)] * cost_per_mile

# Decision variable: x[i,j] -> number of cases shipped from plant i to customer j
x = cp.Variable((len(plants), len(customers)), nonneg=True)

# Objective: Minimize the total shipping cost
total_cost = cp.sum(cp.multiply(cost_matrix, x))

# Constraints
constraints = []

# Supply constraints (sum of shipments from each plant should not exceed its supply)
for i, plant in enumerate(plants):
    constraints.append(cp.sum(x[i, :]) <= supply[plant])

# Demand constraints (sum of shipments to each customer should meet its demand)
for j, customer in enumerate(customers):
    constraints.append(cp.sum(x[:, j]) == demand[customer])

# Define the problem
problem = cp.Problem(cp.Minimize(total_cost), constraints)

# Solve the problem
problem.solve()

# Display results
print("Optimal Shipping Plan:")
for i, plant in enumerate(plants):
    for j, customer in enumerate(customers):
        print(f"Ship {x[i, j].value:.2f} cases from {plant} to {customer}")

print(f"\nTotal Shipping Cost: ${problem.value:.2f}")
