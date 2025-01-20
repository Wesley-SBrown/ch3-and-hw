import cvxpy as cp
import numpy as np

# Define sets: designers and projects
designers = ['A', 'B', 'C']
projects = [1, 2, 3, 4]

# Capability scores (s_ij) as a dictionary
s = {
    ('A', 1): 90, ('A', 2): 80, ('A', 3): 10, ('A', 4): 50,
    ('B', 1): 60, ('B', 2): 70, ('B', 3): 50, ('B', 4): 65,
    ('C', 1): 70, ('C', 2): 40, ('C', 3): 72, ('C', 4): 85
}

# Hours required for each project (h_j)
required_hours = {1: 70, 2: 50, 3: 85, 4: 35}

# Map designers and projects to indices for easier matrix manipulation
designer_index = {'A': 0, 'B': 1, 'C': 2}
project_index = {1: 0, 2: 1, 3: 2, 4: 3}

# Create a matrix for capability scores
s_matrix = np.zeros((3, 4))
for (designer, project), score in s.items():
    i = designer_index[designer]
    j = project_index[project]
    s_matrix[i, j] = score

# Decision variable: hours each designer works on each project
x = cp.Variable((3, 4), nonneg=True)  # 3 designers, 4 projects

# Objective function: Maximize the total capability score
objective = cp.Maximize(
    cp.sum(cp.multiply(s_matrix, x))  # Element-wise multiplication of s_matrix and x
)

# Constraints:
constraints = []

# Each designer works 80 hours total
for i in range(3):
    constraints.append(cp.sum(x[i, :]) == 80)

# Each project requires specific total hours
for j in range(4):
    constraints.append(cp.sum(x[:, j]) == required_hours[projects[j]])

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Output results
print("Optimal work assignment:")
for i, designer in enumerate(designers):
    for j, project in enumerate(projects):
        print(f"Designer {designer} works {x[i, j].value:.2f} hours on Project {project}")

print(f"\nTotal Capability Score: {problem.value:.2f}")
