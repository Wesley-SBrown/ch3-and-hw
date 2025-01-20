import cvxpy as cp

x = cp.Variable(4,nonneg=True)

total_investment = 1_000_000

# Parameters
expected_returns = [13,7,12,14]
worst_return = [0.06, 0.07, 0.1, 0.09] 
durations = [3,4,7.5,9]
diversity_limit = 0.402025

objective = cp.Maximize(expected_returns[0] * x[0] + expected_returns[1] * x[1] + expected_returns[2] * x[2] + expected_returns[3] * x[3])

# Constraints 
constraints = [
    cp.sum(x) <= total_investment,
    (worst_return[0]*x[0]+worst_return[1]*x[1]+worst_return[2]*x[2]+worst_return[3]*x[3]) >= 0.08 * (cp.sum(x)),
    (durations[0]*x[0]+durations[1]*x[1]+durations[2]*x[2]+durations[3]*x[3]) <= 6 * (cp.sum(x)),
    x <= 0.402025 * cp.sum(x)

]

problem = cp.Problem(objective, constraints)

problem.solve()

print("Optimal investments in each bond:")
for i in range(4):
    print(f"Bond {i+1}: ${x[i].value:,.2f}")

print(f"Optimal expected return: ${objective.value:,.2f}")

