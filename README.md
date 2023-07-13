# Chapter 2
### Macro of TEIAS

## Example 1

---

Example Consider the supply and demand functions for three goods given by

```math
\begin{align*}
q_1^s &= -10 + p_1 & q_1^d &= 20 - p_1 - p_3 \\
q_2^s &= 2p_2 & q_2^d &= 40 - 2p_2 - p_3 \\
q_3^s &= -5 + p_3 & q_3^d &= 25 - p_1 - p_2 - p_3
\end{align*}
```

As one can see, the supply of the three goods only depends on their own price, while the demand side shows strong price interdependencies. In order to solve for the equilibrium prices of the system, we set supply equal to demand \(q_i^s=q_i^d\) in each market, which after rearranging yields the linear equation system

```math
\begin{aligned}
2 p_1+p_3 & =30 \\
4 p_2+p_3 & =40 \\
p_1+p_2+2 p_3 & =30 .
\end{aligned}
```

---
To solve this system of linear equations in Julia, you can use the `\` operator, which is used for solving linear systems of equations. First, you need to represent your system of equations as a matrix `A` and a vector `b`, and then you can solve for the vector `x` by evaluating `A\b`.

Here is the corresponding Julia code:

```julia
using LinearAlgebra

# Define the coefficients of the linear equations
A = [2 0 1; 
     0 4 1; 
     1 1 2]

# Define the constants on the right side of the equations
b = [30, 40, 30]

# Solve for x
x = A \ b

println("The solution is ", x)
```

Just replace the `A` and `b` arrays with your coefficients and constants, and then run the program. It will print out the solutions `p1`, `p2`, and `p3` for the system of equations.

Ensure that you have the LinearAlgebra package installed and imported in Julia. If you don't have it, you can install it via `using Pkg; Pkg.add("LinearAlgebra")`.

Please note that the `\` operator assumes that the system of equations is well-determined (i.e., it has a unique solution). If the system of equations is not well-determined (for instance, if it has no solutions or an infinite number of solutions), the `\` operator may not give correct results.

## Example 2

---

Example Suppose the demand function in a goods market is given by

$$
q^d(p)=0.5 p^{-0.2}+0.5 p^{-0.5}
$$

where the first term denotes domestic demand and the second term export demand. Supply should be inelastic and given by $q^s=2$ units. At what price $p^*$ does the market clear? Setting supply equal to demand leads to the equation

$$
0.5 p^{-0.2}+0.5 p^{-0.5}=2
$$

which can be reformulated as

$$
f(p)=0.5 p^{-0.2}+0.5 p^{-0.5}-2=0 .
$$

Equation (2.2) exactly has the form $f(p)=0$ described above. The market clearing price consequently is the solution to a nonlinear equation. Note that it is not possible to derive an analytical solution to this problem. Hence, we need a numerical method to solve for $p^*$.

---

We can use the `Roots.jl` package in Julia which includes a function for the bisection method (`fzero`). 

```julia
using Roots

# Define the function for which we want to find the root
f(p) = 0.5 * p^(-0.2) + 0.5 * p^(-0.5) - 2

# Initial interval [a,b]
a = 0.1
b = 10.0

# Find the root using the bisection method
p_star = fzero(f, (a, b))

println("The market clearing price is ", p_star)
```

In this code, `fzero` is the function that implements the bisection method (as well as other root-finding algorithms, depending on the inputs), and `(a, b)` is the initial interval in which to search for a root. 

Ensure that you have the Roots package installed and imported in Julia. If you don't have it, you can install it via `using Pkg; Pkg.add("Roots")`. 

The initial interval from 0.1 to 10.0 is arbitrary, you may need to adjust these values based on your specific problem and domain knowledge.

Just like other numerical methods, the bisection method may not always find a root, especially if the initial interval does not contain any roots or if the function does not satisfy the method's assumptions (in this case, that the function changes sign over the interval).

## Example 3

---

Example Two firms compete in a simple Cournot duopoly with the inverse demand and the cost functions

$$
P(q)=q^{-1 / \eta} \quad C_k\left(q_k\right)=\frac{c_k}{2} q_k^2 \quad \text { for firm } k=1,2 \text { with } \quad q=q_1+q_2 .
$$

Given the profit functions of the two firms

$$
\Pi_k\left(q_1, q_2\right)=P\left(q_1+q_2\right) q_k-C_k\left(q_k\right)
$$

each firm $k$ takes the other firm's output as given and chooses its own output level in order to solve

$$
\frac{\partial \Pi_k}{\partial q_k}=f(q)=\left(q_1+q_2\right)^{-1 / \eta}-\frac{1}{\eta}\left(q_1+q_2\right)^{-1 / \eta-1} q_k-c_k q_k=0 \text { with } k=1,2 .
$$

---

This problem involves solving a system of two nonlinear equations. In Julia, we can use the `nlsolve` function from the `NLsolve.jl` package to solve this system. 

Let's suppose that `η = 1.6`, `c1 = 0.6` and `c2 = 0.8`:

```julia
using NLsolve

η = 1.6
c1 = 0.6
c2 = 0.8

function cournot!(F, q)
    F[1] = (q[1] + q[2])^(-1/η) - (1/η) * (q[1] + q[2])^(-1/η - 1) * q[1] - c1 * q[1]
    F[2] = (q[1] + q[2])^(-1/η) - (1/η) * (q[1] + q[2])^(-1/η - 1) * q[2] - c2 * q[2]
end

initial_guess = [0.5, 0.5]

result = nlsolve(cournot!, initial_guess)

println("The equilibrium quantities are ", result.zero)
```

This code defines the function `cournot!` which calculates the values of the two first-order conditions, given the quantities `q[1]` and `q[2]`. Then, the `nlsolve` function is called to find the quantities that set these two conditions to zero. The result is stored in the `result` object, and the equilibrium quantities can be extracted using `result.zero`. 

Please remember that `nlsolve` uses the Newton's method to find the roots, and that the quality of the solution depends on the initial guess. In this case, the initial guess is `[0.5, 0.5]`, but you may need to adjust it based on your specific problem and domain knowledge.

## Example 4

---

Example A household can consume two goods $x_1$ and $x_2$. It values the consumption of those goods with the joint utility function

$$
u\left(x_1, x_2\right)=x_1^{0.4}+\left(1+x_2\right)^{0.5} \text {. }
$$

Here $x_2$ acts as a luxury good, i.e. the household will only consume $x_2$, if its available resources $W$ are large enough. $x_1$ on the other hand is a normal good and will always be consumed. Naturally, we have to assume that $x_1, x_2 \geq 0$. With the prices for the goods being $p_1$ and $p_2$, the household has to solve the optimization problem

$$
\max _{x_1, x_2 \geq 0} x_1^{0.4}+\left(1+x_2\right)^{0.5} \quad \text { s.t. } \quad p_1 x_1+p_2 x_2=W .
$$

Note that there is no analytical solution to this problem.

---

This is a constrained optimization problem. In this case, the constraint is the household's budget. 

The utility maximization problem involves maximizing a function subject to a constraint. In this case, we want to maximize the utility function `u(x1, x2) = x1^0.4 + (1 + x2)^0.5` subject to the budget constraint `p1*x1 + p2*x2 = W`.

In Julia, the `JuMP.jl` and `Ipopt.jl` packages can be used to solve this problem. `JuMP` is a package for defining and solving optimization models, and `Ipopt` is a solver that can handle non-linear optimization problems.

Here is an example of how to solve this problem, assuming that `p1 = 1`, `p2 = 2`, and `W = 10`:

```julia
using JuMP, Ipopt

# Parameters
p1 = 1.0
p2 = 2.0
W = 10.0

# Define the model
model = Model(Ipopt.Optimizer)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)

@NLobjective(model, Max, x1^0.4 + (1 + x2)^0.5)

@constraint(model, p1*x1 + p2*x2 == W)

# Solve the model
optimize!(model)

# Get the optimal solution
optimal_x1 = value(x1)
optimal_x2 = value(x2)

println("The optimal consumption of good 1 is ", optimal_x1)
println("The optimal consumption of good 2 is ", optimal_x2)
```

Please ensure that you have the JuMP and Ipopt packages installed and imported. You can install them via `using Pkg; Pkg.add("JuMP"); Pkg.add("Ipopt")`.

This code defines two variables `x1` and `x2` (with the constraint that they are nonnegative), and then defines an objective function to maximize and a constraint for the budget. The function `optimize!` is then called to solve the problem, and the optimal values of `x1` and `x2` are extracted with the `value` function.

Note that the choice of `p1`, `p2`, and `W` is arbitrary, and you may need to adjust these values based on your specific problem and domain knowledge.

## Example 5

---

Example Consider an agricultural commodity market, where planting decisions are based on the price expected at harvest

$$
A=0.5+0.5 E(p)
$$

with $A$ denoting acreage supply and $E(p)$ defining expected price. After the acreage is planted, a normally distributed random yield $y \sim N(1,0.1)$ is realized, giving rise to the quantity $q^s=A y$ which is sold at the market clearing price $p=3-2 q^s$.

In order to solve this system we substitute

$$
q^s=[0.5+0.5 E(p)] y
$$

and therefore

$$
p=3-2[0.5+0.5 E(p)] y .
$$

Taking expectations on both sides leads to

$$
E(p)=3-2[0.5+0.5 E(p)] E(y)
$$

and therefore $E(p)=1$. Consequently, equilibrium acreage is $A=1$. Finally, the equilibrium price distribution has a variance of

```math
\text{{Var}}(p) = 4[0.5+0.5E(p)]^2 \quad \text{{Var}}(y) = 4 \quad \text{{Var}}(y) = 0.4 .
```
Suppose now that the government introduces a price support program which guarantees each producer a minimum price of 1 . If the market price falls below this level, the government pays the producer the difference per unit produced. Consequently, the producer now receives an effective price of $\max (p, 1)$ and the expected price in $(2.10)$ is then calculated via

$$
E(p)=E[\max (3-2 A y, 1)]
$$

The equilibrium acreage supply finally is the supply $A$ that fulfils (2.10) with the above price expectation. Again, this problem cannot be solved analytically.

---

This is a fairly complex problem which involves solving for the equilibrium given an expected price which depends on a support price mechanism. 

The problem can be solved using Monte Carlo simulations. The idea is to simulate the price distribution under the new policy and then adjust the acreage until the expected price is consistent with the acreage choice. 

In Julia, we can use the `Distributions.jl` package to generate draws from the yield distribution, and then calculate the corresponding price, taking into account the support price. We can then use a simple iteration process to find the equilibrium acreage.

Here is a sample code on how you might implement this in Julia:

```julia
using Distributions
using Statistics

# Initial parameters
A = 0.5  # initial guess for acreage
n_draws = 10000  # number of Monte Carlo draws
yield_dist = Normal(1, 0.1)  # distribution of yields

# Function to calculate the effective price given acreage and yield
function effective_price(A, y)
    p = 3 - 2 * A * y
    return max(p, 1)  # apply price support
end

# Monte Carlo simulation
function calculate_expected_price(A)
    y_draws = rand(yield_dist, n_draws)  # draw yields
    p_draws = effective_price.(A, y_draws)  # calculate prices
    return mean(p_draws)  # calculate expected price
end

# Iteration process
tol = 1e-5
max_iter = 1000
for i in 1:max_iter
    E_p = calculate_expected_price(A)
    A_new = 0.5 + 0.5 * E_p
    if abs(A_new - A) < tol
        break
    else
        A = A_new
    end
end

println("The equilibrium acreage is ", A)
```

This script first defines an initial guess for acreage `A` and the number of Monte Carlo draws `n_draws`, then defines the yield distribution. The `effective_price` function calculates the price given the acreage and yield, taking into account the price support mechanism. The `calculate_expected_price` function then calculates the expected price given the acreage by drawing yields from the yield distribution, calculating the corresponding prices, and then calculating the average price. Finally, an iteration process is performed to find the acreage that leads to an expected price that is consistent with the acreage choice.

Please note that the quality of the solution depends on the number of Monte Carlo draws and the tolerance level for the iteration process. You might need to adjust these parameters based on your specific problem and domain knowledge.

## Example 6

---

Example Consider a company that produces two goods $x_1$ and $x_2$ on one type of machine. Due to storage capacity, the production is limited to 100 pieces altogether. The production of the two goods requires time and raw material. Good 1 thereby is a time-and cost-intensive production good which requires four hours of production time and raw material worth $€ 20$ per piece. Good 2 is less intensive and can therefore be produced within one hour and with raw material worth $€ 10$. The company owns eight machines which can run 20 hours a day. The remaining time has to be spent on maintenance. In addition, overall costs of raw material should be limited to $€ 1,100$ per day. The two goods can be sold on the market for $€ 120$ and $€ 40$, respectively. What is the optimal production of goods per day for the company?

Table 2.2 summarizes the problem. The goal of the company obviously is to maximize its profits $120 x_1+40 x_2$ subject to the given production constraints. This problem can be written as a linear program in standard form

$$
\min _{x_1, x_2}-120 x_1-40 x_2, \quad x_1, x_2 \geq 0
$$

subject to the constraints

$$
\begin{aligned}
x_1+x_2 & \leq 100 \\
4 x_1+x_2 & \leq 160 \\
20 x_1+10 x_2 & \leq 1100 .
\end{aligned}
$$

Note that we again just minimize the negative of the objective function in order to maximize it. The same approach was taken in Section 2.3 that dealt with the minimization of nonlinear functions.

---

Certainly! Here's the complete code with explanations:

using JuMP
using GLPK

# Create a model
model = Model(optimizer_with_attributes(GLPK.Optimizer, "msg_lev" => GLPK.GLP_MSG_OFF))

# Variables
@variable(model, x1 >= 0)
@variable(model, x2 >= 0)

# Objective function
@objective(model, Max, 120x1 + 40x2)

# Constraints
@constraint(model, x1 + x2 <= 100)
@constraint(model, 4x1 + x2 <= 160)
@constraint(model, 20x1 + 10x2 <= 1100)

# Solve the model
optimize!(model)

# Check the status of the solution
if termination_status(model) == MOI.OPTIMAL
    # Get the optimal solution
    optimal_x1 = value(x1)
    optimal_x2 = value(x2)

    # Print the optimal production
    println("The optimal production of good 1 is ", optimal_x1)
    println("The optimal production of good 2 is ", optimal_x2)
else
    println("No optimal solution found.")
end


In this code, we start by importing the necessary packages, `JuMP` and `GLPK`.

We then create the optimization model using `model = Model(optimizer_with_attributes(GLPK.Optimizer, "msg_lev" => GLPK.GLP_MSG_OFF))`. This line sets up the model with GLPK as the solver and turns off the solver's output messages.

Next, we define the decision variables `x1` and `x2` using `@variable(model, x1 >= 0)` and `@variable(model, x2 >= 0)`. These variables represent the production quantities of goods 1 and 2, respectively.

The objective function is defined with `@objective(model, Max, 120x1 + 40x2)`, specifying that we want to maximize the total profit, which is the sum of the profits from producing goods 1 and 2.

The constraints are defined using `@constraint(model, ...)` statements. We have three constraints: the total production should not exceed 100 units, the total production time of goods 1 and 2 should not exceed 160 hours, and the total cost of raw materials should not exceed 1100.

Next, we set the solver options using `set_optimizer_attribute(model, "LPMethod", GLPK.GLP_DUAL)`. This line sets the solver to use the dual simplex method for solving the linear programming relaxation of the model.

We then solve the model using `optimize!(model)`. The solver will find the optimal solution that maximizes the objective function while satisfying the given constraints.

After solving the model, we check the status of the solution using `termination_status(model)`. If the status is `MOI.OPTIMAL`, which represents an optimal solution, we retrieve the optimal values of `x1` and `x2` using `value(x1)` and `value(x2)`.

Finally, we print the optimal production quantities of goods 1 and 2 using `println`. If the solver does not find an optimal solution, we print a message indicating that no optimal solution was found.

Make sure you have the JuMP and GLPK packages installed and imported. If you encounter any errors, ensure that you have the latest versions of the packages by running `using Pkg; Pkg.update()`.
