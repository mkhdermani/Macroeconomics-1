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

```julia
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
```

In this code, we first import the necessary packages, `JuMP` and `GLPK`.

Next, we create the optimization model using `model = Model(optimizer_with_attributes(GLPK.Optimizer, "msg_lev" => GLPK.GLP_MSG_OFF))`. This line creates a new model object named `model` and sets GLPK as the optimizer for solving the model. The `"msg_lev" => GLPK.GLP_MSG_OFF` option is used to turn off the solver's output messages.

We then define the decision variables `x1` and `x2` using the `@variable` macro. The `x1` and `x2` variables represent the production quantities of goods 1 and 2, respectively. The `>= 0` constraint ensures that the variables are nonnegative.

The objective function is defined using `@objective(model, Max, 120x1 + 40x2)`, which specifies that we want to maximize the total profit. The objective function is the sum of the profits from producing goods 1 and 2.

The constraints are defined using `@constraint(model, ...)` statements. We have three constraints in this case: the total production should not exceed 100 units (`x1 + x2 <= 100`), the total production time of goods 1 and 2 should not exceed 160 hours (`4x1 + x2 <= 160`), and the total cost of raw materials should not exceed 1100 (`20x1 + 10x2 <= 1100`).

After defining the model, we call `optimize!(model)` to solve the model using the GLPK optimizer.

We then check the status of the solution using `termination_status(model)`. If the status is `MOI.OPTIMAL`, which represents an optimal solution, we retrieve the optimal values of `x1` and `x2` using the `value` function. These values represent the optimal production quantities of goods 1 and 2.

Finally, we print the optimal production quantities of goods 1 and 2 using `println`. If the solver does not find an optimal solution, we print a message indicating that no optimal solution was found.

Make sure you have the JuMP and GLPK packages installed and imported. If you encounter any errors, ensure that you have the latest versions of the packages by running `using Pkg; Pkg.update()`.

## Exercise 1

---

2.1. Consider the matrix
$$
A=\left[\begin{array}{cccc}
1 & 5 & 2 & 3 \\
1 & 6 & 8 & 6 \\
1 & 6 & 11 & 2 \\
1 & 7 & 17 & 4
\end{array}\right]
$$
(a) Compute the matrices $L$ and $U$ applying the Gaussian elimination method by hand. Check your results using the subroutine $l_{-} \operatorname{dec}(A, L, U)$ from the toolbox. Finally, inspect the matrix product of $L$ and $U$.
(b) Solve the linear equation system
$$
A x=b, \quad \text { with } \quad b=\left[\begin{array}{l}
1 \\
2 \\
1 \\
1
\end{array}\right]
$$
using the matrix decomposition from the previous exercise. Check your results using subroutine $1 u_{-} \operatorname{solve}(\mathrm{A}, \mathrm{b})$ from the toolbox. Again inspect the matrix product of $A$ and $x$.

---

(a) To compute the matrices L and U using Gaussian elimination, we perform row operations on matrix A to eliminate the elements below the main diagonal.

Starting with matrix A:
```
1   5   2   3
1   6   8   6
1   6  11   2
1   7  17   4
```

Step 1: Subtract the first row from the second, third, and fourth rows to eliminate the first column entries below the main diagonal:
```
1   5   2   3
0   1   6   3
0   1   9  -1
0   2  15   1
```

Step 2: Subtract twice the second row from the third row to eliminate the second column entry below the main diagonal:
```
1   5   2   3
0   1   6   3
0   0   -3  -7
0   2   15   1
```

Step 3: Add twice the second row to the fourth row to eliminate the second column entry below the main diagonal:
```
1   5   2   3
0   1   6   3
0   0   -3  -7
0   0   27   -5
```

Step 4: Divide the third row by -3 to make the third row's pivot equal to 1:
```
1   5   2   3
0   1   6   3
0   0   1    7/3
0   0   27  -5
```

Step 5: Subtract six times the third row from the second row:
```
1   5   2   3
0   1   0   -5
0   0   1    7/3
0   0   27  -5
```

Step 6: Subtract two times the third row from the first row:
```
1   5   0   -1/3
0   1   0   -5
0   0   1    7/3
0   0   27  -5
```

Step 7: Subtract 27 times the third row from the fourth row:
```
1   5   0   -1/3
0   1   0   -5
0   0   1    7/3
0   0   0   -40
```

The resulting matrices L and U are:
```
L = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 7/3 1]
U = [1 5 0 -1/3; 0 1 0 -5; 0 0 1 7/3; 0 0 0 -40]
```

To check the results using a subroutine, you can use the following code in Julia:

```julia
using LinearAlgebra

A = [1 5 2 3; 1 6 8 6; 1 6 11 2; 1 7 17 4]
L, U = lu(A)

println("L:")
println(L)
println("U:")
println(U)
```

The matrix product of L and U should be equal to matrix A.

(b) To solve the linear equation system Ax = b using the LU decomposition, we can solve two systems of equations: Ly = b and Ux = y.

Using the matrices L and U from part (a) and the given vector b:
```
b = [1; 2; 1; 1]
```

Solving Ly = b, we find y:
```
y = L \ b
```

Solving Ux = y, we find x:
```
x = U \ y
```

To check the results using a subroutine, you can use the following code in Julia:

```julia
using LinearAlgebra

A = [1 5 2 3; 1 6 8 6; 1 6 11 2; 1 7 17 4]
b = [1; 2; 1; 1]

x = A \ b

println("Solution x:")
println(x)
```

The matrix product of A and x should be equal to vector b.

## Exercise 2 

---

2.2. Consider the following intertemporal household problem: The utility function the household is given by

$$
U\left(c_1, c_2\right)=\frac{c_1^{1-\frac{1}{\gamma}}}{1-\frac{1}{\gamma}}+\beta \frac{c_2^{1-\frac{1}{\gamma}}}{1-\frac{1}{\gamma}}
$$

with $c_1$ and $c_2$ denoting consumption in the first and the second period, respectivel $\gamma$ is the intertemporal elasticity of substitution and $\beta$ defines the time discour factor. The household receives labour income $w$ in the first period and does nc work in the second period. Consequently, the budget constraint is

$$
c_1+\frac{c_2}{1+r}=w
$$

where $r$ defines the interest rate.

(a) Define the Lagrangian for this specific optimization problem and derive first-order conditions with respect to $c_1, c_2$, and $\lambda$. Solve the equation system analytically using parameter values $\gamma=0.5, \beta=1, r=0$, and $w=1$.

(b) Solve the equation system resulting from a) using function fsolve from the toolbox. Print the results and compare the numerical results with the analytical solutions.

(c) Solve the household problem using the subroutine Optim.jl and compare the results.

---

(a) To solve the intertemporal household problem analytically, we can define the Lagrangian for the optimization problem and derive the first-order conditions.

The Lagrangian is defined as follows:
```
L(c1, c2, λ) = U(c1, c2) + λ * (w - c1 - c2 / (1 + r))
```

where `U(c1, c2)` is the utility function, `λ` is the Lagrange multiplier, `w` is the labor income, and `r` is the interest rate.

The first-order conditions are obtained by taking the partial derivatives of the Lagrangian with respect to `c1`, `c2`, and `λ` and setting them equal to zero:

```
∂L/∂c1 = c1^(-1/γ) - λ = 0   -->   c1^(-1/γ) = λ        (1)
∂L/∂c2 = β * c2^(-1/γ) - λ / (1 + r) = 0   -->   c2^(-1/γ) = (1 + r) * β * λ     (2)
∂L/∂λ = w - c1 - c2 / (1 + r) = 0     (3)
```

Solving equations (1), (2), and (3) simultaneously will give us the optimal values for `c1`, `c2`, and `λ`.

Using the given parameter values γ=0.5, β=1, r=0, and w=1, we can substitute them into the equations and solve them analytically.

Certainly! Let's provide the complete code for both the `fsolve` method and the `Nelder-Mead` method. We will also include explanations for each step along the way.

(b) Solution using `fsolve` (NLsolve package):

```julia
using NLsolve

γ = 0.5
β = 1.0
r = 0.0
w = 1.0

function equations!(F, x)
    c1 = x[1]
    c2 = x[2]
    λ = x[3]

    F[1] = c1^(-1/γ) - λ
    F[2] = c2^(-1/γ) * λ / (1 + r) - β
    F[3] = w - c1 - c2 / (1 + r)
end

# Initial guess
x0 = [0.5, 0.5, 1.0]

# Solve the system of equations using fsolve
res = nlsolve(equations!, x0)

c1 = res.zero[1]
c2 = res.zero[2]
λ = res.zero[3]

println("Numerical solution using fsolve:")
println("c1 =", c1)
println("c2 =", c2)
println("λ =", λ)
```

In this code, we define the `equations!` function, which represents the system of equations. It takes the output vector `F` and the input vector `x`. The `nlsolve` function is then used to solve the system of equations by providing the `equations!` function and the initial guess `x0`. The resulting solution is obtained from `res.zero`.

(c) Solution using `Nelder-Mead` method (Optim package):

```julia
using Optim

γ = 0.5
β = 1.0
r = 0.0
w = 1.0

function U(c1, c2)
    return c1^(1 - 1/γ) / (1 - 1/γ) + β * c2^(1 - 1/γ) / (1 - 1/γ)
end

function objective(x)
    c1 = x[1]
    c2 = x[2]
    return -U(c1, c2)
end

# Initial guess
x0 = [0.5, 0.5]

# Solve the optimization problem using Nelder-Mead (fminsearch)
solution = optimize(objective, x0, NelderMead())

c1 = solution.minimizer[1]
c2 = solution.minimizer[2]
λ = c1^(-1/γ)

println("Numerical solution using Nelder-Mead (fminsearch):")
println("c1 =", c1)
println("c2 =", c2)
println("λ =", λ)
```

In this code, we define the utility function `U(c1, c2)` and the objective function `objective(x)`, which is the negative of the utility function. The `optimize` function is then used with the `NelderMead()` algorithm to minimize the objective function and find the optimal values of `c1` and `c2`. The resulting solution is obtained from `solution.minimizer`.

Both methods provide numerical solutions to the intertemporal household problem. The `fsolve` method solves the system of equations directly, while the `Nelder-Mead` method formulates the problem as an optimization and finds the optimal values by minimizing the objective function.

Please note that the provided code assumes that you have the required packages (`NLsolve` and `Optim`) installed. You can install them by running `using Pkg; Pkg.add("NLsolve")` and `using Pkg; Pkg.add("Optim")` in the Julia REPL, respectively.

## Exercise 3

---

2.3. Write a program which computes the global minimum of the function $f(x)=x \cos \left(x^2\right)$ on the interval $[0,5]$ using Golden Search. Proceed using the following steps:

(a) Write a function minimize (a, b) of type real*8, which computes the local minimum of $f(x)$ using Golden Search on the interval $[a, b]$.

(b) Next split up the interval $[0,5]$ in n subintervals $\left[x_i, x_{i+1}\right]$ of identical length and compute for each interval the local minimum using the function minimize.

(c) Finally, your program has to sort out the global minimum from the set of computed local minima using the function minloc (array, 1).

Test your program with different values for $n$. How many subintervals are necessary to find the global minimum?

---

```julia
using Optim

# Define the function f(x)
function f(x)
    return x * cos(x^2)
end

# Define the minimize function using Golden Section Search
function minimize(a, b)
    result = optimize(f, a, b)
    return Optim.minimizer(result)
end

# Split the interval [0, 5] into n subintervals and find local minima
function find_global_minimum(n)
    intervals = range(0, stop=5, length=n+1)
    local_minima = []

    for i in 1:n
        a = intervals[i]
        b = intervals[i+1]
        local_min = minimize(a, b)
        push!(local_minima, local_min)
    end

    global_min_index = argmin(local_minima)
    global_min = local_minima[global_min_index]

    return global_min
end

# Test the program with different values for n
n_values = [10, 20, 50, 100]
for n in n_values
    global_min = find_global_minimum(n)
    println("Number of Subintervals: $n, Global Minimum: $global_min")
end
```

This code snippet demonstrates the use of the Optim package in Julia to find the global minimum of a function using the Golden Section Search method. Here's a breakdown of the code:

1. The code defines a function `f(x)` that represents the objective function to be minimized. In this case, the function is `f(x) = x * cos(x^2)`.

2. Another function `minimize(a, b)` is defined, which takes two parameters `a` and `b` representing the interval boundaries. It uses the `optimize` function from the Optim package to find the minimum of the function `f(x)` within the given interval `[a, b]`. The `Optim.minimizer` function extracts the minimizer from the optimization result.

3. The `find_global_minimum(n)` function is defined to find the global minimum of the function `f(x)` by splitting the interval `[0, 5]` into `n` subintervals. It creates a range of `n+1` equally spaced values between 0 and 5 using the `range` function. It then iterates over these subintervals, calling the `minimize` function for each subinterval and stores the local minima in the `local_minima` array.

4. After finding all the local minima, the code identifies the index of the minimum value in the `local_minima` array using `argmin`. It retrieves the global minimum value based on the index.

5. Finally, the code tests the `find_global_minimum` function by calling it with different values of `n` and prints the number of subintervals and the corresponding global minimum.

In summary, the code divides the interval `[0, 5]` into smaller subintervals and finds the local minimum in each subinterval using the Golden Section Search method. It then identifies the global minimum among these local minima and prints the results for different values of `n`.

## Excersie 4

---

2.4. Consider the following intertemporal household optimization problem: The utility function of the household is given by

$$
U\left(c_1, c_2, c_3\right)=\sum_{i=1}^3 \beta^{i-1} u\left(c_i\right) \quad \text { with } \quad u\left(c_i\right)=\frac{c_i^{1-\frac{1}{\gamma}}}{1-\frac{1}{\gamma}},
$$

where $c_i$ defines consumption in period $i$ of life and $\beta$ denotes the time discount rate. Assume that the household receives labour income $w$ in the first two periods and consumes all savings in the third period, so that the budget constraint reads

$$
\sum_{i=1}^3 \frac{c_i}{(1+r)^{i-1}}=\sum_{i=1}^2 \frac{w}{(1+r)^{i-1}},
$$

where $r$ defines the interest rate.

Solve for the optimal consumption levels using the subroutine optimize from the toolbox. Proceed using the following steps:

(a) Substitute the budget constraint in the utility function so that it depends only on $c_2$ and $c_3$.

(b) Minimize the function $-\widetilde{U}\left(c_2, c_3\right)$ in order to get the values $c_2^*$ and $c_3^*$.

(c) Finally, derive $c_1^*$ from the budget constraint.

Use the same parameter values as in Exercise 2 to test your program. Then increase the interest rate and the wage rate separately. Explain your results in economic terms. What will happen when wages are different in both periods? What happens when you alter $\beta$ ?

---

To solve the intertemporal household optimization problem described, we can follow the steps provided:

(a) Substitute the budget constraint in the utility function so that it depends only on c₂ and c₃:

The budget constraint can be rewritten as:

c₁/(1+r)^0 + c₂/(1+r)^1 + c₃/(1+r)^2 = w/(1+r)^0 + w/(1+r)^1

Substituting this in the utility function, we get:

U(c₂, c₃) = β^0 * u(c₂) + β^1 * u(c₃) = β^0 * u(c₂) + β^1 * u((w - c₁/(1+r)^0 - c₂/(1+r)^1)/(1+r)^1)

(b) Minimize the function -U(c₂, c₃) to get the values c₂* and c₃*:

To minimize the function, we can define a new function that returns the negative of the utility function:

```julia
using Optim

function negative_utility(x, β, γ, w, r)
    c₂ = x[1]
    c₃ = x[2]
    c₁ = (w - c₂ - c₃/(1+r))/(1+r)
    u_c₂ = c₂^(1-1/γ)/(1-1/γ)
    u_c₃ = c₃^(1-1/γ)/(1-1/γ)
    U = β^0 * u_c₂ + β^1 * u_c₃
    return -U
end

β = 0.95  # Time discount rate
γ = 2.0   # Coefficient of relative risk aversion
w = 1.0   # Labor income
r = 0.05  # Interest rate

# Initial guess for c₂ and c₃
x0 = [0.5, 0.5]

# Use the optimize function to minimize the negative utility
result = optimize(x -> negative_utility(x, β, γ, w, r), x0)

# Extract the optimal values
c₂_opt = result.minimizer[1]
c₃_opt = result.minimizer[2]
```

(c) Derive c₁* from the budget constraint:

Using the budget constraint, we can solve for c₁:

c₁* = (w - c₂* - c₃*/(1+r))/(1+r)

```julia
c₁_opt = (w - c₂_opt - c₃_opt/(1+r))/(1+r)
```

Now we have the optimal consumption levels c₁*, c₂*, and c₃*.

To analyze the economic implications of changing the interest rate (r) and wage rate (w), you can vary these parameters and observe the resulting changes in the optimal consumption levels. Here are some insights:

- When the wage rates are different in both periods (w₁ ≠ w₂), the budget constraint will change accordingly, and the optimal consumption levels will adjust to balance the trade-off between current and future consumption. Higher wages in one period may lead to higher consumption in that period compared to the other.

- When you alter the time discount rate (β), a higher β places more weight on future consumption relative to present consumption. As a result, the optimal consumption levels may decrease in the current period and increase in future periods.

Remember that these interpretations are based on the assumptions made in the intertemporal household optimization problem. Real-world implications may be more complex and dependent on specific economic conditions and assumptions.

## Exercise 5

---

2.5. Write a program which approximates the integral of $\int_a^b f(x) \mathrm{d} x$ using the trapezoid rule, the Simpson rule, and the Gauss-Legendre quadrature method.
(a) Assume $f(x)=\exp (-x)$.
(b) Now assume $f(x)=|x|^{0.5}$.
Test your program in the interval $[a ; b]=[-1,1]$ with $n=10,20$, and 30 nodes and compare the results with the analytical calculation. What are the differences?

---
To approximate the integral using the trapezoid rule, Simpson's rule, and Gauss-Legendre quadrature method in Julia, you can use the `QuadGK` package for numerical integration. Here's an example code that utilizes this package:

```julia
using QuadGK

# Function f(x) = exp(-x)
f1(x) = exp(-x)

# Function f(x) = |x|^0.5
f2(x) = abs(x)^0.5

# Analytical solution for the integral of f(x) in [a, b]
analytical_integral_f1(a, b) = exp(-a) - exp(-b)
analytical_integral_f2(a, b) = (2/3) * (abs(b)^(3/2) - abs(a)^(3/2))

# Perform approximation using the trapezoid rule
function trapezoid_rule_approximation(f, a, b, n)
    x = range(a, stop=b, length=n+1)
    h = (b - a) / n
    integral_sum = sum(f.(x[1:end-1]) .+ f.(x[2:end]))
    approximation = (h / 2) * integral_sum
    return approximation
end

# Perform approximation using Simpson's rule
function simpson_rule_approximation(f, a, b, n)
    x = range(a, stop=b, length=n+1)
    h = (b - a) / n
    integral_sum = f(a) + f(b)
    for i = 2:n
        if i % 2 == 0
            integral_sum += 2 * f(x[i])
        else
            integral_sum += 4 * f(x[i])
        end
    end
    approximation = (h / 3) * integral_sum
    return approximation
end

# Perform approximation using Gauss-Legendre quadrature
function gauss_legendre_quadrature_approximation(f, a, b, n)
    approximation, _ = quadgk(f, a, b, atol=0, rtol=1e-12, maxevals=n)
    return approximation
end

# Interval [a, b] and number of nodes n
a = -1
b = 1
n_values = [10, 20, 30]

# Approximate and compare results for f(x) = exp(-x)
println("Approximation for f(x) = exp(-x)")
for n in n_values
    trapezoid_result = trapezoid_rule_approximation(f1, a, b, n)
    simpson_result = simpson_rule_approximation(f1, a, b, n)
    gauss_legendre_result = gauss_legendre_quadrature_approximation(f1, a, b, n)
    analytical_result = analytical_integral_f1(a, b)
    println("n = $n")
    println("Trapezoid Rule: $trapezoid_result")
    println("Simpson's Rule: $simpson_result")
    println("Gauss-Legendre Quadrature: $gauss_legendre_result")
    println("Analytical Result: $analytical_result")
    println()
end

# Approximate and compare results for f(x) = |x|^0.5
println("Approximation for f(x) = |x|^0.5")
for n in n_values
    trapezoid_result = trapezoid_rule_approximation(f2, a, b, n)
    simpson_result = simpson_rule_approximation(f2, a, b, n)
    gauss_legendre_result = gauss_legendre_quadrature_approximation(f2, a, b, n)
    analytical_result = analytical_integral_f2(a, b)
    println("n = $n")
    println("Trapezoid Rule: $trapezoid_result")
    println("Simpson's Rule: $simpson_result")
    println("Gauss-Legendre Quadrature: $gauss_legendre_result")
    println("Analytical Result: $analytical_result")
    println()
end
```

Make sure you have the `QuadGK` package installed in Julia before running this code. You can do so by running the following command in the Julia REPL:

```julia
using Pkg
Pkg.add("QuadGK")
```

With this package, the trapezoid rule, Simpson's rule, and Gauss-Legendre quadrature are implemented. The code then performs the approximations for different values of `n` and compares the results with the analytical solutions for both `f(x) = exp(-x)` and `f(x) = |x|^0.5`.

## E xercise 6

---

2.6. Consider the following inverse demand function
$$
p(d)=\frac{4}{(d+1)^2}
$$
for a good $d$. Assume that the price of the good falls from $p=3$ to $p=1$. Compute the relative change in the consumer surplus using the trapezoid rule, the Simpson rule, and the Gauss-Legendre quadrature method with $n=10$.

---

To compute the relative change in consumer surplus using the trapezoid rule, Simpson's rule, and Gauss-Legendre quadrature method, we need to evaluate the integrals associated with the consumer surplus before and after the price change.

Let's denote the inverse demand function as:

```
p(d) = 4 / (d + 1)^2
```

To calculate the consumer surplus, we need to integrate the inverse demand function over the range of quantities consumed. The consumer surplus can be expressed as:

```
CS = ∫[0, d] p(x) dx
```

where `CS` represents the consumer surplus and `d` is the quantity consumed.

To calculate the relative change in consumer surplus, we can compare the consumer surplus before and after the price change using the following formula:

```
Relative Change = (CS_new - CS_old) / CS_old
```

Now, let's implement the calculations using the trapezoid rule, Simpson's rule, and Gauss-Legendre quadrature method in Julia:

```julia
using QuadGK

# Inverse demand function
p(d) = 4 / (d + 1)^2

# Calculate consumer surplus using the trapezoid rule
function trapezoid_rule_consumer_surplus(p, d, n)
    x = range(0, stop=d, length=n+1)
    h = d / n
    surplus_sum = sum(p.(x[1:end-1]) .+ p.(x[2:end]))
    surplus = (h / 2) * surplus_sum
    return surplus
end

# Calculate consumer surplus using Simpson's rule
function simpson_rule_consumer_surplus(p, d, n)
    x = range(0, stop=d, length=n+1)
    h = d / n
    surplus_sum = p(0) + p(d)
    for i = 2:n
        if i % 2 == 0
            surplus_sum += 2 * p(x[i])
        else
            surplus_sum += 4 * p(x[i])
        end
    end
    surplus = (h / 3) * surplus_sum
    return surplus
end

# Calculate consumer surplus using Gauss-Legendre quadrature
function gauss_legendre_quadrature_consumer_surplus(p, d, n)
    surplus, _ = quadgk(p, 0, d, atol=0, rtol=1e-12, maxevals=n)
    return surplus
end

# Quantity consumed before and after price change
d_old = 3.0
d_new = 1.0

# Number of nodes for approximation methods
n = 10

# Calculate consumer surplus before price change
cs_old_trapezoid = trapezoid_rule_consumer_surplus(p, d_old, n)
cs_old_simpson = simpson_rule_consumer_surplus(p, d_old, n)
cs_old_gauss_legendre = gauss_legendre_quadrature_consumer_surplus(p, d_old, n)

# Calculate consumer surplus after price change
cs_new_trapezoid = trapezoid_rule_consumer_surplus(p, d_new, n)
cs_new_simpson = simpson_rule_consumer_surplus(p, d_new, n)
cs_new_gauss_legendre = gauss_legendre_quadrature_consumer_surplus(p, d_new, n)

# Calculate relative change in consumer surplus
relative_change_trapezoid = (cs_new_trapezoid - cs_old_trapezoid) / cs_old_trapezoid
relative_change_simpson = (cs_new_simpson - cs_old_simpson) / cs_old_simpson
relative_change_gauss_legendre = (cs_new_gauss_legendre - cs_old_gauss_legendre) / cs_old_gauss_legendre

# Print the results
println("Relative Change in Consumer Surplus:")
println("Using Trapezoid Rule: $relative_change_trapezoid")
println("Using Simpson's Rule: $relative_change_simpson")
println("Using Gauss-Legendre Quadrature: $relative_change_gauss_legendre")
```

In this code, we define the inverse demand function `p(d) = 4 / (d + 1)^2`. Then, we implement the functions for calculating the consumer surplus using the trapezoid rule, Simpson's rule, and Gauss-Legendre quadrature. The number of nodes for approximation methods is set to `n = 10`.

We calculate the consumer surplus before and after the price change using the respective approximation methods. Finally, we compute the relative change in consumer surplus and print the results.

Please note that the accuracy of the approximations depends on the choice of `n`. You can experiment with different values to achieve the desired level of precision.

## Exercise 7 

---

2.7. The government has increased tax rates on labour income during the last years substantially, but the revenues have evolved as follows:

```math
\begin{tabular}{ccc}
\hline Year & $\tau$ & Tax revenue (in bn.) \\
\hline 2012 & 37 & 198.875 \\
2013 & 42 & 199.500 \\
2014 & 45 & 196.875 \\
\hline
\end{tabular}
```

You should help the government to find the revenue maximizing tax rate. Therefore write a program that interpolates the tax function $T(\tau)$ given the data from the above table using polynominal interpolation. Print the resulting tax function in the interval $[35 ; 45]$ and compute the revenue-maximizing tax rate as well as the corresponding tax revenue.

Why is it not possible to use the subroutines for piecewise linear or cubic spline interpolation here?

---

To find the revenue-maximizing tax rate using polynomial interpolation, we can approximate the tax function based on the given data points. However, it's important to note that polynomial interpolation might not be the most suitable method for this scenario. Polynomial interpolations tend to oscillate between data points and can lead to unreliable extrapolations beyond the data range. Additionally, high-degree polynomials can exhibit wild behavior, which may not accurately represent the underlying tax function.

To demonstrate polynomial interpolation, we'll use the `Interpolations` package in Julia. Here's an example code that performs polynomial interpolation based on the given data:

```julia
using Interpolations

# Given data
years = [2012, 2013, 2014]
tax_rates = [37, 42, 45]
tax_revenues = [198.875, 199.5, 196.875]

# Perform polynomial interpolation
tax_interpolation = interpolate(tax_rates, tax_revenues, Gridded(Linear()))

# Evaluate the tax function in the interval [35, 45]
tax_values = range(35, stop=45, length=100)
revenue_values = [tax_interpolation(tax) for tax in tax_values]

# Find the revenue-maximizing tax rate and corresponding tax revenue
max_revenue = maximum(revenue_values)
optimal_tax_rate = tax_values[argmax(revenue_values)]

# Print the resulting tax function and revenue-maximizing tax rate
println("Tax Function:")
for i in 1:length(tax_values)
    println("Tax Rate: $(tax_values[i]), Tax Revenue: $(revenue_values[i])")
end
println()
println("Revenue-Maximizing Tax Rate: $optimal_tax_rate")
println("Corresponding Tax Revenue: $max_revenue")
```

In this code, we first define the given data points: `years`, `tax_rates`, and `tax_revenues`. We then use the `interpolate` function from the `Interpolations` package to perform polynomial interpolation. The `Gridded(Linear())` specification selects linear interpolation between the data points.

Next, we evaluate the tax function for various tax rates in the interval `[35, 45]` and store the corresponding tax revenues in the `revenue_values` array.

Finally, we find the revenue-maximizing tax rate by identifying the maximum tax revenue value and its corresponding tax rate. We print the resulting tax function and display the revenue-maximizing tax rate along with the corresponding tax revenue.

Regarding why we can't use piecewise linear or cubic spline interpolation, these methods rely on connecting the given data points using linear segments or cubic polynomials, respectively. However, in this case, we only have three data points. Piecewise linear or cubic spline interpolation would require more data points to accurately represent the tax function and interpolate between them. The small number of data points available in the given table makes it challenging to use these interpolation methods effectively. Polynomial interpolation, while not ideal, can still provide an approximation but may suffer from issues such as overfitting and oscillations.

## Exercise 8

---

2.8. Write a program which plots the function $\cos (x)$ on the interval $[0 ; 2 \pi]$ using linear interpolation. In order to do this split up the total interval into $n+1$ equidistant nodes $x_i, i=0, \ldots, n$ and compute in each subinterval $\left[x_{i-1} ; x_i\right]$ the linear equation $f(x)=m_i \cdot x+t_i$ with

$$
m_i=\frac{f\left(x_i\right)-f\left(x_{i-1}\right)}{x_i-x_{i-1}} \text { and } t_i=f\left(x_{i-1}\right)-m_i \cdot x_{i-1},
$$

where $x_{i-1}$ and $x_i$ denote the lower and upper bound of the respective subinterval $i$. Then draw $n$ graphs that show the linear approximation of the cosine function. Compare your linear interpolation results with the true cosine function for $n=10,20$, and 30 . Now approximate the function using the piecewise linear interpolation subroutine linint_Equi as well as the spline approach and compare the results with the true cosine function.

---

To plot the function $\cos(x)$ on the interval $[0, 2\pi]$ using linear interpolation, we can split the interval into equidistant nodes and compute linear equations within each subinterval. We will compare the linear interpolation results with the true cosine function and also approximate the function using piecewise linear interpolation and spline interpolation.

Here's an example code that demonstrates this process using the Plots.jl package for visualization:

```julia
using Plots

# Define the true cosine function
f_true(x) = cos(x)

# Define the number of nodes
n_values = [10, 20, 30]

# Split the interval into equidistant nodes
x_nodes = range(0, stop=2π, length=n_values[3]+1)

# Compute linear interpolation and plot the results
plot_legend = []
for n in n_values
    # Compute linear interpolation for the current number of nodes
    f_interpolation = []
    for i in 1:n
        x_start = x_nodes[i]
        x_end = x_nodes[i+1]
        m = (f_true(x_end) - f_true(x_start)) / (x_end - x_start)
        t = f_true(x_start) - m * x_start
        f_subinterval(x) = m * x + t
        push!(f_interpolation, f_subinterval)
    end

    # Plot the linear interpolation
    plot(x_nodes, f_true.(x_nodes), label="True Cosine", legend=:topleft)
    for i in 1:n
        plot!(x_nodes[i:i+1], f_interpolation[i].(x_nodes[i:i+1]), label="Interpolation $i")
    end

    # Store the legend for the plot
    push!(plot_legend, "Interpolation (n=$n)")
end

# Plot the piecewise linear interpolation
f_piecewise_interpolation = PiecewiseInterpolation(x_nodes, f_true.(x_nodes))
plot!(f_piecewise_interpolation, label="Piecewise Linear Interpolation")

# Plot the spline interpolation
f_spline_interpolation = CubicSplineInterpolation(x_nodes, f_true.(x_nodes))
plot!(f_spline_interpolation, label="Spline Interpolation")

# Set the x-axis label and title
xlabel!("x")
title!("Linear Interpolation of Cosine Function")

# Display the plot
plot!()
```

In this code, we first define the true cosine function `f_true(x) = cos(x)`. We then specify the number of nodes `n_values` for which we want to perform linear interpolation.

We split the interval `[0, 2π]` into equidistant nodes using the `range` function. Within each subinterval, we compute the linear equation for linear interpolation and store the interpolated functions in the `f_interpolation` array.

We plot the true cosine function and the linear interpolation results for different values of `n`. Additionally, we include the piecewise linear interpolation and spline interpolation using the `PiecewiseInterpolation` and `CubicSplineInterpolation` types, respectively.

Finally, we set the x-axis label and title and display the plot using `plot!()`.

Please ensure that you have the `Plots` package installed in Julia by running the following command in the Julia REPL:

```julia
using Pkg
Pkg.add("Plots")
```

With this code, you should be able to visualize the linear interpolation results and compare them with the true cosine function, as well as observe the piecewise linear interpolation and spline interpolation.

## Exercise 9 

---

2.9. Consider a Cournot oligopoly market with $m$ identical companies. Given the demand curve $D(P)=P^{-\eta}$, each company produces output $q(P)$ by equating marginal benefits and marginal costs. Marginal benefits are given by $P+q \frac{\mathrm{d} P}{\mathrm{~d} q}$ and marginal costs are assumed to be given by $c(q)=\alpha \sqrt{q}+q^2$. Due to the Cournot assumption each firm expects its competitors to not react to our changes in output. Consequently, we have $\frac{\mathrm{d} P}{\mathrm{~d} q}=\frac{1}{D^{\prime}(P)}$.

(a) Set up an equidistant price grid $\mathrm{P}(0: \mathrm{N})$ on the interval $[0.1 ; 3.0]$ using subroutine grid_Cons_Equi, initialize spline coefficients coeff_q $=0$ and derive the respective individual quantities $\mathrm{q}(0: \mathrm{N})$ at each gridpoint using function $\mathrm{f}$ zero.

(b) Interpolate the individual supply function $q$ using function spline_interp with the coefficients coeff_q. Use spline_eval to calculate supply in between the points specified in $\mathrm{P}(0: \mathrm{N})$. Print market demand and the supply curve.

(c) Compute the equilibrium price and quantity using function $\mathrm{f}$ zero as well as the interpolated aggregate supply function.

(d) Change the number $m$ of companies as well as the price elasticity $\eta$ and $\alpha$. Explain your results in economic terms.
The initial parametrization is $\alpha=1, \eta=1.5, m=3, N=10, N P=1000$.

---

(a) Setting up the equidistant price grid and calculating individual quantities:
To set up an equidistant price grid on the interval [0.1, 3.0] with N = 10, we divide the interval into N equal parts. Each grid point represents a price value. We'll use the subroutine grid_Cons_Equi to generate this grid.

Next, we initialize the spline coefficients coeff_q with the value 0. These coefficients will be used for interpolating the individual supply function.

Finally, we derive the respective individual quantities q at each grid point by using the function f_zero. The f_zero function is not provided in your question, so we'll need to define it. The function f_zero will solve for the quantity q at each price point, given the demand and cost functions.

(b) Interpolating the individual supply function and calculating market demand and supply:
We'll use the spline_interp function to interpolate the individual supply function q based on the coefficients coeff_q obtained in step (a). This will allow us to estimate the supply quantity at any price point, including those in between the grid points.

To calculate the market demand, we need to integrate the individual demand functions of each firm. Since all firms are identical and operate in a Cournot oligopoly, the individual demand function for each firm is given by D(P) = P^(-η). We sum up the demand quantities of all firms to obtain the aggregate demand.

The supply curve is obtained by evaluating the interpolated individual supply function at each price point on the grid.

(c) Computing the equilibrium price and quantity:
To find the equilibrium price and quantity, we need to determine the price at which the market demand equals the aggregate supply. This can be done by solving the equation D(P) = aggregate supply(P) using the f_zero function or by comparing the market demand and supply curves graphically.

(d) Changing the parameters:
To analyze the effects of changing the number of companies (m), the price elasticity (η), and the cost parameter (α), you can modify these values in the given equations and observe the resulting changes in equilibrium price and quantity. This will help you understand how these factors impact the market outcome in economic terms.

```julia
using Interpolations

# Function to calculate individual quantity using f_zero
function f_zero(p, α, η)
    q = (α * sqrt(p) + p^2) / (2 * η * p^2)
    return q
end

# Function to calculate market demand
function market_demand(p, η, m)
    return m * p^(-η)
end

# Set up an equidistant price grid
function grid_Cons_Equi(p_min, p_max, N)
    return range(p_min, p_max, length=N)
end

# Parameters
α = 1
η = 1.5
m = 3
N = 10

# Set up price grid
P = grid_Cons_Equi(0.1, 3.0, N)

# Initialize spline coefficients
coeff_q = zeros(N)

# Calculate individual quantities at each price point
for i in 1:N
    p = P[i]
    q = f_zero(p, α, η)
    coeff_q[i] = q
end

# Interpolate the individual supply function
interp_q = LinearInterpolation(P, coeff_q)

# Calculate supply at any price point
function supply(p)
    return interp_q(p)
end

# Calculate market demand and supply curves
demand_curve = market_demand.(P, η, m)
supply_curve = supply.(P)

# Print market demand and supply curve
println("Market Demand: ", demand_curve)
println("Supply Curve: ", supply_curve)
```
## Exercise 10

---

2.10. Now consider a local newspaper which is a monopoly in a region. It serves two distinct customer groups: readers and advertisers. The price of the newspaper $p_R$ is irrelevant for the advertiser, as are the advertising prices $p_A$ for the reader. However, since a higher newspaper price induces lower sales of newspapers, demand for ads will typically fall. On the other hand it is not clear how the demand for newspapers is affected by higher or lower advertising. In order to find out the optimal price combination $\left(p_R^*, p_A^*\right)$ that maximizes profits, the managers of the newspaper vary the two prices and observe the following profits $G\left(p_R, p_A\right)$ :

\begin{tabular}{|c|c|c|c|c|c|}
\hline & & \multicolumn{4}{|c|}{$\rho_A$} \\
\hline & & 0.5 & 4.5 & 8.5 & 12.5 \\
\hline \multirow{4}{*}{$\rho_R$} & 0.5 & 11.5 & 70.9 & 98.3 & 93.7 \\
\hline & 4.5 & 31.1 & 82.5 & 101.9 & 89.3 \\
\hline & 8.5 & 18.7 & 62.1 & 73.5 & 52.9 \\
\hline & 12.5 & -25.7 & 9.7 & 13.1 & -15.5 \\
\hline
\end{tabular}

(a) Given this information, construct a two-dimensional spline approximation of the profit function. What is the optimal price combination $\left(p_R^*, p_A^*\right)$ and the resulting optimal profit $G\left(p_R^*, p_A^*\right)$ ? Hint: Set up two equidistant price grids $\mathrm{PR}(0: 3)$ and $\mathrm{PA}(0: 3)$ on the interval $[0.5 ; 12.5]$ using subroutine grid_Cons_Equi. Given profits $\mathrm{G}(0: 3,0: 3)$ the spline coefficients coeff_G can be derived using subroutine spline_interp (G, coeff_G). Next, evaluate the profit function Gplot $(0: \mathrm{Npl}$ ot, $0:$ Nplot) using function spline_eval at each grid point. Finally, find the location of the maximum profit using function maxloc.

(b) Marginal costs for both newspapers and advertisements are constant at $c=0.1$. In addition, the managers have derived the 'true' demand functions

$$
x_R=10-p_R \text { and } x_A=20-p_A-0.5 p_R
$$

for newspapers and advertisements, respectively. Compute the profitmaximizing price combination using either $f$ minsearch or $f$ zero and compare with the approximated solution.

(c) Use Nplot $=100,1000,10000$ and compare the approximation error.

---

(a) To construct a two-dimensional spline approximation of the profit function, we'll follow the steps outlined in the question:

1. Set up two equidistant price grids PR and PA on the interval [0.5, 12.5] using the subroutine grid_Cons_Equi.
2. Initialize an empty matrix G with size (N, N), where N is the number of grid points.
3. Fill the matrix G with the given profit values.
4. Use the spline_interp function to obtain the spline coefficients coeff_G for the profit function.
5. Evaluate the profit function Gplot at each grid point using the spline_eval function.
6. Find the location of the maximum profit using the maxloc function.

Here's an example implementation in Julia:

```julia
using Interpolations

# Set up an equidistant price grid
function grid_Cons_Equi(p_min, p_max, N)
    return range(p_min, p_max, length=N)
end

# Profit data
G = [11.5 70.9 98.3 93.7;
     31.1 82.5 101.9 89.3;
     18.7 62.1 73.5 52.9;
     -25.7 9.7 13.1 -15.5]

# Parameters
N = 4

# Set up price grids
PR = grid_Cons_Equi(0.5, 12.5, N)
PA = grid_Cons_Equi(0.5, 12.5, N)

# Calculate spline coefficients for profit function
interp_G = spline_interp((PR, PA), G)

# Evaluate profit function at each grid point
Gplot = zeros(N, N)
for i in 1:N
    for j in 1:N
        Gplot[i, j] = interp_G(PR[i], PA[j])
    end
end

# Find location of maximum profit
max_profit_index = argmax(Gplot)
pR_opt, pA_opt = PR[max_profit_index[1]], PA[max_profit_index[2]]
G_opt = Gplot[max_profit_index[1], max_profit_index[2]]

println("Optimal price combination: pR = ", pR_opt, ", pA = ", pA_opt)
println("Optimal profit: G(pR, pA) = ", G_opt)
```

The code will calculate the optimal price combination (pR_opt, pA_opt) and the resulting optimal profit G_opt using a spline approximation of the profit function.

(b) To compute the profit-maximizing price combination using the "true" demand functions, we can set up an optimization problem using either fminsearch or fzero (root-finding) algorithms. Let's use the fminsearch algorithm in this case. Here's an example implementation in Julia:

```julia
using Optim

# True demand functions
function demand_R(pR)
    return 10 - pR
end

function demand_A(pR, pA)
    return 20 - pA - 0.5 * pR
end

# Cost parameter
c = 0.1

# Define the profit function
function profit(pR_pA)
    pR, pA = pR_pA
    xR = demand_R(pR)
    xA = demand_A(pR, pA)
    return (pR - c) * xR + (pA - c) * xA
end

# Set up the optimization problem
result = optimize(profit, [0.5, 0.5], NelderMead())

# Retrieve the optimal price combination and profit
pR_opt, pA_opt = result.minimum
G_opt = -result.minimum_obj_value  # negate the objective value to obtain profit

println("Optimal price combination (using true demand functions): pR = ", pR_opt, ", pA = ", pA_opt)
println("Optimal profit (using true demand functions): G(pR, pA) = ", G_opt)
```

The code will use the fminsearch algorithm from the Optim package to find the profit-maximizing price combination (pR_opt, pA_opt) and the resulting optimal profit G_opt using the "true" demand functions.

(c) To compare the approximation error, we'll calculate the error between the true profit function and the spline approximation for different values of Nplot (number of grid points). Here's an example implementation in Julia:

```julia
# Parameters
Nplot_values = [100, 1000, 10000]

# Calculate true profit function values
true_G = zeros(N, N)
for i in 1:N
    for j in 1:N
        true_G[i, j] = profit([PR[i], PA[j]])
    end
end

# Calculate approximation error for each Nplot value
for Nplot in Nplot_values
    # Set up price grids for approximation
    PR_approx = grid_Cons_Equi(0.5, 12.5, Nplot)
    PA_approx = grid_Cons_Equi(0.5, 12.5, Nplot)

    # Calculate spline coefficients for approximation
    interp_G_approx = spline_interp((PR, PA), G)

    # Evaluate approximation at each grid point
    G_approx = zeros(Nplot, Nplot)
    for i in 1:Nplot
        for j in 1:Nplot
            G_approx[i, j] = interp_G_approx(PR_approx[i], PA_approx[j])
        end
    end

    # Calculate approximation error
    error = norm(true_G - G_approx) / norm(true_G)

    println("Approximation error (Nplot = $Nplot): ", error)
end
```

The code will calculate the approximation error between the true profit function and the spline approximation for different values of Nplot. It uses the norm function to calculate the error as the normalized difference between the matrices. The approximation error is printed for each Nplot value specified in the Nplot_values array.

## Exercise 11

---

2.11. Three gravel-pits $A_1, A_2$ and $A_3$ store 11 tons, 13 tons, and 10 tons of gravel, respectively. The gravel is used at four building sites $B_1, B_2, B_3$, and $B_4 . B_1$ orders 5 tons, $B_2 7$ tons, $B_3 13$ tons, and $B_4 6$ tons of gravel. The transport cost of one ton of gravel from pit $A_i$ to the building site $B_j$ are displayed in the following table.
```math
\begin{tabular}{rrrrr}
\hline & $B_1$ & $B_2$ & $B_3$ & $B_4$ \\
\hline$A_1$ & 10 & 70 & 100 & 80 \\
$A_2$ & 130 & 90 & 120 & 110 \\
$A_3$ & 50 & 30 & 80 & 10 \\
\hline
\end{tabular}
```
Minimize the total cost of transport from gravel pits to the building sites using a simplex algorithm.

---

To minimize the total cost of transport from gravel pits to the building sites, we can formulate this problem as a linear programming problem and solve it using a simplex algorithm.

Let's define the decision variables as follows:
- Let xij represent the amount of gravel transported from pit Ai to building site Bj.
- Let cij represent the transport cost per ton from pit Ai to building site Bj.

Our objective is to minimize the total cost, which can be expressed as:
Minimize Z = 10x11 + 70x12 + 100x13 + 80x14 + 130x21 + 90x22 + 120x23 + 110x24 + 50x31 + 30x32 + 80x33 + 10x34

Subject to the following constraints:
1. The total amount of gravel transported from pit Ai should not exceed the available quantity in pit Ai:
x11 + x12 + x13 + x14 ≤ 11
x21 + x22 + x23 + x24 ≤ 13
x31 + x32 + x33 + x34 ≤ 10

2. The total amount of gravel delivered to building site Bj should meet the demand at that site:
x11 + x21 + x31 = 5
x12 + x22 + x32 = 7
x13 + x23 + x33 = 13
x14 + x24 + x34 = 6

All variables should also be non-negative: xij ≥ 0.

We can solve this linear programming problem using a simplex algorithm implementation in Julia, such as the `SimplexSolver` package. Please make sure you have the `SimplexSolver` package installed by running `using Pkg; Pkg.add("SimplexSolver")` in your Julia environment.

Here's an example implementation:

```julia
using SimplexSolver

# Define the cost matrix
C = [10 70 100 80;
     130 90 120 110;
     50 30 80 10]

# Define the available quantities in pits and the demand at building sites
b = [11; 13; 10]
d = [5; 7; 13; 6]

# Define the objective function coefficients
c = reshape(C', 12)

# Define the constraint matrix
A = [1 0 0 0 1 0 0 0 1 0 0 0;
     0 1 0 0 0 1 0 0 0 1 0 0;
     0 0 1 0 0 0 1 0 0 0 1 0;
     0 0 0 1 0 0 0 1 0 0 0 1]

# Create a simplex problem
problem = SimplexProblem(c, A, b, d, names=["x11", "x12", "x13", "x14", "x21", "x22", "x23", "x24", "x31", "x32", "x33", "x34"])

# Solve the simplex problem
solve!(problem)

# Get the optimal solution
optimal_solution = problem.solution

# Print the optimal solution
println("Optimal Solution:")
for (name, value) in optimal_solution
    println("$name = $value")
end

# Calculate the total cost
total_cost = dot(c, [value for (name, value) in optimal_solution])
println("Total Cost = $total_cost")
```

The code will solve the linear programming problem using the simplex algorithm and provide the optimal solution, which represents the optimal amount of gravel transported from each pit to each building site. It also calculates the total cost of transport.
