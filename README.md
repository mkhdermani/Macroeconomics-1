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
