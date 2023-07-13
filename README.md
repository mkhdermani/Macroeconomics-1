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
