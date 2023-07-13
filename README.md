# Macroeconomics-1
Macro of TEIAS
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable columnalign="left left" columnspacing="1em" rowspacing="4pt">
    <mtr>
      <mtd>
        <mrow>
          <maligngroup/>
          <malignmark/>
          <mrow>
            <msubsup>
              <mrow>
                <mi>q</mi>
              </mrow>
              <mn>1</mn>
              <mi>s</mi>
            </msubsup>
            <mo>=</mo>
            <mo>−</mo>
            <mn>10</mn>
            <mo>+</mo>
            <msub>
              <mi>p</mi>
              <mn>1</mn>
            </msub>
            <mo>    </mo>
          </mrow>
          <maligngroup/>
          <malignmark/>
          <mrow>
            <msubsup>
              <mrow>
                <mi>q</mi>
              </mrow>
              <mn>1</mn>
              <mi>d</mi>
            </msubsup>
            <mo>=</mo>
            <mn>20</mn>
            <mo>−</mo>
            <msub>
              <mi>p</mi>
              <mn>1</mn>
            </msub>
            <mo>−</mo>
            <msub>
              <mi>p</mi>
              <mn>3</mn>
            </msub>
          </mrow>
        </mrow>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mrow>
          <maligngroup/>
          <malignmark/>
          <mrow>
            <msubsup>
              <mrow>
                <mi>q</mi>
              </mrow>
              <mn>2</mn>
              <mi>s</mi>
            </msubsup>
            <mo>=</mo>
            <mn>2</mn>
            <msub>
              <mi>p</mi>
              <mn>2</mn>
            </msub>
            <mo>    </mo>
          </mrow>
          <maligngroup/>
          <malignmark/>
          <mrow>
            <msubsup>
              <mrow>
                <mi>q</mi>
              </mrow>
              <mn>2</mn>
              <mi>d</mi>
            </msubsup>
            <mo>=</mo>
            <mn>40</mn>
            <mo>−</mo>
            <mn>2</mn>
            <msub>
              <mi>p</mi>
              <mn>2</mn>
            </msub>
            <mo>−</mo>
            <msub>
              <mi>p</mi>
              <mn>3</mn>
            </msub>
          </mrow>
        </mrow>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mrow>
          <maligngroup/>
          <malignmark/>
          <mrow>
            <msubsup>
              <mrow>
                <mi>q</mi>
              </mrow>
              <mn>3</mn>
              <mi>s</mi>
            </msubsup>
            <mo>=</mo>
            <mo>−</mo>
            <mn>5</mn>
            <mo>+</mo>
            <msub>
              <mi>p</mi>
              <mn>3</mn>
            </msub>
            <mo>    </mo>
          </mrow>
          <maligngroup/>
          <malignmark/>
          <mrow>
            <msubsup>
              <mrow>
                <mi>q</mi>
              </mrow>
              <mn>3</mn>
              <mi>d</mi>
            </msubsup>
            <mo>=</mo>
            <mn>25</mn>
            <mo>−</mo>
            <msub>
              <mi>p</mi>
              <mn>1</mn>
            </msub>
            <mo>−</mo>
            <msub>
              <mi>p</mi>
              <mn>2</mn>
            </msub>
            <mo>−</mo>
            <msub>
              <mi>p</mi>
              <mn>3</mn>
            </msub>
          </mrow>
        </mrow>
      </mtd>
    </mtr>
  </mtable>
</math>

How to solve it in julia
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
