# Linear Algebra with Python

This repository contains Jupyter notebooks and Python scripts for exploring various topics in linear algebra using Python. It includes visualization, computation, and practical examples of concepts such as linear systems, matrix operations, determinants, and factorizations. The code utilizes popular Python libraries including Matplotlib, NumPy, and SymPy for data analysis and mathematical computations.

## Table of Contents

1. [Chapter 1 - Linear Equation System](#chapter-1---linear-equation-system)
    - [Visualization and Solving Linear Systems in Python](#visualization-and-solving-linear-systems-in-python)
    - [Requirements](#requirements)
    - [Overview](#overview)
      - [Visualization of a System of Two Linear Equations](#1-visualization-of-a-system-of-two-linear-equations)
      - [Drawing a Plane](#2-drawing-a-plane)
      - [Visualization of a System of Three Linear Equations](#3-visualization-of-a-system-of-three-linear-equations)
      - [Visualization of a System with Infinite Solutions](#4-visualization-of-a-system-with-infinite-solutions)
      - [Reduced Row Echelon Form](#5-reduced-row-echelon-form)
      - [Example: Symbolic Solution and Polynomial Fitting](#6-example-symbolic-solution-and-polynomial-fitting)
      - [Solving Linear Systems Using NumPy](#7-solving-linear-systems-using-numpy)

2. [Chapter 2 - Basic Matrix Algebra](#chapter-2---basic-matrix-algebra)
    - [Matrix Operations and SymPy Demonstrations](#matrix-operations-and-sympy-demonstrations)
    - [Dependencies](#dependencies)
    - [Matrix Operations](#matrix-operations)
      - [Matrix Addition](#matrix-addition)
      - [Matrix Multiplication](#matrix-multiplication)
      - [Commutability](#commutability)
    - [SymPy Demonstrations](#sympy-demonstrations)
      - [Addition](#addition)
      - [Multiplication](#multiplication)
      - [Transpose](#transpose)
      - [Identity Matrices](#identity-matrices)
      - [Elementary Matrices](#elementary-matrices)
      - [Inverse Matrices](#inverse-matrices)
      - [Gauss-Jordan Elimination](#gauss-jordan-elimination)

3. [Chapter 3 - Determinant](#chapter-3---determinant)
    - [Determinants Visualization and Computation](#determinants-visualization-and-computation)
    - [Overview](#overview-1)
    - [Installation](#installation)
    - [Usage](#usage)
      - [Visualization of Determinants](#visualization-of-determinants)
      - [Determinant Computation](#determinant-computation)
      - [Cramer's Rule](#cramers-rule)
    - [Explanation](#explanation)

4. [Chapter 4 - LU Factorization](#chapter-4---lu-factorization)
    - [Overview](#overview-2)
    - [LU Factorization](#lu-factorization)
      - [Key Points](#key-points)
      - [Example](#example)
    - [Cholesky Factorization](#cholesky-factorization)
      - [Key Points](#key-points-1)
      - [Example](#example-1)
    - [Usage](#usage-1)
    - [Dependencies](#dependencies-1)

5. [Chapter 5 - Vector Addition, Subtraction and Scalar Multiplication](#chapter-5---vector-addition-subtraction-and-scalar-multiplication)
    - [Vector Operations and Visualization](#vector-operations-and-visualization)
    - [Overview](#overview-3)
    - [Features](#features)
    - [Usage](#usage-2)
      - [Vector Norm Calculation](#vector-norm-calculation)
      - [2D Vector Operations](#2d-vector-operations)
      - [3D Vector Operations](#3d-vector-operations)
    - [Examples](#examples)
    - [Visualization](#visualization-1)
---


# Chapter 1 - Linear Equation System



## Visualization and Solving Linear Systems in Python

This file contains code for visualizing and solving systems of linear equations using Python libraries such as Matplotlib, NumPy, and SymPy. The code covers basic plotting of linear systems, 3D plane visualization, solving systems of equations, and polynomial fitting.

## Table of Contents

1. [Requirements](#requirements)
2. [Overview](#overview)
    - [Visualization of a System of Two Linear Equations](#1-visualization-of-a-system-of-two-linear-equations)
    - [Drawing a Plane](#2-drawing-a-plane)
    - [Visualization of a System of Three Linear Equations](#3-visualization-of-a-system-of-three-linear-equations)
    - [Visualization of a System with Infinite Solutions](#4-visualization-of-a-system-with-infinite-solutions)
    - [Reduced Row Echelon Form](#5-reduced-row-echelon-form)
    - [Example: Symbolic Solution and Polynomial Fitting](#6-example-symbolic-solution-and-polynomial-fitting)
    - [Solving Linear Systems Using NumPy](#7-solving-linear-systems-using-numpy)



## Requirements

Ensure you have the following Python packages installed:
- `matplotlib`
- `numpy`
- `sympy`

You can install them using pip:
```bash
pip install matplotlib numpy sympy
```

## Overview

### 1. Visualization of a System of Two Linear Equations

This section plots a system of two linear equations:
- `x + y = 6`
- `x - y = -4`

The code uses Matplotlib to plot these equations and visualize their intersection. The intersection point is highlighted on the graph.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y1 = -x + 6
y2 = x + 4

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(x, y1, lw=3, label='$x+y=6$')
ax.plot(x, y2, lw=3, label='$x-y=-4$')
ax.scatter(1, 5, s=200, zorder=5, color='r', alpha=0.8)
ax.text(1, 5.5, '$(1,5)$', fontsize=20)
ax.set_title('Solution of $x+y=6$, $x-y=-4$', size=22)
ax.grid()
ax.legend()
plt.show()
```

### 2. Drawing a Plane

This section demonstrates how to create meshgrids and plot a simple plane in 3D space using Matplotlib.

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x, y = np.arange(-3, 4, 1), np.arange(-3, 4, 1)
X, Y = np.meshgrid(x, y)
Z = X + Y

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.set_title('$z=x+y$', size=18)
plt.show()
```

### 3. Visualization of a System of Three Linear Equations

This section visualizes the solution of a system of three linear equations using Matplotlib's 3D plotting capabilities.

```python
x1 = np.linspace(25, 35, 20)
x2 = np.linspace(10, 20, 20)
X1, X2 = np.meshgrid(x1, x2)

X3 = 2*X2 - X1
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, X3, cmap='viridis', alpha=1)

X3 = .25*X2 - 1
ax.plot_surface(X1, X2, X3, cmap='summer', alpha=1)

X3 = -5/9*X2 + 4/9*X1 - 1
ax.plot_surface(X1, X2, X3, cmap='spring', alpha=1)

ax.scatter(29, 16, 3, s=200, color='black')
plt.show()
```

### 4. Visualization of a System with Infinite Solutions

This section demonstrates how to visualize a system of linear equations that has an infinite number of solutions.

```python
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.mgrid[-2:2:21j, 2:6:21j]
Z1 = Y - 4
Z2 = 2 - X - Y/2
Z3 = 8 - 2*X - 2*Y

ax.plot_surface(X, Y, Z1, cmap='spring', alpha=0.5)
ax.plot_surface(X, Y, Z2, cmap='summer', alpha=0.5)
ax.plot_surface(X, Y, Z3, cmap='autumn', alpha=0.5)

ZL = np.linspace(-2, 2, 20)
XL = -3 * ZL / 2
YL = ZL + 4
ax.plot(XL, YL, ZL, color='black', linewidth=2, label='Infinite Solutions')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.title('A System of Linear Equations With Infinite Number of Solutions')
ax.legend()
plt.show()
```

### 5. Reduced Row Echelon Form

This section uses SymPy to perform row reduction on an augmented matrix to solve a linear system.

```python
import sympy as sy

M = sy.Matrix([[5, 0, 11, 3], [7, 23, -3, 7], [12, 11, 3, -4]])
M_rref = M.rref()
print(M_rref[0].astype(float))
```

### 6. Example: Symbolic Solution and Polynomial Fitting

The following example shows how to find a cubic polynomial that fits given points using SymPy, and how to solve a system symbolically.

```python
# Symbolic solution
a, b, c = sy.symbols('a b c')
A = sy.Matrix([[1, 2, -3, a], [4, -1, 8, b], [2, -6, -4, c]])
A_rref = A.rref()
print(A_rref)

# Polynomial fitting
x = np.linspace(-5, 5, 40)
y = poly_coef[0] + poly_coef[1]*x + poly_coef[2]*x**2 + poly_coef[3]*x**3
```

### 7. Solving Linear Systems Using NumPy

This section demonstrates how to solve a system of linear equations using NumPy.

```python
import numpy as np

A = np.round(10 * np.random.rand(5, 5))
b = np.round(10 * np.random.rand(5,))
x = np.linalg.solve(A, b)
print(x)
```

# Chapter 2 - Basic Matrix Algebra

## Matrix Operations and SymPy Demonstrations

This file contains Python code for demonstrating various matrix operations using NumPy and SymPy. It includes operations such as addition, multiplication, transpose, and inversion of matrices, as well as the use of elementary matrices and Gauss-Jordan elimination.

## Table of Contents
1. [Dependencies](#dependencies)
2. [Matrix Operations](#matrix-operations)
    - [Matrix Addition](#matrix-addition)
    - [Matrix Multiplication](#matrix-multiplication)
    - [Commutability](#commutability)
3. [SymPy Demonstrations](#sympy-demonstrations)
    - [Addition](#addition)
    - [Multiplication](#multiplication)
    - [Transpose](#transpose)
    - [Identity Matrices](#identity-matrices)
    - [Elementary Matrices](#elementary-matrices)
    - [Inverse Matrices](#inverse-matrices)
    - [Gauss-Jordan Elimination](#gauss-jordan-elimination)


## Dependencies

The code requires the following Python libraries:
- `numpy`
- `sympy`
- `IPython`

You can install the necessary libraries using pip:

```bash
pip install numpy sympy ipython
```

## Matrix Operations

### Matrix Addition

Matrix addition is straightforward. For any matrices \(A\) and \(B\) of the same dimensions, their sum is computed element-wise.

Example:
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B
print(C)
```

### Matrix Multiplication

Matrix multiplication can be performed in two ways:
- **Hadamard Multiplication** (element-wise)
- **Matrix Product**

Example:
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

hadamard_product = A * B
matrix_product = A @ B

print("Hadamard Product:\n", hadamard_product)
print("Matrix Product:\n", matrix_product)
```

### Commutability

Matrix multiplication is generally not commutative. For example:

```python
import sympy as sy

A = sy.Matrix([[3, 4], [7, 8]])
B = sy.Matrix([[5, 3], [2, 1]])

AB = A * B
BA = B * A

print("AB:\n", AB)
print("BA:\n", BA)
```

## SymPy Demonstrations

### Addition

Symbolic matrix addition using SymPy:

```python
import sympy as sy

a, b, c, d, e, f, g, h, i, j, k, l = sy.symbols('a b c d e f g h i j k l')
A = sy.Matrix([[a, b, c], [d, e, f]])
B = sy.Matrix([[g, h, i], [j, k, l]])

sum_matrix = A + B
print("Sum:\n", sum_matrix)
```

### Multiplication

Symbolic matrix multiplication:

```python
import sympy as sy

A = sy.Matrix([[a, b, c], [d, e, f]])
B = sy.Matrix([[g, h, i], [j, k, l], [m, n, o]])

product = A * B
print("Product:\n", product)
```

### Transpose

Transpose of a matrix:

```python
import sympy as sy

A = sy.Matrix([[1, 2, 3], [4, 5, 6]])
A_transpose = A.transpose()
print("Transpose:\n", A_transpose)
```

### Identity Matrices

Creating and verifying identity matrices:

```python
import numpy as np

I = np.eye(5)
print("Identity Matrix:\n", I)
```

### Elementary Matrices

Demonstrating elementary matrices:

```python
import sympy as sy

E = sy.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
A = sy.randMatrix(3, percent=80)
EA = E * A
print("Elementary Matrix Transformation:\n", EA)
```

### Inverse Matrices

Finding and verifying the inverse of a matrix:

```python
import numpy as np

A = np.round(10 * np.random.randn(5, 5))
Ainv = np.linalg.inv(A)
print("Inverse Matrix:\n", Ainv)
print("Verification (A @ Ainv):\n", A @ Ainv)
```

### Gauss-Jordan Elimination

Performing Gauss-Jordan elimination to find the inverse:

```python
import sympy as sy

A = sy.Matrix([[3, 4], [7, 8]])
I = sy.eye(2)
AI = A.row_join(I)
AI_rref = AI.rref()
Ainv = AI_rref[0][:, 2:]
print("Gauss-Jordan Elimination Result:\n", Ainv)
```

# Chapter 3 - Determinant

## Determinants Visualization and Computation

This project demonstrates the visualization and computation of determinants using both `NumPy` and `SymPy` libraries. It includes code for plotting determinants in 2D and 3D, calculating determinants using cofactor expansion, and solving linear systems using Cramer's Rule.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Visualization of Determinants](#visualization-of-determinants)
    - [Determinant Computation](#determinant-computation)
    - [Cramer's Rule](#cramers-rule)
4. [Explanation](#explanation)

## Overview

1. **Visualization of Determinants**: 
   - Plotting 2D matrices and their determinants.
   - Visualization of 3D determinants (parallelepipeds).

2. **Determinant Computation**:
   - Using SymPy for symbolic calculations and cofactor expansion.
   - Verification of results with numerical methods.

3. **Cramer's Rule**:
   - Solving linear systems using determinants.
   - Comparison with NumPy's built-in methods.

## Installation

Ensure you have `numpy`, `sympy`, and `matplotlib` installed. You can install these using pip:

```bash
pip install numpy sympy matplotlib
```

## Usage

### Visualization of Determinants

This script plots vectors forming a parallelogram (2D) or parallelepiped (3D) and calculates the determinant.

**Example Code:**

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matrix_2d = np.array([[2, 0], [0, 3]])
plot_2ddet(matrix_2d)

matrix_2d_non_diag = np.array([[2, 3], [1, 3]])
plot_2ddet(matrix_2d_non_diag)

matrix_2d_linearly_dependent = np.array([[2, 3], [4, 6]])
plot_2ddet(matrix_2d_linearly_dependent)

# 3D plot example
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
# Define and plot vertices and edges here
```

### Determinant Computation

Calculate determinants using SymPy's cofactor expansion and compare with NumPy.

**Example Code:**

```python
import sympy as sy

a, b, c, d = sy.symbols('a b c d')
matrix = sy.Matrix([[a, b], [c, d]])
det = matrix.det()
print(f"Determinant: {det}")

# Cofactor expansion
def cof_exp(matrix, c):
    detA = 0
    for i in range(matrix.shape[0]):
        minor_matrix = matrix.minor_submatrix(i, c)
        cofactor = (-1)**(i + c) * minor_matrix.det()
        detA += matrix[i, c] * cofactor
    return detA

matrix = sy.Matrix([[49, 0, 61], [73, 22, 96], [2, 0, 32]])
cof_exp(matrix, 1)
```

### Cramer's Rule

Solve linear systems using determinants.

**Example Code:**

```python
import numpy as np

A = np.array([[2, -1, 3], [3, 3, -1], [-1, -1, 1]])
b = np.array([-3, 10, -4])

det_A = np.linalg.det(A)
A_1b = np.copy(A)
A_1b[:, 0] = b
x1 = np.linalg.det(A_1b) / det_A

A_2b = np.copy(A)
A_2b[:, 1] = b
x2 = np.linalg.det(A_2b) / det_A

A_3b = np.copy(A)
A_3b[:, 2] = b
x3 = np.linalg.det(A_3b) / det_A

print(f"Solutions: x1 = {x1}, x2 = {x2}, x3 = {x3}")
```

## Explanation

- **2D Determinants**: The code uses `matplotlib` to visualize the area (or parallelogram) formed by vectors corresponding to a 2x2 matrix. Different matrices are plotted to show varying areas and orientations.

- **3D Determinants**: Plots a 3D parallelepiped to represent the volume corresponding to a 3x3 matrix.

- **Cofactor Expansion**: Calculates determinants using symbolic cofactor expansion, showcasing its application for matrices with symbolic elements.

- **Cramer's Rule**: Demonstrates solving a system of linear equations by calculating determinants of modified matrices.


# Chapter 4 - LU Factorization


### Overview

This file provides a detailed explanation and implementation of LU Factorization and Cholesky Factorization, including practical examples and numerical results using Python libraries such as NumPy and SciPy. 

## Table of Contents

1. [LU Factorization](#lu-factorization)
   - [Key Points](#key-points)
   - [Example](#example)
2. [Cholesky Factorization](#cholesky-factorization)
   - [Key Points](#key-points-1)
   - [Example](#example-1)
3. [Usage](#usage)
4. [Dependencies](#dependencies)



  
### LU Factorization

LU Factorization decomposes a matrix \( A \) into the product of a lower triangular matrix \( L \) and an upper triangular matrix \( U \). This method is computationally efficient for solving systems of linear equations, especially when multiple solutions are needed.

#### Key Points:
- LU Factorization is advantageous for solving multiple systems with the same coefficient matrix.
- SciPy's `sp.linalg.lu()` function provides the LU decomposition along with a permutation matrix \( P \), necessary for some matrices to perform the factorization.

#### Example:
```python
import numpy as np
import scipy as sp

A = np.array([[9, 3, 6], [3, 4, 6], [0, 8, 8]])
P, L, U = sp.linalg.lu(A)
print(P)
print(L)
print(U)
```

### Cholesky Factorization

Cholesky Factorization decomposes a positive-definite matrix into the product of a lower triangular matrix and its transpose. This is commonly used for covariance matrices in statistics and in simulations to introduce correlations.

#### Key Points:
- Cholesky Factorization is analogous to taking the square root of a matrix.
- It is particularly useful for transforming non-correlated random variables into correlated ones.

#### Example:
```python
import numpy as np

n = 1000
z = np.random.normal(size=(1000, 3))
z2 = -z
z = np.concatenate((z, z2), axis=0)

cov_matrix = np.array([[1, 0.8, 0.7], [0.8, 1, 0.8], [0.7, 0.8, 1]])
L = np.linalg.cholesky(cov_matrix)

z = z @ L.T

print(np.cov(z.T))
```

### Usage

1. **LU Factorization**: Decompose matrix \( A \) to solve linear systems efficiently.
2. **Cholesky Factorization**: Transform random variables or analyze covariance matrices.

### Dependencies

- NumPy
- SciPy
- Matplotlib (for plotting, not used in this example)
- Sympy (for symbolic mathematics, not used in this example)





# Chapter 5 - Vector Addition, Subtraction and Scalar Multiplication

## Vector Operations and Visualization

## Overview

This Python script demonstrates various vector operations and visualizations using the `numpy` and `matplotlib` libraries. It includes examples of vector norms, addition, subtraction, and scalar multiplication, as well as 2D and 3D vector visualizations.

## Table of Contents


1. [Features](#features)
2. [Usage](#usage)
   - [Vector Norm Calculation](#vector-norm-calculation)
   - [2D Vector Operations](#2d-vector-operations)
   - [3D Vector Operations](#3d-vector-operations)
3. [Examples](#examples)
4. [Visualization](#visualization)



## Features

- **Vector Norms**: Calculates and displays the norms (magnitudes) of 2D and 3D vectors using `numpy.linalg.norm()`.
- **2D Vector Visualization**: Plots 2D vectors, including vector addition, subtraction, and scalar multiplication, with annotations showing their coordinates and magnitudes.
- **3D Vector Visualization**: Plots 3D vectors to illustrate operations like vector addition, subtraction, and scalar multiplication, using `matplotlib`'s 3D plotting capabilities.



## Usage

### Vector Norm Calculation

- Computes the norms of vectors in both 2D and 3D spaces.
- Example vectors: `[4, 7]`, `[8, 6]` for 2D; `[1, 1, 1]`, `[-2, 2, 5]`, `[3, -2, 1]` for 3D.

### 2D Vector Operations

- Visualizes vectors in 2D space.
- Demonstrates vector addition, subtraction, and scalar multiplication with graphical annotations.

### 3D Vector Operations

- Visualizes vectors in 3D space.
- Demonstrates vector addition, subtraction, and scalar multiplication in 3D with graphical annotations.

## Examples

- **2D Vector Norms and Addition**:
  Shows vectors `[4, 7]` and `[8, 6]` with their norms and graphical representation of their addition.

- **2D Vector Subtraction and Scalar Multiplication**:
  Illustrates how vectors are subtracted and scaled in 2D space.

- **3D Vector Addition and Scalar Multiplication**:
  Visualizes vector operations in 3D, demonstrating how vectors combine and scale.

## Visualization

The script produces plots with:
- 2D vectors displayed as arrows.
- Annotations for vector endpoints and magnitudes.
- 3D vectors displayed in a 3D coordinate system with arrows and annotations.


# Chapter 6 - Linear Combination

## Overview

In this chapter, we delve into the concept of linear combinations and their visual representations in 2D and 3D spaces. We explore how vectors can span a space and visualize these concepts using Python libraries.

## Table of Contents

1. [2D Linear Combination](#2d-linear-combination)
   - [Visualization](#visualization)
   - [Basis Coordinates](#basis-coordinates)
2. [3D Linear Combination](#3d-linear-combination)
   - [Visualization Function](#visualization-function)
3. [Inconsistent Systems](#inconsistent-systems)
   - [One Free Variable](#one-free-variable)
   - [Two Free Variables](#two-free-variables)
4. [Polynomial Combinations](#polynomial-combinations)

## 2D Linear Combination

### Visualization

This section covers the visualization of linear combinations of two vectors in 2D. We plot the vectors and their linear combinations using `matplotlib`:

```python
import matplotlib.pyplot as plt
import numpy as np

# Define vectors
vectors = np.array([[[0, 0, 4, 2]], [[0, 0, -2, 2]], [[0, 0, 2, 10]], [[0, 0, 8, 4]], [[0, 0, -6, 6]]])
colors = ['b', 'b', 'r', 'b', 'b']

# Plot vectors
fig, ax = plt.subplots()
for i in range(vectors.shape[0]):
    X, Y, U, V = zip(*vectors[i, :, :])
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', color=colors[i])
```

### Basis Coordinates

Here, we plot the grid representing the basis vectors and their linear combinations in 2D. The pink grid lines illustrate the span of the basis vectors:

```python
a = np.arange(-11, 20, 1)
x = np.arange(-11, 20, 1)

for i in a:
    y1 = i + 0.5 * x
    ax.plot(x, y1, ls='--', color='pink')
    y2 = i - x
    ax.plot(x, y2, ls='--', color='pink')
```

## 3D Linear Combination

### Visualization Function

This function visualizes linear combinations in 3D space. It illustrates how scalar multipliers affect the standard basis vectors:

```python
def linearCombo(a, b, c):
    '''Visualize linear combination in 3D.'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vec = np.array([[[0, 0, 0, a, 0, 0]], [[0, 0, 0, 0, b, 0]], [[0, 0, 0, 0, 0, c]], [[0, 0, 0, a, b, c]]])
    colors = ['b', 'r', 'g']
    for i in range(vec.shape[0]):
        X, Y, Z, U, V, W = zip(*vec[i,:,:])
        ax.quiver(X, Y, Z, U, V, W, color=colors[i])
```

## Inconsistent Systems

### One Free Variable

We explore how an inconsistent system with one free variable can be represented as a line in 3D space:

```python
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
t = np.linspace(-1, 1, 10)
c1 = 3/2 + t/2
c2 = -1/2 - 5/2 * t
ax.plot(c1, c2, t)
```

### Two Free Variables

In this section, we visualize an inconsistent system with two free variables. The resulting solution spans a plane in 3D space:

```python
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x2 = np.linspace(-2, 2, 10)
x3 = np.linspace(-2, 2, 10)
X2, X3 = np.meshgrid(x2, x3)
X1 = 3 * X2 + 2 * X3
ax.plot_wireframe(X1, X2, X3)
```

## Polynomial Combinations

This section demonstrates how polynomials can be expressed as linear combinations of other polynomials. We use `sympy` to solve the corresponding system:

```python
import sympy as sy
sy.init_printing()

A = sy.Matrix([[1,2,1,4],[2,1,-1,5],[-1,-1,-1,-2],[1,1,-4,7]])
A.rref()
```
