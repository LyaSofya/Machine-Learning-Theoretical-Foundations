# Chapter 1 - Linear Equation System.ipynb



## Visualization and Solving Linear Systems in Python

This file contains code for visualizing and solving systems of linear equations using Python libraries such as Matplotlib, NumPy, and SymPy. The code covers basic plotting of linear systems, 3D plane visualization, solving systems of equations, and polynomial fitting.

## Requirements

Ensure you have the following Python packages installed:
- `matplotlib`
- `numpy`
- `sympy`

You can install them using pip:
```bash
pip install matplotlib numpy sympy
```

## Code Overview

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

# Chapter 2 - Basic Matrix Algebra.ipynb

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

