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
