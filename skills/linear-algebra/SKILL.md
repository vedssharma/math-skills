---
name: linear-algebra
description: Expert linear algebra skill for vector and matrix operations. Use when performing calculations involving vectors (dot product, cross product, magnitude, normalization, projection, angle between vectors), matrices (multiplication, transpose, inverse, determinant, trace, rank, eigenvalues/eigenvectors), decompositions (LU, QR, SVD, Cholesky), solving linear systems, or any numerical linear algebra task.
---

# Linear Algebra

Expert system for vector and matrix computations using NumPy/SciPy.

## Quick Reference

### Vector Operations

```python
import numpy as np

# Creation
v = np.array([1, 2, 3])

# Magnitude (L2 norm)
np.linalg.norm(v)

# Normalize
v / np.linalg.norm(v)

# Dot product
np.dot(v1, v2)  # or v1 @ v2

# Cross product (3D only)
np.cross(v1, v2)

# Angle between vectors (radians)
np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Projection of v1 onto v2
(np.dot(v1, v2) / np.dot(v2, v2)) * v2
```

### Matrix Operations

```python
# Creation
A = np.array([[1, 2], [3, 4]])

# Transpose
A.T

# Matrix multiplication
A @ B  # or np.matmul(A, B)

# Element-wise multiplication
A * B

# Determinant
np.linalg.det(A)

# Inverse
np.linalg.inv(A)

# Trace
np.trace(A)

# Rank
np.linalg.matrix_rank(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# For symmetric matrices (more stable)
eigenvalues, eigenvectors = np.linalg.eigh(A)
```

### Decompositions

```python
from scipy import linalg

# LU decomposition: A = P @ L @ U
P, L, U = linalg.lu(A)

# QR decomposition: A = Q @ R
Q, R = np.linalg.qr(A)

# SVD: A = U @ S @ Vt
U, s, Vt = np.linalg.svd(A)
S = np.diag(s)  # Convert singular values to diagonal matrix

# Cholesky (for positive definite): A = L @ L.T
L = np.linalg.cholesky(A)
```

### Solving Linear Systems

```python
# Solve Ax = b
x = np.linalg.solve(A, b)

# Least squares (for overdetermined systems)
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

# Pseudo-inverse solution
x = np.linalg.pinv(A) @ b
```

## Script Usage

Run `scripts/linalg.py` for interactive computations:

```bash
python scripts/linalg.py <operation> [args...]
```

Operations:
- `dot v1 v2` - Dot product
- `cross v1 v2` - Cross product
- `norm v` - Vector magnitude
- `normalize v` - Unit vector
- `angle v1 v2` - Angle between vectors (degrees)
- `proj v1 v2` - Project v1 onto v2
- `matmul A B` - Matrix multiplication
- `det A` - Determinant
- `inv A` - Inverse
- `trans A` - Transpose
- `trace A` - Trace
- `rank A` - Rank
- `eig A` - Eigenvalues/eigenvectors
- `lu A` - LU decomposition
- `qr A` - QR decomposition
- `svd A` - Singular value decomposition
- `chol A` - Cholesky decomposition
- `solve A b` - Solve linear system

Vectors: `[1,2,3]`
Matrices: `[[1,2],[3,4]]`

## Advanced Topics

See `references/formulas.md` for:
- Condition number and numerical stability
- Gram-Schmidt orthogonalization
- Matrix norms (Frobenius, spectral, etc.)
- Sparse matrix operations
- Complex matrices and Hermitian operations
