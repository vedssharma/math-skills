# Linear Algebra Reference

## Table of Contents
- [Matrix Norms](#matrix-norms)
- [Condition Number](#condition-number)
- [Gram-Schmidt Orthogonalization](#gram-schmidt-orthogonalization)
- [Sparse Matrices](#sparse-matrices)
- [Complex and Hermitian Matrices](#complex-and-hermitian-matrices)
- [Special Matrices](#special-matrices)
- [Numerical Stability Tips](#numerical-stability-tips)

---

## Matrix Norms

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])

# Frobenius norm: sqrt(sum of squared elements)
np.linalg.norm(A, 'fro')

# Spectral norm (2-norm): largest singular value
np.linalg.norm(A, 2)

# 1-norm: max column sum
np.linalg.norm(A, 1)

# Infinity norm: max row sum
np.linalg.norm(A, np.inf)

# Nuclear norm: sum of singular values
np.linalg.norm(A, 'nuc')
```

## Condition Number

Measures sensitivity to numerical errors. High condition number = ill-conditioned.

```python
# Condition number (ratio of largest to smallest singular value)
np.linalg.cond(A)

# Rule of thumb: lose log10(cond) digits of precision
# cond > 1e10 is typically problematic
```

**Improving conditioning:**
- Scale rows/columns to similar magnitudes
- Use pivoting in decompositions
- Consider regularization for near-singular systems

## Gram-Schmidt Orthogonalization

Convert linearly independent vectors to orthonormal basis:

```python
def gram_schmidt(V):
    """Orthonormalize columns of V."""
    n = V.shape[1]
    U = np.zeros_like(V, dtype=float)
    for i in range(n):
        u = V[:, i].astype(float)
        for j in range(i):
            u -= np.dot(U[:, j], V[:, i]) * U[:, j]
        U[:, i] = u / np.linalg.norm(u)
    return U

# Or use QR decomposition (more stable)
Q, R = np.linalg.qr(V)
# Q contains orthonormal columns
```

## Sparse Matrices

For matrices with mostly zero elements:

```python
from scipy import sparse

# Create sparse matrix (CSR format - efficient for row operations)
data = [1, 2, 3]
row = [0, 1, 2]
col = [0, 1, 2]
A = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

# Convert dense to sparse
A_sparse = sparse.csr_matrix(A_dense)

# Sparse operations
A @ x                           # Matrix-vector product
sparse.linalg.spsolve(A, b)    # Solve Ax = b
sparse.linalg.eigs(A, k=3)     # k largest eigenvalues
sparse.linalg.svds(A, k=3)     # k largest singular values

# Formats:
# CSR (Compressed Sparse Row) - efficient row slicing, matrix-vector products
# CSC (Compressed Sparse Column) - efficient column slicing
# COO (Coordinate) - efficient construction
# LIL (List of Lists) - efficient incremental construction
```

## Complex and Hermitian Matrices

```python
# Complex matrix
A = np.array([[1+2j, 3], [4, 5-1j]])

# Conjugate transpose (Hermitian adjoint)
A.conj().T  # or A.T.conj()

# Check if Hermitian (A = A^H)
np.allclose(A, A.conj().T)

# Eigenvalues of Hermitian matrix (always real)
eigenvalues = np.linalg.eigvalsh(A_hermitian)

# Eigenvalues and eigenvectors of Hermitian
vals, vecs = np.linalg.eigh(A_hermitian)

# Unitary matrix check (U^H @ U = I)
np.allclose(U.conj().T @ U, np.eye(len(U)))
```

## Special Matrices

```python
# Identity
np.eye(n)

# Zeros/Ones
np.zeros((m, n))
np.ones((m, n))

# Diagonal
np.diag([1, 2, 3])        # Create diagonal matrix
np.diag(A)                 # Extract diagonal

# Block diagonal
from scipy.linalg import block_diag
block_diag(A, B, C)

# Toeplitz (constant diagonals)
from scipy.linalg import toeplitz
toeplitz([1, 2, 3], [1, 4, 5])

# Circulant
from scipy.linalg import circulant
circulant([1, 2, 3])

# Hankel
from scipy.linalg import hankel
hankel([1, 2, 3], [3, 4, 5])

# Vandermonde
np.vander([1, 2, 3, 4], N=3)

# Hilbert (notoriously ill-conditioned)
from scipy.linalg import hilbert
hilbert(4)

# Random orthogonal matrix
from scipy.stats import ortho_group
Q = ortho_group.rvs(dim=3)
```

## Numerical Stability Tips

**Avoid:**
- Computing A^(-1) @ b; use `np.linalg.solve(A, b)` instead
- Explicit matrix inverses when solving systems
- Repeated matrix multiplications (accumulates error)

**Prefer:**
- QR over Gram-Schmidt for orthogonalization
- SVD for rank-deficient problems
- `eigh` over `eig` for symmetric/Hermitian matrices
- Pivoted decompositions for ill-conditioned matrices

**Check:**
```python
# Verify solution quality
x = np.linalg.solve(A, b)
residual = np.linalg.norm(A @ x - b)
relative_residual = residual / np.linalg.norm(b)
```

**For ill-conditioned systems:**
```python
# Truncated SVD (regularization)
U, s, Vt = np.linalg.svd(A)
k = np.sum(s > 1e-10)  # Keep significant singular values
x = Vt[:k].T @ (np.diag(1/s[:k]) @ (U[:, :k].T @ b))

# Tikhonov regularization
lambda_reg = 0.01
x = np.linalg.solve(A.T @ A + lambda_reg * np.eye(n), A.T @ b)
```
