#!/usr/bin/env python3
"""Linear algebra operations CLI tool."""

import sys
import json
import numpy as np
from scipy import linalg

def parse_array(s):
    """Parse a string like '[1,2,3]' or '[[1,2],[3,4]]' into numpy array."""
    return np.array(json.loads(s))

def format_output(result, name="Result"):
    """Format numpy array for display."""
    if isinstance(result, tuple):
        return result
    if isinstance(result, np.ndarray):
        if result.ndim == 1:
            return f"{name}: [{', '.join(f'{x:.6g}' for x in result)}]"
        else:
            rows = ['  [' + ', '.join(f'{x:.6g}' for x in row) + ']' for row in result]
            return f"{name}:\n[\n" + ',\n'.join(rows) + "\n]"
    return f"{name}: {result:.6g}"

# Vector operations
def dot(v1, v2):
    return format_output(np.dot(v1, v2), "Dot product")

def cross(v1, v2):
    return format_output(np.cross(v1, v2), "Cross product")

def norm(v):
    return format_output(np.linalg.norm(v), "Magnitude")

def normalize(v):
    return format_output(v / np.linalg.norm(v), "Unit vector")

def angle(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
    degrees = np.degrees(np.arccos(cos_angle))
    return format_output(degrees, "Angle (degrees)")

def proj(v1, v2):
    """Project v1 onto v2."""
    return format_output((np.dot(v1, v2) / np.dot(v2, v2)) * v2, "Projection")

# Matrix operations
def matmul(A, B):
    return format_output(A @ B, "Product")

def det(A):
    return format_output(np.linalg.det(A), "Determinant")

def inv(A):
    return format_output(np.linalg.inv(A), "Inverse")

def trans(A):
    return format_output(A.T, "Transpose")

def trace(A):
    return format_output(np.trace(A), "Trace")

def rank(A):
    return format_output(np.linalg.matrix_rank(A), "Rank")

def eig(A):
    vals, vecs = np.linalg.eig(A)
    print("Eigenvalues:", [f"{v:.6g}" for v in vals])
    print("Eigenvectors (columns):")
    print(format_output(vecs, ""))
    return None

def lu(A):
    P, L, U = linalg.lu(A)
    print("P (permutation):")
    print(format_output(P, ""))
    print("\nL (lower triangular):")
    print(format_output(L, ""))
    print("\nU (upper triangular):")
    print(format_output(U, ""))
    return None

def qr(A):
    Q, R = np.linalg.qr(A)
    print("Q (orthogonal):")
    print(format_output(Q, ""))
    print("\nR (upper triangular):")
    print(format_output(R, ""))
    return None

def svd(A):
    U, s, Vt = np.linalg.svd(A)
    print("U:")
    print(format_output(U, ""))
    print("\nSingular values:", [f"{v:.6g}" for v in s])
    print("\nV^T:")
    print(format_output(Vt, ""))
    return None

def chol(A):
    L = np.linalg.cholesky(A)
    return format_output(L, "L (A = L @ L.T)")

def solve(A, b):
    x = np.linalg.solve(A, b)
    return format_output(x, "Solution x")

def lstsq(A, b):
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(format_output(x, "Least squares solution"))
    if len(residuals) > 0:
        print(f"Residual sum of squares: {residuals[0]:.6g}")
    return None

def pinv(A):
    return format_output(np.linalg.pinv(A), "Pseudo-inverse")

def cond(A):
    return format_output(np.linalg.cond(A), "Condition number")

OPERATIONS = {
    'dot': (dot, 2, "Dot product of two vectors"),
    'cross': (cross, 2, "Cross product of two 3D vectors"),
    'norm': (norm, 1, "Magnitude/L2 norm of vector"),
    'normalize': (normalize, 1, "Normalize vector to unit length"),
    'angle': (angle, 2, "Angle between two vectors in degrees"),
    'proj': (proj, 2, "Project first vector onto second"),
    'matmul': (matmul, 2, "Matrix multiplication"),
    'det': (det, 1, "Determinant of square matrix"),
    'inv': (inv, 1, "Inverse of square matrix"),
    'trans': (trans, 1, "Transpose of matrix"),
    'trace': (trace, 1, "Trace of square matrix"),
    'rank': (rank, 1, "Rank of matrix"),
    'eig': (eig, 1, "Eigenvalues and eigenvectors"),
    'lu': (lu, 1, "LU decomposition"),
    'qr': (qr, 1, "QR decomposition"),
    'svd': (svd, 1, "Singular value decomposition"),
    'chol': (chol, 1, "Cholesky decomposition"),
    'solve': (solve, 2, "Solve Ax = b"),
    'lstsq': (lstsq, 2, "Least squares solution"),
    'pinv': (pinv, 1, "Moore-Penrose pseudo-inverse"),
    'cond': (cond, 1, "Condition number"),
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        print("Linear Algebra CLI")
        print("\nUsage: linalg.py <operation> [args...]")
        print("\nOperations:")
        for name, (_, nargs, desc) in sorted(OPERATIONS.items()):
            print(f"  {name:<10} - {desc}")
        print("\nExamples:")
        print("  linalg.py dot '[1,2,3]' '[4,5,6]'")
        print("  linalg.py det '[[1,2],[3,4]]'")
        print("  linalg.py solve '[[2,1],[1,3]]' '[4,5]'")
        sys.exit(0)

    op = sys.argv[1].lower()
    if op not in OPERATIONS:
        print(f"Unknown operation: {op}")
        print(f"Available: {', '.join(sorted(OPERATIONS.keys()))}")
        sys.exit(1)

    func, nargs, _ = OPERATIONS[op]
    if len(sys.argv) - 2 != nargs:
        print(f"Operation '{op}' requires {nargs} argument(s)")
        sys.exit(1)

    args = [parse_array(a) for a in sys.argv[2:2+nargs]]
    result = func(*args)
    if result is not None:
        print(result)

if __name__ == '__main__':
    main()
