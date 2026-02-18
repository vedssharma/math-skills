# Math Skills for Claude Code

This repository contains custom Claude Code skills for mathematical computation. Each skill bundles a system prompt, reference formulas, and a CLI Python script.

## Skills

### Linear Algebra (`linear-algebra.skill`)

Expert system for vector and matrix operations using NumPy/SciPy.

**Use when you need to:**
- Compute vector operations: dot product, cross product, magnitude, normalization, projection, angle between vectors
- Perform matrix operations: multiplication, transpose, inverse, determinant, trace, rank
- Find eigenvalues and eigenvectors
- Run decompositions: LU, QR, SVD, Cholesky
- Solve linear systems (exact, least-squares, or pseudo-inverse)

**CLI script:** `linear-algebra/scripts/linalg.py`

```bash
python scripts/linalg.py <operation> [args...]
# e.g.
python scripts/linalg.py dot "[1,2,3]" "[4,5,6]"
python scripts/linalg.py eig "[[1,2],[3,4]]"
python scripts/linalg.py solve "[[2,1],[1,3]]" "[8,13]"
```

**Reference formulas:** `linear-algebra/references/formulas.md` (condition number, Gram-Schmidt, matrix norms, sparse matrices)

---

### Calculus (`calculus.skill`)

Expert system for symbolic and numerical calculus using SymPy and SciPy.

**Use when you need to:**
- Evaluate limits (one-sided, at infinity)
- Differentiate: basic rules, chain/product/quotient rule, higher-order, partial, implicit
- Integrate: indefinite, definite, improper, double integrals; techniques include substitution, parts, partial fractions
- Expand Taylor/Maclaurin series
- Solve ordinary differential equations (ODEs) symbolically or numerically
- Compute vector calculus: gradient, divergence, curl, Laplacian
- Optimize: find critical points, classify extrema, Lagrange multipliers
- Compute arc length, surface area, and volume of revolution
- Work with Fourier series

**CLI script:** `calculus/scripts/calculus.py`

```bash
python scripts/calculus.py <operation> [args...]
# e.g.
python scripts/calculus.py diff "x**3 + 2*x" x
python scripts/calculus.py defint "sin(x)" x 0 pi
python scripts/calculus.py series "exp(x)" x 0 6
python scripts/calculus.py solve_ode "-x" --ic "f(0)=1"
```

**Reference formulas:** `calculus/references/formulas.md` (integration techniques, derivative/integral tables, series convergence tests, Jacobian, Hessian, numerical methods)

---

## Structure

```
skills/
├── README.md
├── .claude/
│   └── settings.local.json      # Allows pip3 install and WebSearch
├── linear-algebra/
│   ├── SKILL.md                 # Skill prompt and quick reference
│   ├── scripts/linalg.py        # CLI computation script
│   └── references/formulas.md  # Extended formula reference
├── linear-algebra.skill         # Packaged skill archive
├── calculus/
│   ├── SKILL.md                 # Skill prompt and quick reference
│   ├── scripts/calculus.py      # CLI computation script
│   └── references/formulas.md  # Extended formula reference
└── calculus.skill               # Packaged skill archive
```

## Usage

Skills are invoked automatically by Claude Code when you describe a relevant math task, or you can trigger them explicitly:

- **Linear algebra:** "compute the eigenvalues of [[1,2],[3,4]]"
- **Calculus:** "find the derivative of x³·sin(x)" or "solve the ODE y' = -y with y(0)=1"

The `.claude/settings.local.json` pre-approves `pip3 install` (so scripts can install NumPy/SymPy/SciPy if needed) and `WebSearch` for looking up formulas or methods.
