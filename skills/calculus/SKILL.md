---
name: calculus
description: Expert calculus skill for symbolic and numerical computation. Use when performing calculations involving limits, derivatives (basic rules, chain/product/quotient rule, higher-order, partial, implicit), integrals (indefinite/definite/improper), integration techniques (substitution, parts, partial fractions), Taylor/Maclaurin series, ordinary differential equations (ODEs), vector calculus (gradient, divergence, curl, Laplacian), optimization (critical points, extrema), or any numerical calculus task. Also use for related rates, arc length, surface/volume of revolution, and Fourier series.
---

# Calculus

Expert system for symbolic and numerical calculus using SymPy and SciPy.

## Quick Reference

### Symbolic Setup

```python
import sympy as sp

x, y, z, t = sp.symbols('x y z t')
n = sp.Symbol('n', positive=True)
```

### Limits

```python
# lim_{x->0} sin(x)/x
sp.limit(sp.sin(x)/x, x, 0)          # => 1

# One-sided limits
sp.limit(1/x, x, 0, '+')             # => oo
sp.limit(1/x, x, 0, '-')             # => -oo

# Limit at infinity
sp.limit(x**2 / sp.exp(x), x, sp.oo)  # => 0
```

### Derivatives

```python
f = x**3 + 2*x**2 - 5*x + 1

# First derivative
sp.diff(f, x)                          # 3x² + 4x - 5

# nth derivative
sp.diff(f, x, 3)                       # d³f/dx³

# Partial derivatives
g = x**2 * sp.sin(y)
sp.diff(g, x)                          # ∂g/∂x
sp.diff(g, x, y)                       # ∂²g/∂x∂y

# Implicit differentiation: x² + y² = 1
F = x**2 + y**2 - 1
dy_dx = -sp.diff(F, x) / sp.diff(F, y)
```

### Integrals

```python
# Indefinite integral
sp.integrate(x**2, x)                  # x³/3

# Definite integral
sp.integrate(x**2, (x, 0, 1))         # 1/3

# Improper integral
sp.integrate(sp.exp(-x), (x, 0, sp.oo))  # 1

# Double integral ∫∫ x*y dy dx
sp.integrate(x*y, (y, 0, x), (x, 0, 1))

# Numerical integration (when symbolic fails)
from scipy import integrate
result, err = integrate.quad(lambda x: x**2, 0, 1)
```

### Taylor / Maclaurin Series

```python
# Taylor series around x=0 (Maclaurin), order n
sp.series(sp.exp(x), x, 0, 6)         # 1 + x + x²/2! + ...
sp.series(sp.sin(x), x, 0, 7)
sp.series(sp.cos(x), x, 0, 6)

# Around arbitrary point
sp.series(sp.log(x), x, 1, 5)         # Taylor at x=1

# Remove O() remainder
sp.series(sp.exp(x), x, 0, 4).removeO()
```

### Differential Equations

```python
f = sp.Function('f')

# First-order: f'(x) = f(x)
ode = sp.Eq(f(x).diff(x), f(x))
sp.dsolve(ode, f(x))                   # f(x) = C1*exp(x)

# Second-order: f''(x) + f(x) = 0
ode2 = sp.Eq(f(x).diff(x, 2) + f(x), 0)
sp.dsolve(ode2, f(x))                  # f(x) = C1*sin(x) + C2*cos(x)

# With initial conditions
ics = {f(0): 1, f(x).diff(x).subs(x, 0): 0}
sp.dsolve(ode2, f(x), ics=ics)

# Numerical ODE (scipy)
from scipy.integrate import solve_ivp
sol = solve_ivp(lambda t, y: y, [0, 5], [1])  # dy/dt = y, y(0)=1
```

### Vector Calculus

```python
# Gradient of scalar field f(x,y,z)
f = x**2 + y**2 + z**2
grad = [sp.diff(f, var) for var in (x, y, z)]   # [2x, 2y, 2z]

# Divergence of F = [Fx, Fy, Fz]
Fx, Fy, Fz = x**2, y**2, z**2
div = sp.diff(Fx, x) + sp.diff(Fy, y) + sp.diff(Fz, z)   # 2x+2y+2z

# Curl of F
curl_x = sp.diff(Fz, y) - sp.diff(Fy, z)
curl_y = sp.diff(Fx, z) - sp.diff(Fz, x)
curl_z = sp.diff(Fy, x) - sp.diff(Fx, y)

# Laplacian of scalar f
laplacian = sum(sp.diff(f, v, 2) for v in (x, y, z))
```

### Optimization

```python
f = x**3 - 3*x

# Critical points
critical = sp.solve(sp.diff(f, x), x)       # [-1, 1]

# Classify via second derivative test
d2 = sp.diff(f, x, 2)
for cp in critical:
    val = d2.subs(x, cp)
    print(cp, "min" if val > 0 else "max" if val < 0 else "inflection")

# Constrained optimization (Lagrange multipliers)
g = x**2 + y**2 - 1   # constraint g=0
L = f - sp.Symbol('lam') * g
```

## Script Usage

Run `scripts/calculus.py` for quick CLI computations:

```bash
python scripts/calculus.py <operation> [args...]
```

Operations:
- `limit "expr" var point` — Compute a limit
- `diff "expr" var [n]` — Differentiate (optionally nth order)
- `partial "expr" "var1 var2 ..."` — Partial/mixed derivative
- `integrate "expr" var` — Indefinite integral
- `defint "expr" var a b` — Definite integral
- `series "expr" var [point] [order]` — Taylor series
- `solve_ode "ode_rhs" [--ic "f(0)=v,fp(0)=v"]` — Solve ODE f'=rhs or f''+f=0
- `critical "expr" var` — Find and classify critical points
- `gradient "expr" "var1 var2 ..."` — Gradient vector
- `divergence "Fx Fy Fz" "x y z"` — Divergence of a vector field
- `laplacian "expr" "var1 var2 ..."` — Laplacian of scalar field

Expression syntax uses Python/SymPy: `x**2`, `sin(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, `pi`, `E`.

## Advanced Topics

See `references/formulas.md` for:
- Integration techniques (substitution, by parts, partial fractions, trig sub)
- Common derivatives and integrals table
- Series convergence tests
- Multivariable calculus (chain rule, Jacobian, Hessian)
- Numerical differentiation and integration
- Arc length, surface area, volume of revolution
- Fourier series
