# Calculus Reference

## Table of Contents
- [Derivative Rules](#derivative-rules)
- [Common Derivatives](#common-derivatives)
- [Integration Techniques](#integration-techniques)
- [Common Integrals](#common-integrals)
- [Series & Convergence](#series--convergence)
- [Multivariable Calculus](#multivariable-calculus)
- [Vector Calculus Identities](#vector-calculus-identities)
- [ODEs Cheat Sheet](#odes-cheat-sheet)
- [Numerical Methods](#numerical-methods)
- [Geometric Applications](#geometric-applications)

---

## Derivative Rules

| Rule | Formula |
|------|---------|
| Power | d/dx xⁿ = n·xⁿ⁻¹ |
| Constant | d/dx c = 0 |
| Sum | d/dx [f±g] = f'±g' |
| Product | d/dx [fg] = f'g + fg' |
| Quotient | d/dx [f/g] = (f'g − fg') / g² |
| Chain | d/dx f(g(x)) = f'(g(x))·g'(x) |
| Inverse | d/dx f⁻¹(x) = 1 / f'(f⁻¹(x)) |
| Implicit | F(x,y)=0 → dy/dx = −Fₓ/Fᵧ |

---

## Common Derivatives

| f(x) | f'(x) |
|------|-------|
| xⁿ | n·xⁿ⁻¹ |
| eˣ | eˣ |
| aˣ | aˣ ln a |
| ln x | 1/x |
| log_a x | 1/(x ln a) |
| sin x | cos x |
| cos x | −sin x |
| tan x | sec²x |
| cot x | −csc²x |
| sec x | sec x tan x |
| csc x | −csc x cot x |
| arcsin x | 1/√(1−x²) |
| arccos x | −1/√(1−x²) |
| arctan x | 1/(1+x²) |
| sinh x | cosh x |
| cosh x | sinh x |
| tanh x | sech²x |

```python
import sympy as sp
x = sp.Symbol('x')

# Verify any derivative
sp.diff(sp.atan(x), x)   # => 1/(x**2 + 1)
```

---

## Integration Techniques

### 1. U-Substitution

Use when integrand contains f(g(x))·g'(x):

```python
# ∫ 2x·cos(x²) dx  →  let u = x², du = 2x dx
sp.integrate(2*x * sp.cos(x**2), x)   # => sin(x²)
```

**Pattern:** look for a function and its derivative inside the integrand.

### 2. Integration by Parts

∫ u dv = uv − ∫ v du

**LIATE priority for u:** Logarithmic, Inverse trig, Algebraic, Trig, Exponential

```python
# ∫ x·eˣ dx  →  u=x, dv=eˣ dx
sp.integrate(x * sp.exp(x), x)   # => (x-1)*exp(x)

# ∫ ln(x) dx  →  u=ln(x), dv=dx
sp.integrate(sp.log(x), x)       # => x*log(x) - x

# Tabular method for repeated IBP (e.g. ∫ x³eˣ dx):
sp.integrate(x**3 * sp.exp(x), x)
```

### 3. Partial Fractions

For rational functions P(x)/Q(x) where deg(P) < deg(Q):

```python
from sympy import apart
expr = (x**2 + 1) / ((x - 1) * (x + 2) * (x + 3))
apart(expr, x)   # Decompose into partial fractions

# Then integrate term by term
sp.integrate(apart(expr, x), x)
```

**Forms:**
- `A/(x-a)` for linear factors
- `(Ax+B)/(x²+bx+c)` for irreducible quadratics
- `A/(x-a)ⁿ` for repeated factors

### 4. Trigonometric Substitution

| Integrand contains | Substitution | Identity used |
|-------------------|-------------|----------------|
| √(a²−x²) | x = a sin θ | 1−sin²θ = cos²θ |
| √(a²+x²) | x = a tan θ | 1+tan²θ = sec²θ |
| √(x²−a²) | x = a sec θ | sec²θ−1 = tan²θ |

```python
# ∫ √(1-x²) dx  →  x = sin(θ)
sp.integrate(sp.sqrt(1 - x**2), x)   # => x*sqrt(1-x²)/2 + asin(x)/2
```

### 5. Trigonometric Integrals

```python
# ∫ sinⁿx cosᵐx dx strategies:
# - Odd power of sin: save one sin, convert rest via cos²=1-sin²
# - Odd power of cos: save one cos, convert rest via sin²=1-cos²
# - Both even: use half-angle identities sin²x=(1-cos2x)/2

# Half-angle identities
sp.integrate(sp.sin(x)**2, x)    # => x/2 - sin(2x)/4
sp.integrate(sp.cos(x)**2, x)    # => x/2 + sin(2x)/4

# ∫ tanⁿx: use tan²x = sec²x - 1 reduction
sp.integrate(sp.tan(x)**4, x)
```

### 6. Completing the Square

For integrals with ax²+bx+c in denominator or under radical:

```
ax² + bx + c = a(x + b/2a)² + (c - b²/4a)
```

```python
# ∫ 1/(x²+4x+5) dx
expr = 1/(x**2 + 4*x + 5)
sp.integrate(expr, x)   # => atan(x+2)
```

### 7. Improper Integrals

```python
# Infinite bounds
sp.integrate(sp.exp(-x), (x, 0, sp.oo))       # => 1
sp.integrate(1/(x**2 + 1), (x, -sp.oo, sp.oo)) # => pi

# Discontinuity at endpoint
sp.integrate(1/sp.sqrt(x), (x, 0, 1))          # => 2

# Test convergence without computing:
# p-integral ∫₁^∞ 1/xᵖ dx converges iff p > 1
# ∫₀^1 1/xᵖ dx converges iff p < 1
```

---

## Common Integrals

| ∫ f(x) dx | Result |
|-----------|--------|
| xⁿ | xⁿ⁺¹/(n+1) + C, n≠−1 |
| 1/x | ln|x| + C |
| eˣ | eˣ + C |
| aˣ | aˣ/ln(a) + C |
| sin x | −cos x + C |
| cos x | sin x + C |
| tan x | −ln|cos x| + C |
| sec²x | tan x + C |
| sec x tan x | sec x + C |
| 1/√(1−x²) | arcsin x + C |
| 1/(1+x²) | arctan x + C |
| 1/√(x²+a²) | ln|x+√(x²+a²)| + C |
| sinh x | cosh x + C |
| cosh x | sinh x + C |

---

## Series & Convergence

### Maclaurin Series (around x=0)

| f(x) | Series | Radius |
|------|--------|--------|
| eˣ | Σ xⁿ/n! | ∞ |
| sin x | Σ (−1)ⁿx^(2n+1)/(2n+1)! | ∞ |
| cos x | Σ (−1)ⁿx^(2n)/(2n)! | ∞ |
| ln(1+x) | Σ (−1)ⁿ⁺¹xⁿ/n | (−1,1] |
| 1/(1−x) | Σ xⁿ | (−1,1) |
| arctan x | Σ (−1)ⁿx^(2n+1)/(2n+1) | [−1,1] |
| (1+x)ᵅ | Σ C(α,n)xⁿ | (−1,1) |

```python
x = sp.Symbol('x')
sp.series(sp.exp(x), x, 0, 8).removeO()
sp.series(sp.sin(x), x, 0, 9).removeO()
```

### Convergence Tests

| Test | When to use | Converges if |
|------|-------------|--------------|
| Ratio | aₙ involves n! or aⁿ | lim|aₙ₊₁/aₙ| < 1 |
| Root | aₙ involves nᵗʰ powers | lim|aₙ|^(1/n) < 1 |
| Integral | aₙ = f(n), f decreasing | ∫₁^∞ f(x)dx converges |
| Comparison | aₙ ≤ bₙ (known) | Σbₙ converges |
| Limit Comparison | Similar growth | lim aₙ/bₙ = L > 0, same as Σbₙ |
| Alternating (Leibniz) | (−1)ⁿaₙ | aₙ→0, decreasing |
| p-series | 1/nᵖ | p > 1 |

### Taylor Remainder

|Rₙ(x)| ≤ M|x−a|ⁿ⁺¹/(n+1)! where M = max|f^(n+1)| on interval

```python
# Error bound example: approximate sin(0.1) with 3 terms
# |R₃| ≤ |0.1|⁵/5! ≈ 8.3×10⁻⁹
```

---

## Multivariable Calculus

### Chain Rule (Multivariable)

If z = f(x,y), x = x(t), y = y(t):
```
dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)
```

```python
x, y, t = sp.symbols('x y t')
f = x**2 + y**2
x_t = sp.cos(t); y_t = sp.sin(t)
dz_dt = sp.diff(f, x)*sp.diff(x_t, t) + sp.diff(f, y)*sp.diff(y_t, t)
```

### Directional Derivative

Dᵤf = ∇f · û  (û = unit vector in direction u)

```python
import numpy as np
grad_f = [2, 4]          # ∇f at a point
u = np.array([1, 1]) / np.sqrt(2)   # direction
D_u = np.dot(grad_f, u)             # directional derivative
```

### Jacobian Matrix

For F: Rⁿ → Rᵐ, J[i,j] = ∂Fᵢ/∂xⱼ

```python
x, y = sp.symbols('x y')
F = [x**2 - y, x*y + 1]
vars_ = [x, y]
J = sp.Matrix([[sp.diff(f, v) for v in vars_] for f in F])
```

### Hessian Matrix

H[i,j] = ∂²f/∂xᵢ∂xⱼ

```python
f = x**3 + y**3 - 3*x*y
H = sp.hessian(f, [x, y])
# Classify critical point: det(H)>0, H[0,0]>0 → min; det(H)>0, H[0,0]<0 → max; det(H)<0 → saddle
```

### Lagrange Multipliers

Maximize/minimize f(x,y) subject to g(x,y) = 0:
∇f = λ∇g  and  g = 0

```python
x, y, lam = sp.symbols('x y lambda')
f = x**2 + y**2          # objective
g = x + y - 1            # constraint
system = [
    sp.Eq(sp.diff(f,x), lam * sp.diff(g,x)),
    sp.Eq(sp.diff(f,y), lam * sp.diff(g,y)),
    sp.Eq(g, 0)
]
sp.solve(system, [x, y, lam])
```

---

## Vector Calculus Identities

### Fundamental Identities

| Identity | Formula |
|----------|---------|
| Gradient of product | ∇(fg) = f∇g + g∇f |
| Divergence of fF | ∇·(fF) = f(∇·F) + F·∇f |
| Curl of gradient | ∇×(∇f) = 0 |
| Divergence of curl | ∇·(∇×F) = 0 |
| Curl of curl | ∇×(∇×F) = ∇(∇·F) − ∇²F |
| Laplacian | ∇²f = ∇·(∇f) |

### Theorems

**Green's Theorem (2D):**
∮_C (P dx + Q dy) = ∬_D (∂Q/∂x − ∂P/∂y) dA

**Stokes' Theorem (3D):**
∮_C F·dr = ∬_S (∇×F)·dS

**Divergence Theorem:**
∯_S F·dS = ∭_V (∇·F) dV

```python
# Verify a field is conservative (curl = 0)
Fx = y; Fy = x   # F = (y, x)
conservative = sp.diff(Fy, x) - sp.diff(Fx, y) == 0   # True
# Find potential: φ where ∇φ = F
phi = sp.integrate(Fx, x)   # integrate ∂φ/∂x = y → φ = xy
```

---

## ODEs Cheat Sheet

### First-Order

| Type | Form | Solution method |
|------|------|-----------------|
| Separable | dy/dx = f(x)g(y) | Separate and integrate |
| Linear | y' + P(x)y = Q(x) | Integrating factor μ = e^∫P dx |
| Exact | M dx + N dy = 0, Mᵧ=Nₓ | F: ∂F/∂x=M, ∂F/∂y=N |
| Bernoulli | y' + P(x)y = Q(x)yⁿ | Sub v = y^(1-n) → linear |
| Homogeneous | y' = F(y/x) | Sub v = y/x |

```python
f = sp.Function('f')
x = sp.Symbol('x')

# Separable: dy/dx = xy
sp.dsolve(sp.Eq(f(x).diff(x), x * f(x)), f(x))   # C1*exp(x²/2)

# Linear: y' + 2y = 4x
sp.dsolve(sp.Eq(f(x).diff(x) + 2*f(x), 4*x), f(x))

# Bernoulli: y' - y = xy²  (n=2, v=1/y)
sp.dsolve(sp.Eq(f(x).diff(x) - f(x), x*f(x)**2), f(x))
```

### Second-Order Linear with Constant Coefficients

ay'' + by' + cy = g(x)

```python
# Characteristic equation: ar² + br + c = 0
# Roots determine homogeneous solution:
# r real distinct: C1*e^(r1*x) + C2*e^(r2*x)
# r real repeated: (C1 + C2*x)*e^(r*x)
# r complex α±βi: e^(αx)(C1*cos(βx) + C2*sin(βx))

# Method of Undetermined Coefficients for particular solution:
# g(x) = polynomial → try polynomial of same degree
# g(x) = e^(ax) → try A*e^(ax)
# g(x) = sin/cos → try A*sin + B*cos

ode = sp.Eq(f(x).diff(x,2) - 5*f(x).diff(x) + 6*f(x), sp.exp(2*x))
sp.dsolve(ode, f(x))
```

### Numerical ODE Solvers

```python
from scipy.integrate import solve_ivp
import numpy as np

# dy/dt = -2y, y(0) = 1
def ode(t, y): return [-2*y[0]]
sol = solve_ivp(ode, [0, 5], [1], dense_output=True)
# sol.y[0] contains y values, sol.t contains t values

# System of ODEs: Lotka-Volterra
def lotka_volterra(t, state, a=1.5, b=1, c=3, d=1):
    x, y = state
    return [a*x - b*x*y, -c*y + d*x*y]

sol = solve_ivp(lotka_volterra, [0, 15], [10, 5], max_step=0.1)
```

---

## Numerical Methods

### Numerical Differentiation

```python
import numpy as np

# Forward difference: f'(x) ≈ [f(x+h) - f(x)] / h  (O(h))
# Central difference: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)  (O(h²))
# Second derivative: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²

def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def second_diff(f, x, h=1e-5):
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

# Example
f = lambda x: np.sin(x)
central_diff(f, np.pi/4)   # ≈ cos(π/4) ≈ 0.7071
```

### Numerical Integration

```python
from scipy import integrate

# Gaussian quadrature (general, high accuracy)
result, err = integrate.quad(lambda x: np.sin(x), 0, np.pi)   # => (2.0, ~2e-14)

# Double integral ∫₀¹ ∫₀¹ x*y dy dx
result, err = integrate.dblquad(lambda y, x: x*y, 0, 1, 0, 1)

# Trapezoidal rule (for tabulated data)
x = np.linspace(0, np.pi, 100)
y = np.sin(x)
np.trapz(y, x)   # ≈ 2.0

# Simpson's rule
from scipy.integrate import simpson
simpson(y, x=x)  # more accurate
```

### Root Finding (for solving equations)

```python
from scipy.optimize import brentq, fsolve

# Find root of f on interval [a, b]
f = lambda x: x**3 - x - 2
root = brentq(f, 1, 2)   # => 1.5214

# Newton's method
from scipy.optimize import newton
newton(f, x0=1.5, fprime=lambda x: 3*x**2 - 1)

# Sympy exact roots
sp.solve(x**3 - x - 2, x)
```

### Optimization

```python
from scipy.optimize import minimize, minimize_scalar

# Scalar minimization
result = minimize_scalar(lambda x: x**4 - 3*x**2, bounds=(-2, 0), method='bounded')

# Multivariate minimization
result = minimize(lambda v: (v[0]-1)**2 + (v[1]-2.5)**2, x0=[0, 0])

# Constrained optimization
from scipy.optimize import minimize
constraints = {'type': 'eq', 'fun': lambda v: v[0] + v[1] - 1}
minimize(lambda v: v[0]**2 + v[1]**2, x0=[0.5, 0.5], constraints=constraints)
```

---

## Geometric Applications

### Arc Length

L = ∫_a^b √(1 + [f'(x)]²) dx

```python
f = x**2
arc = sp.integrate(sp.sqrt(1 + sp.diff(f,x)**2), (x, 0, 1))
sp.simplify(arc)   # => sqrt(5)/2 + asinh(2)/4
```

### Surface Area of Revolution

About x-axis: S = 2π ∫_a^b f(x) √(1 + [f'(x)]²) dx

```python
f = sp.sqrt(x)
S = 2*sp.pi * sp.integrate(f * sp.sqrt(1 + sp.diff(f,x)**2), (x, 0, 4))
```

### Volume of Revolution

**Disk method** (about x-axis): V = π ∫_a^b [f(x)]² dx

**Shell method** (about y-axis): V = 2π ∫_a^b x·f(x) dx

**Washer method**: V = π ∫_a^b ([f(x)]² − [g(x)]²) dx

```python
# Volume of sphere: f(x) = √(r²-x²), about x-axis
r = sp.Symbol('r', positive=True)
V = sp.pi * sp.integrate((r**2 - x**2), (x, -r, r))   # => 4πr³/3
```

### Fourier Series

f(x) = a₀/2 + Σₙ [aₙcos(nπx/L) + bₙsin(nπx/L)]

```python
L = sp.pi

def fourier_coefficients(f_expr, n_terms=5):
    a0 = (1/L) * sp.integrate(f_expr, (x, -L, L))
    coeffs = [(a0/2, 0)]
    for n in range(1, n_terms+1):
        an = (1/L) * sp.integrate(f_expr * sp.cos(n*x), (x, -L, L))
        bn = (1/L) * sp.integrate(f_expr * sp.sin(n*x), (x, -L, L))
        coeffs.append((an, bn))
    return coeffs

# Square wave: f(x) = sign(x) on (-π, π)
# bn = 4/(nπ) for odd n, 0 for even n  =>  f(x) ≈ (4/π)[sin(x) + sin(3x)/3 + ...]
```

### L'Hôpital's Rule

For 0/0 or ∞/∞ forms: lim f/g = lim f'/g' (if limit exists)

```python
# ∞/∞ form: lim x→∞ (x²/eˣ)
# Apply twice: x²/eˣ → 2x/eˣ → 2/eˣ → 0
sp.limit(x**2 / sp.exp(x), x, sp.oo)   # => 0

# 0·∞ form: rewrite as 0/0 or ∞/∞
sp.limit(x * sp.log(x), x, 0, '+')     # => 0  (rewrite: log(x)/(1/x))

# 1^∞, 0⁰, ∞⁰: take ln first
sp.limit((1 + 1/x)**x, x, sp.oo)       # => E
```
