#!/usr/bin/env python3
"""Calculus operations CLI tool using SymPy and SciPy."""

import sys
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# ── helpers ──────────────────────────────────────────────────────────────────

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
SYMS = {s: sp.Symbol(s) for s in 'xyztnuvabc'}
SYMS.update({'pi': sp.pi, 'E': sp.E, 'oo': sp.oo, 'inf': sp.oo})

FUNC_MAP = {
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
    'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
    'sqrt': sp.sqrt, 'abs': sp.Abs, 'factorial': sp.factorial,
    'sec': sp.sec, 'csc': sp.csc, 'cot': sp.cot,
}

def parse(expr_str):
    """Parse an expression string into a SymPy expression."""
    local_dict = {**SYMS, **FUNC_MAP}
    return parse_expr(expr_str, local_dict=local_dict, transformations=TRANSFORMS)

def sym(s):
    return SYMS.get(s, sp.Symbol(s))

def parse_point(s):
    """Parse a limit point: 'oo', '-oo', '0', 'pi', numbers, etc."""
    if s in ('oo', 'inf', '+oo'):
        return sp.oo
    if s in ('-oo', '-inf'):
        return -sp.oo
    try:
        return sp.Rational(s)
    except Exception:
        return parse(s)

def pp(expr, label=""):
    """Pretty-print a SymPy expression."""
    prefix = f"{label}: " if label else ""
    print(prefix + str(sp.simplify(expr)))

# ── operations ───────────────────────────────────────────────────────────────

def op_limit(args):
    """limit "expr" var point [side]
    Compute lim_{var->point} expr.
    side: '+' or '-' for one-sided limits (optional)."""
    if len(args) < 3:
        print("Usage: limit \"expr\" var point [+|-]"); return
    expr = parse(args[0])
    v = sym(args[1])
    point = parse_point(args[2])
    side = args[3] if len(args) > 3 else '+-'
    result = sp.limit(expr, v, point, side)
    pp(result, f"lim_{{{args[1]}→{args[2]}}} ({args[0]})")


def op_diff(args):
    """diff "expr" var [n]
    Differentiate expr with respect to var, optionally n times."""
    if len(args) < 2:
        print("Usage: diff \"expr\" var [n]"); return
    expr = parse(args[0])
    v = sym(args[1])
    n = int(args[2]) if len(args) > 2 else 1
    result = sp.diff(expr, v, n)
    order = f"d{n}" if n > 1 else "d"
    pp(result, f"{order}/d{args[1]}{n if n>1 else ''} ({args[0]})")


def op_partial(args):
    """partial "expr" "var1 [var2 ...]"
    Compute partial/mixed derivative. List vars in order of differentiation."""
    if len(args) < 2:
        print('Usage: partial "expr" "var1 var2 ..."'); return
    expr = parse(args[0])
    vars_ = [sym(v) for v in args[1].split()]
    result = sp.diff(expr, *vars_)
    wrt = ''.join(f'∂{v}' for v in args[1].split())
    pp(result, f"∂/{wrt} ({args[0]})")


def op_integrate(args):
    """integrate "expr" var
    Compute indefinite integral ∫ expr d(var)."""
    if len(args) < 2:
        print('Usage: integrate "expr" var'); return
    expr = parse(args[0])
    v = sym(args[1])
    result = sp.integrate(expr, v)
    pp(result, f"∫({args[0]}) d{args[1]}  (+C)")


def op_defint(args):
    """defint "expr" var a b
    Compute definite integral ∫_a^b expr d(var)."""
    if len(args) < 4:
        print('Usage: defint "expr" var a b'); return
    expr = parse(args[0])
    v = sym(args[1])
    a, b = parse_point(args[2]), parse_point(args[3])
    result = sp.integrate(expr, (v, a, b))
    pp(result, f"∫_{args[2]}^{args[3]} ({args[0]}) d{args[1]}")


def op_series(args):
    """series "expr" var [point] [order]
    Taylor/Maclaurin series of expr around point (default 0) to given order (default 6)."""
    if len(args) < 2:
        print('Usage: series "expr" var [point] [order]'); return
    expr = parse(args[0])
    v = sym(args[1])
    point = parse_point(args[2]) if len(args) > 2 else 0
    order = int(args[3]) if len(args) > 3 else 6
    result = sp.series(expr, v, point, order)
    print(f"Series of ({args[0]}) around {args[1]}={point}, O({args[1]}^{order}):")
    print(str(result))


def op_solve_ode(args):
    """solve_ode "rhs" [--order 1|2] [--ic "f(0)=v,fp(0)=v"]
    Solve ODE. For order 1: f'(x)=rhs. For order 2: f''(x)+f(x)=rhs (rhs=0 for homogeneous).
    Use --ic to specify initial conditions: f(0)=1 and/or fp(0)=0."""
    if not args:
        print('Usage: solve_ode "rhs" [--order 2] [--ic "f(0)=1,fp(0)=0"]'); return

    order = 1
    ic_str = None
    rhs_str = args[0]

    i = 1
    while i < len(args):
        if args[i] == '--order' and i+1 < len(args):
            order = int(args[i+1]); i += 2
        elif args[i] == '--ic' and i+1 < len(args):
            ic_str = args[i+1]; i += 2
        else:
            i += 1

    x = sp.Symbol('x')
    f = sp.Function('f')
    # Use Function('f') (not f(x)) so "f(x)" in rhs_str is parsed as the applied function
    rhs = parse_expr(rhs_str, local_dict={**SYMS, **FUNC_MAP, 'f': f},
                     transformations=TRANSFORMS)

    if order == 1:
        ode = sp.Eq(f(x).diff(x), rhs)
    else:
        ode = sp.Eq(f(x).diff(x, order), rhs)

    ics = {}
    if ic_str:
        for part in ic_str.split(','):
            part = part.strip()
            if part.startswith('fp('):
                # f'(point) = value
                pt = float(part[3:part.index(')')])
                val = float(part.split('=')[1])
                ics[f(x).diff(x).subs(x, pt)] = val
            elif part.startswith('f('):
                pt = float(part[2:part.index(')')])
                val = float(part.split('=')[1])
                ics[f(pt)] = val

    try:
        sol = sp.dsolve(ode, f(x), ics=ics if ics else None)
        primes = "'" * order
        print(f"ODE: f{primes}(x) = {rhs_str}")
        print(f"Solution: {sol}")
    except Exception as e:
        print(f"Could not solve symbolically: {e}")
        print("Try scipy.integrate.solve_ivp for numerical solution.")


def op_critical(args):
    """critical "expr" var
    Find critical points of expr and classify them (min/max/inflection)."""
    if len(args) < 2:
        print('Usage: critical "expr" var'); return
    expr = parse(args[0])
    v = sym(args[1])
    d1 = sp.diff(expr, v)
    d2 = sp.diff(expr, v, 2)
    critical_pts = sp.solve(d1, v)
    print(f"f({args[1]}) = {args[0]}")
    print(f"f'({args[1]}) = {d1}")
    if not critical_pts:
        print("No critical points found."); return
    print(f"\nCritical points:")
    for cp in critical_pts:
        val = expr.subs(v, cp)
        d2val = d2.subs(v, cp)
        try:
            d2_sign = sp.sign(d2val)
            if d2_sign > 0:
                kind = "local minimum"
            elif d2_sign < 0:
                kind = "local maximum"
            else:
                kind = "possible inflection (inconclusive)"
        except Exception:
            kind = "unknown"
        print(f"  {args[1]} = {cp}  =>  f = {val}  [{kind}, f''={d2val}]")


def op_gradient(args):
    """gradient "expr" "var1 var2 ..."
    Compute gradient vector of a scalar field."""
    if len(args) < 2:
        print('Usage: gradient "expr" "var1 var2 ..."'); return
    expr = parse(args[0])
    vars_ = [sym(v) for v in args[1].split()]
    grad = [sp.diff(expr, v) for v in vars_]
    print(f"∇({args[0]}) w.r.t. [{args[1]}]:")
    for v, g in zip(args[1].split(), grad):
        print(f"  ∂/∂{v} = {g}")


def op_divergence(args):
    """divergence "Fx" "Fy" ["Fz"] ["var1 var2 [var3]"]
    Compute divergence of a vector field F = (Fx, Fy, Fz)."""
    if len(args) < 3:
        print('Usage: divergence "Fx" "Fy" ["Fz"] ["x y [z]"]'); return
    # Determine if last arg is a variable list or a component
    # Heuristic: if last arg has spaces, it's the variable list
    if ' ' in args[-1]:
        vars_ = [sym(v) for v in args[-1].split()]
        components = [parse(a) for a in args[:-1]]
    else:
        # Default variable names
        var_names = ['x', 'y', 'z'][:len(args)]
        vars_ = [sym(v) for v in var_names]
        components = [parse(a) for a in args]

    div = sum(sp.diff(f, v) for f, v in zip(components, vars_))
    comp_str = ', '.join(str(c) for c in components)
    print(f"div F = ∇·({comp_str}) = {sp.simplify(div)}")


def op_curl(args):
    """curl "Fx" "Fy" "Fz" ["x y z"]
    Compute curl of a 3D vector field."""
    if len(args) < 3:
        print('Usage: curl "Fx" "Fy" "Fz" ["x y z"]'); return
    if len(args) >= 4 and ' ' in args[3]:
        xv, yv, zv = [sym(s) for s in args[3].split()[:3]]
        Fx, Fy, Fz = [parse(args[i]) for i in range(3)]
    else:
        xv, yv, zv = sym('x'), sym('y'), sym('z')
        Fx, Fy, Fz = [parse(a) for a in args[:3]]

    cx = sp.diff(Fz, yv) - sp.diff(Fy, zv)
    cy = sp.diff(Fx, zv) - sp.diff(Fz, xv)
    cz = sp.diff(Fy, xv) - sp.diff(Fx, yv)
    print(f"curl F = ∇×F:")
    print(f"  i: {sp.simplify(cx)}")
    print(f"  j: {sp.simplify(cy)}")
    print(f"  k: {sp.simplify(cz)}")


def op_laplacian(args):
    """laplacian "expr" "var1 var2 ..."
    Compute Laplacian (∇²) of a scalar field."""
    if len(args) < 2:
        print('Usage: laplacian "expr" "var1 var2 ..."'); return
    expr = parse(args[0])
    vars_ = [sym(v) for v in args[1].split()]
    lap = sum(sp.diff(expr, v, 2) for v in vars_)
    print(f"∇²({args[0]}) = {sp.simplify(lap)}")


def op_jacobian(args):
    """jacobian "f1,f2,..." "var1 var2 ..."
    Compute Jacobian matrix of a vector-valued function."""
    if len(args) < 2:
        print('Usage: jacobian "f1,f2,..." "var1 var2 ..."'); return
    funcs = [parse(f.strip()) for f in args[0].split(',')]
    vars_ = [sym(v) for v in args[1].split()]
    J = sp.Matrix([[sp.diff(f, v) for v in vars_] for f in funcs])
    print(f"Jacobian of [{args[0]}] w.r.t. [{args[1]}]:")
    sp.pprint(J)


def op_hessian(args):
    """hessian "expr" "var1 var2 ..."
    Compute Hessian matrix of a scalar function."""
    if len(args) < 2:
        print('Usage: hessian "expr" "var1 var2 ..."'); return
    expr = parse(args[0])
    vars_ = [sym(v) for v in args[1].split()]
    H = sp.hessian(expr, vars_)
    print(f"Hessian of ({args[0]}) w.r.t. [{args[1]}]:")
    sp.pprint(H)
    det = H.det()
    print(f"det(H) = {sp.simplify(det)}")


def op_arclength(args):
    """arclength "expr" var a b
    Compute arc length of y=expr from a to b."""
    if len(args) < 4:
        print('Usage: arclength "expr" var a b'); return
    expr = parse(args[0])
    v = sym(args[1])
    a, b = parse_point(args[2]), parse_point(args[3])
    dy = sp.diff(expr, v)
    integrand = sp.sqrt(1 + dy**2)
    result = sp.integrate(integrand, (v, a, b))
    print(f"Arc length of y={args[0]} from {args[1]}={args[2]} to {args[1]}={args[3]}:")
    print(f"  L = {sp.simplify(result)}")


# ── dispatch ─────────────────────────────────────────────────────────────────

OPERATIONS = {
    'limit':      (op_limit,      'Compute a limit'),
    'diff':       (op_diff,       'Differentiate (optionally nth order)'),
    'partial':    (op_partial,    'Partial/mixed derivative'),
    'integrate':  (op_integrate,  'Indefinite integral'),
    'defint':     (op_defint,     'Definite integral ∫_a^b'),
    'series':     (op_series,     'Taylor/Maclaurin series'),
    'solve_ode':  (op_solve_ode,  'Solve an ODE symbolically'),
    'critical':   (op_critical,   'Find & classify critical points'),
    'gradient':   (op_gradient,   'Gradient ∇f of a scalar field'),
    'divergence': (op_divergence, 'Divergence ∇·F of a vector field'),
    'curl':       (op_curl,       'Curl ∇×F of a 3D vector field'),
    'laplacian':  (op_laplacian,  'Laplacian ∇²f of a scalar field'),
    'jacobian':   (op_jacobian,   'Jacobian matrix of a vector function'),
    'hessian':    (op_hessian,    'Hessian matrix of a scalar function'),
    'arclength':  (op_arclength,  'Arc length of a curve'),
}

EXAMPLES = [
    ('limit "sin(x)/x" x 0',             'Classic sinc limit → 1'),
    ('limit "1/x" x 0 +',                'One-sided limit → ∞'),
    ('diff "x**3 + 2*x" x',              'Derivative → 3x²+2'),
    ('diff "exp(x)*sin(x)" x 2',         'Second derivative'),
    ('partial "x**2*sin(y)" "x y"',      'Mixed partial ∂²/∂x∂y'),
    ('integrate "x**2 * exp(x)" x',      'Integration by parts candidate'),
    ('defint "sin(x)" x 0 pi',           'Definite integral → 2'),
    ('series "exp(x)" x 0 6',            'Maclaurin series for eˣ'),
    ('series "log(1+x)" x 0 5',          'ln(1+x) series'),
    ('solve_ode "f(x)" --order 1',       "f'=f → Cexp(x)"),
    ('solve_ode "0" --order 2 --ic "f(0)=1,fp(0)=0"', "f''+f=0 with ICs → cos(x)"),
    ('critical "x**3 - 3*x" x',          'Find extrema of cubic'),
    ('gradient "x**2 + y**2 + z**2" "x y z"', 'Gradient of r²'),
    ('divergence "x**2" "y**2" "z**2" "x y z"', 'Divergence of r²-field'),
    ('laplacian "x**2 + y**2" "x y"',    'Laplacian → 4'),
    ('arclength "x**2" x 0 1',           'Arc length of parabola'),
]

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        print("Calculus CLI  (SymPy-powered)")
        print("\nUsage: calculus.py <operation> [args...]")
        print("\nOperations:")
        for name, (_, desc) in sorted(OPERATIONS.items()):
            print(f"  {name:<12} - {desc}")
        print("\nExamples:")
        for cmd, note in EXAMPLES:
            print(f"  calculus.py {cmd}")
            print(f"    # {note}")
        print("\nExpression syntax: x**2, sin(x), exp(x), log(x), sqrt(x), pi, E, oo")
        sys.exit(0)

    op = sys.argv[1].lower()
    if op not in OPERATIONS:
        print(f"Unknown operation: {op}")
        print(f"Available: {', '.join(sorted(OPERATIONS.keys()))}")
        sys.exit(1)

    func, _ = OPERATIONS[op]
    try:
        func(sys.argv[2:])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
