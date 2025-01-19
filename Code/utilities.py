def bisection_method(f, a, b, tol=1e-6):
    m = (a + b) / 2

    if f(a) * f(m) < 0:
        b = m
    else:
        a = m

    if abs(f(m)) < tol:
        return m
    else:
        return bisection_method(f, a, b, tol)