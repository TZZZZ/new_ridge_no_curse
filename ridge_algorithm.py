import random 
import logging
from math import erf, sqrt
from time import time

import numpy as np
from numpy.polynomial import polynomial
import matplotlib.pyplot as plt
import sympy
from scipy import integrate


# Utility functions

def embed_polynomials_l2(p1, p2, l=1.0):
    """Find lambda for inclusion p1->p2, i.e., such that p1(t) ~ p2(t/lambda), |t|<l.
    
    Params:
        p1, p2  -- instances of polynomial.Polynomial class
        l       -- defines embedding segment [-l,l]
    """

    # we minimize S(mu) = int_{-l}^l |p1(t)-p2(mu t)|^2 dt
    # S(mu) is a polynomial in mu; to calculate it we use sympy and Polynomial class
    mu = sympy.Symbol('mu')
    assert p1.degree() == p2.degree()
    q = polynomial.Polynomial([p1.coef[i] - p2.coef[i] * mu**i for i in range(p1.degree())])
    q = q**2
    q = q.integ()
    S = q(l) - q(-l)
    S_coeff = [float(c) for c in reversed(sympy.Poly(S.expand()).all_coeffs())]  # sympy magic
    S_poly = polynomial.Polynomial(S_coeff)
    min_value, min_mu = minimize_polynomial(S_poly, -1, 1)
    return 1 / min_mu


def extremize_polynomial(poly, a, b):
    """Find min/max values and args for polynomial on [a,b]."""

    # polynomial P has extremum either in P'(x)=0, or x=a, or x=b
    roots = poly.deriv().roots()
    real_roots = [root.real for root in roots if abs(root.imag) < 1e-8]
    active_roots = [root for root in real_roots if a <= root <= b]
    points = active_roots + [a, b]

    values = polynomial.polyval(points, poly.coef)
    min_idx = np.argmin(values)
    max_idx = np.argmax(values)
    return {
        'min': values[min_idx], 'argmin': points[min_idx],
        'max': values[max_idx], 'argmax': points[max_idx],
    }


def minimize_polynomial(poly, a, b):
    """Find minimal value and argmin for poly(t)->min, a <= t <= b."""
    extr = extremize_polynomial(poly, a, b)
    return extr['min'], extr['argmin']


class RidgeSolver:
    """Recover a ridge function f(x)=phi(<a,x>) using f evaluations.
    Params:
        n       --  dimension of the problem
        f_eps   --  function with error
        a       --  true vector a (may be None; used for quality analysis)
        phi     --  true function phi (may be None; use for quality analysis)
    Also some techical params for internal use:
        M, M1, N1, N2, N3, l, ...
    """

    def __init__(self, n, f_eps, M, M1, N1, N2, N3, l=1.0, a=None, phi=None):
        self.n = n
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.M = M
        self.M1 = M1
        self.l = l
        self.f_eps = f_eps
        self.a = np.array(a) if a is not None else None
        self.phi = phi

    def get_random_unit_vector(self):
        v = np.array([random.gauss(0, 1) for _ in range(self.n)])
        return v / np.linalg.norm(v)

    def fit_polynomial(self, gamma):
        """Fit polynomial to a function phi(t_k <a,gamma>), |t_k|<=1."""
        ts = np.linspace(-1, 1, 2 * self.N1 + 1)
        ys = [self.f_eps(t * gamma) for t in ts]
        return polynomial.Polynomial.fit(ts, ys, deg=self.M)

    def check_fitting(self, gamma, poly):
        v_gamma = np.dot(gamma, self.a)
        ts = np.linspace(-self.l*np.sqrt(50), self.l*np.sqrt(50), 10)
        return max(abs(poly(t) - self.phi(v_gamma * t)) for t in ts)

    def get_oscillation(self, poly, h=1):
        """return max_{|t|<=h} |poly(t)-poly(0)|"""
        extr = extremize_polynomial(poly - poly(0), -h, h)
        return max(abs(extr['max']), abs(extr['min']))

    def solve(self):
        typical_gamma = self.step_get_typical_gamma()
        if len(typical_gamma) == 1:
            print("Function is almost constant:", self.f_eps(np.zeros(self.n)))
        else:
            newa = self.step_approximate_a(typical_gamma)
            newphr = self.step_approximate_phi(newa)

    def step_get_typical_gamma(self):
        """Find gamma with 0.45<|v_gamma|<0.75."""

        N2 = self.N2

        # Generate N2 gammas
        gammas = [self.get_random_unit_vector() for _ in range(N2)]

        if self.a is not None:
            real_v = sqrt(self.n) * np.array([np.dot(self.a, gamma) for gamma in gammas])
            real_abs_v = np.abs(real_v)

        # Second part of algorithm
        all_poly = [self.fit_polynomial(gamma) for gamma in gammas]  # polynom coefficients for all gammas
        
        is_const = self.substep_check_constant(all_poly)
        if is_const:
            return [None]
        
        logging.warning('start embeddings ...')
        embed_info = {i: {j: None for j in range(N2)} for i in range(N2)}  # if phi_i -> phi_j, embed_info[i][j] = corresponding lambda
        bound_minus = N2 * 0.43
        bound_plus = N2 *0.45
        v0 = -1
        best_err = 1000
        for j in range(N2):
            logging.warning('embed j=%d', j)
            am_good = 0
            for i in range(N2):
                vall = embed_polynomials_l2(all_poly[i], all_poly[j], l=self.l)
                if abs(vall) != 1:
                    embed_info[i][j] = vall
                if embed_info[i][j] is not None:
                    am_good += 1
            print('am_good', am_good)
            if am_good < bound_plus and am_good >= bound_minus:
                v0 = j
                break#
            err = max(abs(am_good - bound_plus), abs(am_good - bound_minus))
            if err < best_err:
                best_err = err
                v0 = j

        if self.a is not None:
            print(real_abs_v[v0], "check that this number is in [0.45, 0.75]")
        if j == N2 - 1:
            print("There is no v0 with am_good in [0.43N2 ... 0.45N2]")
        return gammas[v0]
    
    def substep_check_constant(self, all_poly, omeg=0.015):
        h = np.sqrt(self.n)
        deltas = [self.get_oscillation(poly, h) for poly in all_poly]
        deltamed = np.median(deltas_i_h_nu)
        #print("Deltamed", deltamed)
        return deltamed < omeg

    def step_approximate_a(self, gamma):
        """Approximate vector a.
        
        Params:
            gamma   --  vector with typical |v_gamma| < 3/4
        """

        n = self.n
        w = np.zeros(n)  # ws[k] will approximate a[k]*sqrt(n) / |v_gamma|

        poly0 = self.fit_polynomial(gamma)
        max_lambda = -1
        for i in range(self.n):
            ei = np.zeros(self.n)
            ei[i] = 1
            poly_ei = self.fit_polynomial(ei)
            lambda_i = embed_polynomials_l2(poly0, poly_ei, l=self.l)
            if abs(lambda_i) > max_lambda:
                max_lambda = abs(lambda_i)
                max_idx = i

        if self.a is not None:
            self.sign = 1 if self.a[max_idx] >= 0 else -1

        for i in range(n):
            if i == max_idx:
                w[i] = max_lambda
            else:
                fi = np.zeros(n)
                fi[max_idx] = 0.9
                fi[i] = 0.1
                poly_fi = self.fit_polynomial(fi)
                lambda_fi = embed_polynomials_l2(poly0, poly_fi, l=self.l)
                w[i] = 10*abs(lambda_fi) - 9*max_lambda

        newa = w / np.linalg.norm(w)
        if self.a is not None:
            a_compare = self.a * self.sign
            print("Approximation error of a, linf-norm:", max(abs(a_compare - newa)))
            print(newa)
            print(a_compare)
            print("Approximation error of a, l2-norm:", np.linalg.norm(a_compare - newa))
        return newa

    def step_approximate_phi(self, newa):
        ts = np.linspace(-1, 1, 2 * self.N3 + 1, endpoint = True)
        values_phi = np.array([self.f_eps(t * newa) for t in ts])
        poly_phi = polynomial.Polynomial.fit(ts, values_phi, deg=self.M1)
        ts1 = np.linspace(-1, 1, 500)
        values_phi1 = polynomial.polyval(ts1, poly_phi.coef)
        if self.phi is not None:
            values_phi_real = np.array([self.phi(t * self.sign) for t in ts1])
        plt.plot(ts1, values_phi_real, linewidth=5)
        plt.plot(ts1, values_phi1, linewidth=5)
        plt.show()
        #plt.plot(ts1, values_phi_real- values_phi1, linewidth=5)
        #plt.show()
        if self.phi is not None:
            print("Approximation error of phi in C:", max(np.abs(values_phi1 - values_phi_real)))
        #print("Omega1", omega1())

    def analyze_embedding(self, gamma1, gamma2):
        poly1 = self.fit_polynomial(gamma1)
        poly2 = self.fit_polynomial(gamma2)

        true_lambda = np.dot(self.a, gamma2) / np.dot(self.a, gamma1)
        ts = np.linspace(-1, 1, 500)
        fig, axs = plt.subplots(2, 2)

        def get_range(values):
            min_val = np.min(values)
            max_val = np.max(values)
            size = max_val - min_val
            return min_val - 0.3*size, max_val + 0.3*size

        ax0 = axs[0,0]
        ax0.set_title('poly1, lambda = u2/u1 = {:.6f}'.format(true_lambda))
        poly1_values = [poly1(t) for t in ts]
        min_y1, max_y1 = get_range(poly1_values)
        ax0.plot(ts, poly1_values, color='red', label='poly_1')
        ax0.plot(ts, [min(max_y1, max(min_y1, poly2(t/true_lambda))) for t in ts], color='blue', label='poly_2')
        ax0.plot(ts, [self.phi(t * np.dot(self.a, gamma1)) for t in ts], linewidth=4, alpha=0.2, color='green', label='phi')
        ax0.legend()

        ax1 = axs[0,1]
        ax1.set_title('poly2')
        poly2_values = [poly2(t) for t in ts]
        min_y2, max_y2 = get_range(poly2_values)
        ax1.plot(ts, [min(max_y2, max(min_y2, poly1(t*true_lambda))) for t in ts], color='red', label='poly_1')
        ax1.plot(ts, poly2_values, color='blue', label='poly_2')
        ax1.plot(ts, [self.phi(t * np.dot(self.a, gamma2)) for t in ts], linewidth=4, alpha=0.2, color='green', label='phi')
        ax1.legend()

        ax2 = axs[1,0]
        phi_values = [self.phi(t) for t in ts]
        ax2.plot(ts, phi_values, color='green')
        min_y, max_y = get_range(phi_values)

        u1 = -np.dot(self.a, gamma1)
        ax2.axvline(x=u1, color='red')
        ax2.axvline(x=-u1, color='red')

        u2 = -np.dot(self.a, gamma2)
        ax2.axvline(x=u2, color='blue')
        ax2.axvline(x=-u2, color='blue')

        ax2.set_title('u1={:.6f}, u2={:.6f}'.format(u1,u2))
        ax2.plot(ts, [min(max_y, max(min_y, poly1(t/u1))) for t in ts], linestyle='dotted', color='red')
        ax2.plot(ts, [min(max_y, max(min_y, poly2(t/u2))) for t in ts], linestyle='dotted', color='blue')

        # est embedding
        est_lambda = embed_polynomials_l2(poly1, poly2)
        ax3 = axs[1,1]
        ax3.set_title('est embedding, lambda = {:.6f}'.format(est_lambda))
        ax3.plot(ts, poly1_values, color='red')
        ax3.plot(ts, [min(max_y, max(min_y, poly2(t/est_lambda))) for t in ts], color='blue')
        ax3.plot(ts, [self.phi(t * np.dot(self.a, gamma1)) for t in ts], linewidth=4, alpha=0.2, color='green')

        plt.show()


def get_alpha(phi_deriv, n):
    """Integrate phi_deriv(<a,x>)^2 dx over probability measure on the sphere S^{n-1}"""
    dens = lambda t: (1-t**2)**((n - 3)/2)
    func = lambda t: phi_deriv(t)**2 * dens(t)
    return integrate.quad(func, -1, 1)[0] / integrate.quad(dens, -1, 1)[0]


def get_test_func(deg, trigpcos, trigpsin):
    """Returns function x**deg * T(x).

    T is a trigonometric polynomial:
    T(x) = a_0 + sum_{k=1}^K a_k*cos(kx) + b_k*sin(kx)

    trigpcos = [a_0, a_1, a_2, ..., a_K]
    trigpsin =      [b_1, b_2, ..., b_K]
    """
    def f(x):
        res = 0 * x  # works for vector and scalar x
        res += trigpcos[0]
        K = len(trigpsin)
        # k=1: trigpcos[1], trigpsin[0], sin(x), cos(x);  1 <= k <= K
        for k in range(1, K + 1):
            res += trigpcos[k] * np.cos(k*x)
            res += trigpsin[k-1] * np.sin(k*x)
        return (x**deg) * res

    return f


def get_test_func_deriv(deg, trigpcos, trigpsin):
    """Derivative of function from get_test_func."""
    # f = x^deg (a_0 + sum_1^K a_k cos(kx) + b_k sin(kx))
    # f' = deg x^{deg-1} * (a_0 + sum_1^K a_k cos(kx) + b_k sin(kx)) +
    #       + x^deg * sum_1^K (-a_k*ksin(kx) + b_k*kcos(kx)) = 
    #    = x^{deg-1} * (a_0*deg + sum_1^K a_k{deg*cos(kx)-kx*sin(kx)} + b_k{deg*sin(kx)+kx*cos(kx)}
    def f_deriv(x):
        res = 0 * x
        res += trigpcos[0] * deg
        K = len(trigpsin)
        for k in range(1, K+1):
            res += trigpcos[k] * (deg*np.cos(k*x) - k*x*np.sin(k*x))
            res += trigpsin[k-1] * (deg*np.sin(k*x) + k*x*np.cos(k*x))
        return (x**(deg-1)) * res

    return f_deriv


def test_example():
    n = 50
    seed = int(time() % 10000)
    print('seed:', seed)
    eps = 1e-20#0.000001#

    a = np.array([random.gauss(0, 1) for _ in range(n)])
    a = a / np.linalg.norm(a)
    
    N1 = 200
    M = 9
    M1 = 30
    N2 = 25
    N3 = 200
    K = 5
    trigparams = np.array([random.gauss(0, 1) for _ in range(2*K + 1)])
    trigparams /= np.linalg.norm(trigparams)
    trigpcos = trigparams[np.arange(0, 2*K + 1, 2)]#a0, a1, a2, ... aK
    trigpcos[0] /= np.sqrt(2)
    trigpsin = trigparams[np.arange(1, 2*K + 1, 2)]#b1, b2, ... bK
    l = 0.4
    deg = -1#if deg == -1, test trigonometric, else test x**(2deg)


    def phi(x, deg = -1):
        if deg == -1:
            return np.sum(trigpsin * np.sin(np.arange(1, K + 1) * np.pi * x)) + np.sum(trigpcos * np.cos(np.arange(0, K + 1) * np.pi * x))
        return (x*x)**deg

    def f(x):
        return phi(np.dot(a, x), deg=deg)

    def f_eps(x):
        return f(x) + eps * (2 * random.random() - 1)

    solver = RidgeSolver(n=n, f_eps=f_eps, M=M, M1=M1, N1=N1, N2=N2, N3=N3, a=a, phi= lambda x: phi(x, deg=deg))
    solver.solve()


#multiplies argument by sqrt(n) and calculates polynom with coefficients = coeff (highest power first)
def polynom_normed(coeff, x):
    x *= sqrt(n)
    res = 0
    power = 1
    for i in range(len(coeff)-1, -1, -1):
        res += coeff[i] *power
        power *= x
    return res

def quality1step(ind):
    tmax = min(1/real_abs_v[ind], 10)
    p = tmax * np.arange(-20, 20)/20.
    y = np.zeros(40)
    yy = np.zeros(40)
    v = gammas[ind]
    vgamma = sum(a * v) * np.sqrt(n)
    coeff = fit_polynomial(v)
    for i in range(40):
        #y[i] = f_changed(p[i] * v, gp, gq)
        #yy[i] = polynom_normed(coeff, p[i]/np.sqrt(n))
        y[i] = phi(vgamma * p[i], gp, gq, deg=-1)
        yy[i] = polynom_normed(coeff, p[i])
    return max(abs(yy - y))


def omega1():
    res = -100
    index = -1
    for i in range(N2):
        nqual = quality1step(i)
        if nqual > res:
            res = nqual
            index = i 
    return (res, index)


if __name__ == "__main__":
    test_example()
