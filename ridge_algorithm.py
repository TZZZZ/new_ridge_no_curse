import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt
from time import time
from numpy.polynomial import polynomial
from scipy import integrate


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
        K2 = len(trigpsin)
        # k=1: trigpcos[1], trigpsin[0], sin(x), cos(x);  1 <= k <= K
        for k in range(1, K2 + 1):
            res += trigpcos[k] * np.cos(k * x)
            res += trigpsin[k - 1] * np.sin(k * x)
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
        K2 = len(trigpsin)
        for k in range(1, K2 + 1):
            res += trigpcos[k] * (deg * np.cos(k * x) - k * x * np.sin(k * x))
            res += trigpsin[k - 1] * (deg * np.sin(k * x) + k * x * np.cos(k * x))
        return (x**(deg - 1)) * res

    return f_deriv



# Utility functions

def get_random_unit_vector(n):
    v = np.array([random.gauss(0, 1) for _ in range(n)])
    return v / np.linalg.norm(v)


def embed_polynomials_l2(p1, p2, l=1.0, calc_score=False):
    """Find lambda for inclusion p1->p2, i.e., such that p1(t) ~ p2(t/lambda), |t|<l.
    
    Params:
        p1, p2  -- instances of polynomial.Polynomial class
        l       -- defines embedding segment [-l,l]
    """

    # we minimize S(mu) = int_{-l}^l |p1(t)-p2(mu t)|^2 dt
    # S(mu) is a polynomial in mu: sum_{i,j} (a_i - b_i*mu^i)(a_j - b_j*mu^j)int_{l}^l t^{i+j}
    a, b, d = p1.coef, p2.coef, p1.degree()
    assert p2.degree() == d
    s_coef = np.zeros(2 * d + 1)
    for i in range(d + 1):
        for j in range(d + 1):
            # (a_i - b_i*mu^i)(a_j - b_j*mu^j)int_{l}^l t^{i+j}
            # int t^k = t^{k+1)|_{-l}^l = 0 if k+1 is even else 2*l^{k+1}/(k+1)
            if (i + j + 1) % 2 == 0:
                continue
            int_t = 2 * l**(i + j + 1) / (i + j + 1)
            s_coef[0] += int_t * a[i] * a[j]
            s_coef[i] -= int_t * b[i] * a[j]
            s_coef[j] -= int_t * a[i] * b[j]
            s_coef[i+j] += int_t * b[i] * b[j]

    s_poly = polynomial.Polynomial(s_coef)
    min_value, min_mu = minimize_polynomial(s_poly, -1, 1)
    if not calc_score:
        return 1 / min_mu
    if min_value == 0:
        score = 10000
    else:
        score = s_poly(0) / min_value
    return {'lambda': 1 / min_mu, 'score': score}


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
    Technical params for internal use: 
        M       --  degree of fitting polynomials
        M1      --  degree of fitting polynomial for final reconstruction
        N1      --  defines the number of points for fitting polynomials 
        N2      --  the number of generated gammas 
        N3      --  defines the number of points for fitting final polynomial
        l       --  defines the segment for embedding
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

    def fit_polynomial(self, gamma):
        """Fit polynomial to a function phi(t_k <a,gamma>), |t_k|<=1."""
        ts = np.linspace(-1, 1, 2 * self.N1 + 1)
        ys = [self.f_eps(t * gamma) for t in ts]
        return polynomial.Polynomial.fit(ts, ys, deg=self.M)

    def check_fitting(self, gamma, poly):
        v_gamma = np.dot(gamma, self.a)
        ts = np.linspace(-self.l * np.sqrt(50), self.l * np.sqrt(50), 10)
        return max(abs(poly(t) - self.phi(v_gamma * t)) for t in ts)

    def get_oscillation(self, poly, h=1):
        """return max_{|t|<=h} |poly(t)-poly(0)|"""
        extr = extremize_polynomial(poly - poly(0), -h, h)
        return max(abs(extr['max']), abs(extr['min']))

    def solve(self):
        typical_gamma = self.step_get_typical_gamma(log_info = False, check_constant = False)
        if len(typical_gamma) == 1:
            print("Function is almost constant:", self.f_eps(np.zeros(self.n)))
        else:
            newa = self.step_approximate_a(typical_gamma, log_info = False)
            newphr = self.step_approximate_phi(newa)

    def step_get_typical_gamma_for_test(self):
        """Find gamma with 0.45<|v_gamma|<0.75 using True values of a."""
        N2 = self.N2

        # Generate N2 gammas
        gammas = [get_random_unit_vector(self.n) for _ in range(N2)]
        self.gammas = gammas
        if self.a is not None:
            real_v = sqrt(self.n) * np.array([np.dot(self.a, gamma) for gamma in gammas])
            real_abs_v = np.abs(real_v)
        pind = np.where(abs(real_abs_v - 0.5) < 0.15)[0][0]
        print("gammas[ind] is typical for ind =", pind, "and n**0.5 * |<a, gammas[ind]>| = ", real_abs_v[pind])
        return gammas[pind]
    
    def step_get_typical_gamma(self, log_info = False, check_constant = False):
        """Find gamma with 0.45<|v_gamma|<0.75."""

        N2 = self.N2

        # Generate N2 gammas
        gammas = [get_random_unit_vector(self.n) for _ in range(N2)]

        if self.a is not None:
            real_v = sqrt(self.n) * np.array([np.dot(self.a, gamma) for gamma in gammas])
            real_abs_v = np.abs(real_v)

        # Second part of algorithm
        all_poly = [self.fit_polynomial(gamma) for gamma in gammas]  # polynom coefficients for all gammas
        
        if check_constant:
            is_const = self.substep_check_constant(all_poly)
            if is_const:
                return [None]
        logging.warning('start embeddings ...')
        embed_info = {i: {j: None for j in range(N2)} for i in range(N2)}  # if phi_i -> phi_j, embed_info[i][j] = corresponding lambda
        bound_minus = N2 * 0.43
        bound_plus = N2 * 0.45
        v0 = -1
        best_err = 1000
        if log_info:
            self.emb_lambdas = []
            self.emb_scores = []
            self.emb_good = []
        for j in range(N2):
            logging.warning('embed j=%d', j)
            am_good = 0
            for i in range(N2):
                vall = embed_polynomials_l2(all_poly[i], all_poly[j], l=self.l, calc_score = True)
                if log_info:
                    print(vall['lambda'], vall['score'])
                    self.emb_lambdas.append(vall['lambda'])
                    self.emb_scores.append(vall['score'])
                if abs(vall['lambda']) > (1 + 1e-5) and abs(vall['score']) > (1 + 1e-5):
                    embed_info[i][j] = vall['lambda']
                #if abs(vall) != 1:
                #    embed_info[i][j] = vall
                if embed_info[i][j] is not None:
                    am_good += 1
            if log_info:
                print('am_good', am_good)
                self.emb_good.append(am_good)
            if am_good < bound_plus and am_good >= bound_minus:
                v0 = j
                break
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
        deltamed = np.median(deltas)
        #print("Median of deltas", deltamed)
        return deltamed < omeg


    def step_approximate_a(self, gamma, log_info = False):
        """Approximate vector a.
        
        Params:
            gamma   --  vector with typical |v_gamma| < 3/4
        """
        vgamma = self.a.dot(gamma)
        n = self.n
        w = np.zeros(n)  # ws[k] will approximate (a[k] * 2 / |v_gamma| / 5) + 2
        poly0 = self.fit_polynomial(gamma / 2)
        max_lambda = -1
        if log_info:
            self.lambdas04 = []
            self.lambdas08 = []
            self.lambdas_true = []
            self.osc = []
            self.appr = []
            self.scores = []
        for i in range(self.n):
            ei = np.zeros(self.n)
            ei[i] = 1 / 5 / self.n**0.5
            ei += gamma.copy()
            poly_ei = self.fit_polynomial(ei)
            lambda_i = embed_polynomials_l2(poly0, poly_ei, l=self.l, calc_score=True)
            scc = lambda_i["score"]
            if abs(scc) < 1.001:
                print("Warning: bad score", scc, "when embed poly0 -> ei for i =", i)
            lambda_i = lambda_i["lambda"]
            if log_info:
                self.scores.append(scc)                
                lambda_i2 = embed_polynomials_l2(poly0, poly_ei, l=0.4, calc_score=True)
                print("score", lambda_i2["score"])
                self.scores.append(lambda_i2["score"])
                lambda_i2 = lambda_i2["lambda"]
                lambda_i = embed_polynomials_l2(poly0, poly_ei, l=0.8)
                self.lambdas04.append(lambda_i)
                self.lambdas08.append(lambda_i2)
                self.lambdas_true.append((self.a[i] * 2 / vgamma / 5 / self.n**0.5) + 2)
                print(lambda_i, lambda_i2, (self.a[i] * 2 / vgamma / 5 / self.n**0.5) + 2)
                oscc = self.get_oscillation(poly0)
                self.osc.append(oscc)
                ts1 = np.linspace(-1, 1, 300)
                values_phi1 = polynomial.polyval(ts1, poly_ei.coef)
                values_phi_real = np.array([self.phi(t * self.a.dot(ei)) for t in ts1])
                appr = abs(values_phi_real - values_phi1).mean()
                self.appr.append(appr)
            w[i] = abs(lambda_i) - 2
        newa = w / np.linalg.norm(w)
        if self.a is not None:
            self.sign = np.sign(self.a[0] / newa[0])
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
        if self.phi is not None:
            print("Approximation error of phi in C:", max(np.abs(values_phi1 - values_phi_real)))

    def analyze_embedding(self, gamma1, gamma2, poly1=None, poly2=None):
        if poly1 is None:
            poly1 = self.fit_polynomial(gamma1)
        if poly2 is None:
            poly2 = self.fit_polynomial(gamma2)

        def get_range(values):
            min_val = np.min(values)
            max_val = np.max(values)
            size = max_val - min_val
            return min_val - size, max_val + size

        ts = np.linspace(-1, 1, 500)
        fig, axs = plt.subplots(3, 2)
        u1 = np.dot(self.a, gamma1)
        u2 = np.dot(self.a, gamma2)
        u = {1: u1, 2: u2}
        color = {1: 'blue', 2: 'red'}
        poly = {1: poly1, 2: poly2}

        for idx, (src, dst) in enumerate([(1, 2), (2, 1)]):
            ax = axs[0, idx]
            ax.set_title(r'$poly_{},\;\lambda = u_{}/u_{} = {:.6f}$'.format(src, dst, src, u[dst]/u[src]))
            src_values = [poly[src](t) for t in ts]
            min_y, max_y = get_range(src_values)
            #ax.axis('off')
            ax.plot(ts, src_values, color=color[src], label='poly_{}'.format(src))
            ax.plot(ts, [min(max_y, max(min_y, poly[dst](t * u[src]/u[dst]))) for t in ts], color=color[dst], label='poly_{}'.format(dst))
            ax.plot(ts, [self.phi(t * u[src]) for t in ts], linewidth=4, alpha=0.2, color='green', label='phi')
            ax.legend()

            # embedding
            est = embed_polynomials_l2(poly[src], poly[dst], calc_score=True)
            ax2 = axs[1,idx]
            #ax2.axis('off')
            
            ax2.set_title(r'embed $poly_{}\to poly_{}, \lambda = {:.6f}$, score={:.6f}'.format(
                src, dst, est['lambda'], est['score']
            ))
            ax2.plot(ts, src_values, color=color[src])
            ax2.plot(ts, [min(max_y, max(min_y, poly[dst](t/est['lambda']))) for t in ts], color=color[dst])

        ax2 = fig.add_subplot(3, 1, 3)
        phi_values = [self.phi(t) for t in ts]
        ax2.plot(ts, phi_values, color='green')
        min_y, max_y = get_range(phi_values)

        ax2.axvline(x=u1, color='red')
        ax2.axvline(x=-u1, color='red')

        ax2.axvline(x=u2, color='blue')
        ax2.axvline(x=-u2, color='blue')

        ax2.set_title('u1={:.6f}, u2={:.6f}'.format(u1, u2))
        ax2.plot(ts, [min(max_y, max(min_y, poly1(t/u1))) for t in ts], linestyle='dotted', color='red')
        ax2.plot(ts, [min(max_y, max(min_y, poly2(t/u2))) for t in ts], linestyle='dotted', color='blue')

        fig.subplots_adjust()
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
        K2 = len(trigpsin)
        # k=1: trigpcos[1], trigpsin[0], sin(x), cos(x);  1 <= k <= K
        for k in range(1, K2 + 1):
            res += trigpcos[k] * np.cos(k * x)
            res += trigpsin[k - 1] * np.sin(k * x)
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
        K2 = len(trigpsin)
        for k in range(1, K2 + 1):
            res += trigpcos[k] * (deg * np.cos(k * x) - k * x * np.sin(k * x))
            res += trigpsin[k - 1] * (deg * np.sin(k * x) + k * x * np.cos(k * x))
        return (x**(deg - 1)) * res

    return f_deriv


def test_example():
    n = 50
    seed = int(time() % 10000)
    print('seed:', seed)
    eps = 1e-20#1e-10#0.000001#
    a = get_random_unit_vector(n)
    N1 = 200
    M = 12
    M1 = 30
    N2 = 25
    N3 = 200
    K2 = 7
    trigparams = get_random_unit_vector(2 * K2 + 1)
    trigpcos = trigparams[np.arange(0, 2 * K2 + 1, 2)] # a0, a1, a2, ... aK
    trigpcos[0] /= np.sqrt(2)
    trigpsin = trigparams[np.arange(1, 2 * K2 + 1, 2)] # b1, b2, ... bK
    l = 1#0.4
    deg = 8
    phi = get_test_func(deg, trigpcos, trigpsin)
    phi_deriv = get_test_func_deriv(deg, trigpcos, trigpsin)
    print("Alpha: ", get_alpha(phi_deriv, n))
    def f(x):
        return phi(np.dot(a, x))

    def f_eps(x):
        return f(x) + eps * (2 * random.random() - 1)

    solver = RidgeSolver(n=n, f_eps=f_eps, M=M, M1=M1, N1=N1, N2=N2, N3=N3, a=a, phi=phi)
    solver.solve()
    return solver


if __name__ == "__main__":
    solver = test_example()
