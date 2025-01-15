"""
Discrete programming example:
QuantEcon website:  https://quantecon.org/quantecon-py/

Originally developed by Thomas J. Sargent and John Stachurski
Please see lecture on Discrete State Dynamic Programming: 
https://python-advanced.quantecon.org/_downloads/pdf/discrete_dp.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
import scipy.sparse as sparse
from quantecon import compute_fixed_point
from quantecon.markov import DiscreteDP


class SimpleOG:

    def __init__(self, B=10, M=5, α=0.5, β=0.9):
        """
        Set up R, Q and β, the three elements that define an instance of
        the DiscreteDP class.
        """

        self.B, self.M, self.α, self.β  = B, M, α, β
        self.n = B + M + 1
        self.m = M + 1

        self.R = np.empty((self.n, self.m))
        self.Q = np.zeros((self.n, self.m, self.n))

        self.populate_Q()
        self.populate_R()


    def u(self, c):
        return c**self.α


    def populate_R(self):
        """
        Populate the R matrix, with R[s, a] = -np.inf for infeasible
        state-action pairs.
        """
        for s in range(self.n):
            for a in range(self.m):
                self.R[s, a] = self.u(s - a) if a <= s else -np.inf


    def populate_Q(self):
        """
        Populate the Q matrix by setting

            Q[s, a, s'] = 1 / (1 + B) if a <= s' <= a + B

        and zero otherwise.
        """

        for a in range(self.m):
            self.Q[:, a, a:(a + self.B + 1)] = 1.0 / (self.B + 1)


if __name__ == '__main__':
        
    g = SimpleOG()  # Use default parameters
    ddp = qe.markov.DiscreteDP(g.R, g.Q, g.β)
    results = ddp.solve(method='policy_iteration')
    dir(results)
    results.v
    results.sigma
    results.max_iter
    results.num_iter
    results.mc.stationary_distributions
    
    ddp = qe.markov.DiscreteDP(g.R, g.Q, 0.99)  # Increase β to 0.99
    results = ddp.solve(method='policy_iteration')
    results.mc.stationary_distributions
    
    B, M, α, β = 10, 5, 0.5, 0.9
    n = B + M + 1
    m = M + 1
    
    def u(c):
        return c**α
    
    s_indices = []
    a_indices = []
    Q = []
    R = []
    b = 1.0 / (B + 1)
    
    for s in range(n):
        for a in range(min(M, s) + 1):  # All feasible a at this s
            s_indices.append(s)
            a_indices.append(a)
            q = np.zeros(n)
            q[a:(a + B + 1)] = b        # b on these values, otherwise 0
            Q.append(q)
            R.append(u(s - a))
    
    ddp = qe.markov.DiscreteDP(R, Q, β, s_indices, a_indices)
    
    α = 0.65
    f = lambda k: k**α
    u = np.log
    β = 0.95
    
    grid_max = 2
    grid_size = 500
    grid = np.linspace(1e-6, grid_max, grid_size)
    
    # Consumption matrix, with nonpositive consumption included
    C = f(grid).reshape(grid_size, 1) - grid.reshape(1, grid_size)
    
    # State-action indices
    s_indices, a_indices = np.where(C > 0)
    
    # Number of state-action pairs
    L = len(s_indices)
    
    print(L)
    print(s_indices)
    print(a_indices)
    
    R = u(C[s_indices, a_indices])
    
    Q = sparse.lil_matrix((L, grid_size))
    Q[np.arange(L), a_indices] = 1
    
    # data = np.ones(L)
    # indptr = np.arange(L+1)
    # Q = sparse.csr_matrix((data, a_indices, indptr), shape=(L, grid_size))
    
    ddp = DiscreteDP(R, Q, β, s_indices, a_indices)
    
    res = ddp.solve(method='policy_iteration')
    v, σ, num_iter = res.v, res.sigma, res.num_iter
    num_iter
    
    # Optimal consumption in the discrete version
    c = f(grid) - grid[σ]
    
    # Exact solution of the continuous version
    ab = α * β
    c1 = (np.log(1 - ab) + np.log(ab) * ab / (1 - ab)) / (1 - β)
    c2 = α / (1 - ab)
    
    def v_star(k):
        return c1 + c2 * np.log(k)
    
    def c_star(k):
        return (1 - ab) * k**α
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].set_ylim(-40, -32)
    ax[0].set_xlim(grid[0], grid[-1])
    ax[1].set_xlim(grid[0], grid[-1])
    
    lb0 = 'discrete value function'
    ax[0].plot(grid, v, lw=2, alpha=0.6, label=lb0)
    
    lb0 = 'continuous value function'
    ax[0].plot(grid, v_star(grid), 'k-', lw=1.5, alpha=0.8, label=lb0)
    ax[0].legend(loc='upper left')
    
    lb1 = 'discrete optimal consumption'
    ax[1].plot(grid, c, 'b-', lw=2, alpha=0.6, label=lb1)
    
    lb1 = 'continuous optimal consumption'
    ax[1].plot(grid, c_star(grid), 'k-', lw=1.5, alpha=0.8, label=lb1)
    ax[1].legend(loc='upper left')
    plt.show()
    
    np.abs(v - v_star(grid)).max()
    np.abs(v - v_star(grid))[1:].max()
    np.abs(c - c_star(grid)).max()
    
    diff = np.diff(c)
    (diff >= 0).all()
    
    dec_ind = np.where(diff < 0)[0]
    len(dec_ind)
    
    np.abs(diff[dec_ind]).max()
    (np.diff(v) > 0).all()
    
    ddp.epsilon = 1e-4
    ddp.max_iter = 500
    res1 = ddp.solve(method='value_iteration')
    res1.num_iter
    
    np.array_equal(σ, res1.sigma)
    
    res2 = ddp.solve(method='modified_policy_iteration')
    res2.num_iter
    
    np.array_equal(σ, res2.sigma)
    
    res1 = ddp.solve(method='value_iteration')
    res2 = ddp.solve(method='policy_iteration')
    res2 = ddp.solve(method='modified_policy_iteration')
    
    w = 5 * np.log(grid) - 25  # Initial condition
    n = 35
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_ylim(-40, -20)
    ax.set_xlim(np.min(grid), np.max(grid))
    lb = 'initial condition'
    ax.plot(grid, w, color=plt.cm.jet(0), lw=2, alpha=0.6, label=lb)
    for i in range(n):
        w = ddp.bellman_operator(w)
        ax.plot(grid, w, color=plt.cm.jet(i / n), lw=2, alpha=0.6)
    lb = 'true value function'
    ax.plot(grid, v_star(grid), 'k-', lw=2, alpha=0.8, label=lb)
    ax.legend(loc='upper left')
    
    plt.show()
    w = 5 * u(grid) - 25           # Initial condition
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    true_c = c_star(grid)
    
    for i, n in enumerate((2, 4, 6)):
        ax[i].set_ylim(0, 1)
        ax[i].set_xlim(0, 2)
        ax[i].set_yticks((0, 1))
        ax[i].set_xticks((0, 2))
    
        w = 5 * u(grid) - 25       # Initial condition
        compute_fixed_point(ddp.bellman_operator, w, max_iter=n, print_skip=1)
        σ = ddp.compute_greedy(w)  # Policy indices
        c_policy = f(grid) - grid[σ]
    
        ax[i].plot(grid, c_policy, 'b-', lw=2, alpha=0.8,
                   label='approximate optimal consumption policy')
        ax[i].plot(grid, true_c, 'k-', lw=2, alpha=0.8,
                   label='true optimal consumption policy')
        ax[i].legend(loc='upper left')
        ax[i].set_title(f'{n} value function iterations')
    plt.show()
    
    discount_factors = (0.9, 0.94, 0.98)
    k_init = 0.1
    
    # Search for the index corresponding to k_init
    k_init_ind = np.searchsorted(grid, k_init)
    
    sample_size = 25
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlabel("time")
    ax.set_ylabel("capital")
    ax.set_ylim(0.10, 0.30)
    
    # Create a new instance, not to modify the one used above
    ddp0 = DiscreteDP(R, Q, β, s_indices, a_indices)
    
    for beta in discount_factors:
        ddp0.beta = beta
        res0 = ddp0.solve()
        k_path_ind = res0.mc.simulate(init=k_init_ind, ts_length=sample_size)
        k_path = grid[k_path_ind]
        ax.plot(k_path, 'o-', lw=2, alpha=0.75, label=f'$\\beta = {beta}$')
    
    ax.legend(loc='lower right')
    plt.show()
