import numpy as np
from scipy.stats import qmc


# ------------------------------------------------------------------
# 1) Orthogonal projection to hyperplane sum(x) = t
# ------------------------------------------------------------------
def project_to_hyperplane_sum_t(X, t=1.0):
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    s = X.sum(axis=1, keepdims=True)
    return X + (t - s) / d


# ------------------------------------------------------------------
# 2) Equi-distance (maximin) farthest-point sampling
# ------------------------------------------------------------------
def farthest_point_sampling_equidistant(
    points,
    N_target,
    seed=0,
    start="closest_to_ones",
):
    """
    Greedy maximin farthest-point sampling.

    start:
      - "random"
      - "center"            (closest to ones-vector)
      - "farthest_pair"
      - "closest_to_ones"   (alias of center, default)
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(points, dtype=float)
    M, d = P.shape

    if N_target >= M:
        return P.copy()

    # ---- initialization ----
    if start in ("center", "closest_to_ones"):
        ones = np.ones(d)
        idx0 = np.argmin(np.linalg.norm(P - ones, axis=1))
        selected = [idx0]

    elif start == "random":
        idx0 = rng.integers(M)
        selected = [idx0]

    elif start == "farthest_pair":
        i0 = rng.integers(M)
        i1 = np.argmax(np.linalg.norm(P - P[i0], axis=1))
        i2 = np.argmax(np.linalg.norm(P - P[i1], axis=1))
        selected = [i1, i2]

    else:
        raise ValueError("Invalid start mode")

    # ---- distance bookkeeping ----
    dists = np.full(M, np.inf)
    for idx in selected:
        dists = np.minimum(dists, np.linalg.norm(P - P[idx], axis=1))
    dists[selected] = -np.inf

    # ---- greedy maximin FPS ----
    while len(selected) < N_target:
        idx = int(np.argmax(dists))
        selected.append(idx)

        new_d = np.linalg.norm(P - P[idx], axis=1)
        dists = np.minimum(dists, new_d)
        dists[selected] = -np.inf

    return P[selected]


# ------------------------------------------------------------------
# 3) Generate interior points along cube edges
# ------------------------------------------------------------------
def generate_points_per_edge(d, k_per_edge):
    if k_per_edge <= 0:
        return np.zeros((0, d), dtype=float)

    ts = np.linspace(0.0, 1.0, k_per_edge + 2)[1:-1]
    pts = []

    for free_dim in range(d):
        fixed_dims = [j for j in range(d) if j != free_dim]
        for mask in range(2 ** (d - 1)):
            base = np.zeros(d)
            for bit, j in enumerate(fixed_dims):
                base[j] = 1.0 if ((mask >> bit) & 1) else 0.0
            for t in ts:
                p = base.copy()
                p[free_dim] = t
                pts.append(p)

    return np.asarray(pts)


# ------------------------------------------------------------------
# 4) Sobol → projection → equi-distance FPS (FULL PIPELINE)
# ------------------------------------------------------------------
def sobol_fps_sampler(
    d,
    N_target,
    seed=0,
    n_candidates=500000,
    k_per_edge=50,
):
    """
    Parameter-free equi-distance sampler.

    Steps:
      1. Sobol low-discrepancy sampling in [0,1]^d
      2. Optional edge-point enrichment
      3. Orthogonal projection to hyperplane sum(x)=d
      4. Greedy maximin farthest-point sampling
    """
    # k_per_edge = 2 for d=15 used for testing

    # Sobol requires powers of two
    m = int(np.ceil(np.log2(n_candidates)))

    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    Z = sampler.random_base2(m=m)[:n_candidates]

    # Edge enrichment
    E = generate_points_per_edge(d, k_per_edge)

    print(f"Generated {len(E)} edge points for enrichment (k_per_edge={k_per_edge})")

    # Candidate pool
    C = np.vstack([Z, E]) if len(E) > 0 else Z

    # Project to hyperplane sum(x) = d
    X = project_to_hyperplane_sum_t(C, t=d)

    # Equi-distance FPS
    samples = farthest_point_sampling_equidistant(
        X,
        N_target,
        seed=seed,
        start="farthest_pair",
    )

    return samples