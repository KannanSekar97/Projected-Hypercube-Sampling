#!/usr/bin/env python3
"""
Hypervolume QMC reporter (approximate)

Process:
  - Load a point set (rows = points, cols = objectives).
  - Optionally compress with regular-grid quantization: map points into
    `compress_bins` per-dimension bins and keep one representative per occupied
    cell (componentwise minimum of points in the cell). Optionally cap size
    with `compress_target`.
  - Compute `hv_compressed` (HV on compressed set) as a coarse/conservative check.
  - Estimate hypervolume via QMC (Sobol or PRNG) using quantile-based bitset
    acceleration plus exact candidate verification to count dominated samples.
  - Report hv_qmc ± se and timings.

Assumptions:
  - Objectives are compared componentwise after min–max scaling; the code
    assumes the HV semantics used by the caller (min vs max) when interpreting
    whether componentwise minima are conservative.
  - Representative = componentwise minimum is meaningful for the intended
    dominance direction (change to maxima if your semantics differ).
  - Input fits in memory for bitset construction (bitset size ~ Np/64 words).

Limitations & cautions:
  - Quantization introduces approximation error; finer `compress_bins` → less
    error but more representatives. In high M, many occupied bins may arise.
  - `compress_target` truncation (keeps lowest coordinate-sum reps) biases the
    compressed set; choose it with care.
  - QMC result is approximate (se is reported). Convergence depends on
    N_samples, S, jitter, and bitset binning K.
  - Performance sensitive to Np, M, K and available memory; enable NUMBA/Sobol
    when available for speed/repeatability (`rng_seed` controls PRNG).

# set threads (run once in the shell)
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# run script with tuned params
python hv_qmc.py \
  --file IDTLZ2_M8.npy \
  --ref 1.1 \
  --K 512 \
  --n-samples 10000000 \
  --S 3 \
  --jitter 0.3 \
  --batch 100000 \
  --compress-bins 512 \
  --compress-target 30000 \
  --out hv_summary.json

"""

from __future__ import annotations
import argparse
import json
import math
import os
import time
from typing import Optional, Dict, Any

import numpy as np

# silence numpy warnings
import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda *args, **kwargs: None

# numba optional but recommended
try:
    import numba
    from numba import njit, prange
    NUMBA = True
except Exception:
    NUMBA = False

# Sobol sampler optional
try:
    from scipy.stats import qmc
    HAVE_SOBOL = True
except Exception:
    HAVE_SOBOL = False


# -------------------------
# I/O helpers
# -------------------------
def load_npz_points(fname: str):
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    d = np.load(fname, allow_pickle=False)
    if "points" not in d:
        raise KeyError(f"{fname} does not contain 'points' array")
    pts = d["points"]
    metadata = None
    if "metadata" in d:
        try:
            metadata = json.loads(d["metadata"].tolist())
        except Exception:
            metadata = d["metadata"].tolist()
    return pts, metadata

# -------------------------
# Conservative quantization compression
# -------------------------
def compress_by_quantization(points: np.ndarray, bins_per_dim: int = 128, target_k: Optional[int] = None) -> np.ndarray:
    pts = np.ascontiguousarray(points, dtype=np.float64)
    N, M = pts.shape
    if N == 0:
        return pts.copy()
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = np.where(maxs > mins, maxs - mins, 1.0)
    scaled = ((pts - mins) / ranges) * (bins_per_dim - 1)
    keys = np.floor(scaled).astype(np.int64)
    dtype = np.dtype([("f%d" % i, np.int64) for i in range(M)])
    structured = keys.view(dtype)
    uniq_keys, inv_idx = np.unique(structured, return_inverse=True)
    comp = np.full((uniq_keys.shape[0], M), np.inf, dtype=np.float64)
    for i in range(N):
        k = inv_idx[i]
        comp[k] = np.minimum(comp[k], pts[i])
    comp = comp[np.isfinite(comp).all(axis=1)]
    if target_k is not None and comp.shape[0] > target_k:
        norms = comp.sum(axis=1)
        order = np.argsort(norms)
        comp = comp[order[:target_k]]
    return comp

# -------------------------
# Bitset helpers (same as your approach)
# -------------------------
def build_quantile_edges(PF: np.ndarray, K: int, jitter: float = 0.0) -> np.ndarray:
    PF = np.ascontiguousarray(PF)
    Np, M = PF.shape
    edges2d = np.empty((M, K + 1), dtype=np.float64)
    for a in range(M):
        q = np.linspace(0.0, 1.0, K + 1)
        edges = np.quantile(PF[:, a], q) if hasattr(np, "quantile") else np.percentile(PF[:, a], q * 100)
        if jitter > 0.0:
            rng = np.random.default_rng()
            for t in range(1, K):
                left = edges[t - 1]
                right = edges[t + 1]
                width = max(right - left, 1e-12)
                shift = (rng.random() - 0.5) * jitter * width
                edges[t] = edges[t] + shift
        edges = np.maximum.accumulate(edges)
        edges2d[a, :] = edges
    return edges2d

def build_bitsets_from_edges(PF: np.ndarray, edges2d: np.ndarray) -> np.ndarray:
    PF = np.ascontiguousarray(PF)
    Np, M = PF.shape
    K = edges2d.shape[1] - 1
    words = (Np + 63) // 64
    bitsets = np.zeros((M, K, words), dtype=np.uint64)
    for a in range(M):
        edges = edges2d[a]
        bins_idx = np.digitize(PF[:, a], edges[1:-1], right=False)
        for t in range(K):
            pos = np.nonzero(bins_idx == t)[0]
            if pos.size == 0:
                continue
            widx = pos // 64
            offs = pos % 64
            uw, inv = np.unique(widx, return_inverse=True)
            for ui_idx, w in enumerate(uw):
                mask_offs = offs[inv == ui_idx]
                masks = (np.uint64(1) << mask_offs.astype(np.uint64))
                mval = np.bitwise_or.reduce(masks)
                bitsets[a, t, w] = mval
        for t in range(1, K):
            np.bitwise_or(bitsets[a, t - 1], bitsets[a, t], out=bitsets[a, t])
    return bitsets

# -------------------------
# Candidate verification: numba / python fallback
# -------------------------
if NUMBA:
    @njit
    def _check_candidates_from_acc_numba(acc, PF, y, Np):
        words = acc.shape[0]
        M = PF.shape[1]
        for w in range(words):
            aw = acc[w]
            if aw == 0:
                continue
            base = w * 64
            for b in range(64):
                if (aw >> b) & np.uint64(1):
                    idx = base + b
                    if idx >= Np:
                        continue
                    ok = True
                    for j in range(M):
                        if PF[idx, j] > y[j]:
                            ok = False
                            break
                    if ok:
                        return True
        return False

    @njit
    def _find_bin_numba(edges_row, K_plus1, val):
        lo = 0
        hi = K_plus1
        while lo < hi:
            mid = (lo + hi) // 2
            if edges_row[mid] > val:
                hi = mid
            else:
                lo = mid + 1
        j = lo
        t = j - 1
        if t < 0:
            t = 0
        if t >= K_plus1 - 1:
            t = K_plus1 - 2
        return t

    @njit(parallel=True, nogil=True)
    def dominated_mask_bitset_verify_numba(Y, bitsets, edges2d, PF, Np, M, K, words):
        b = Y.shape[0]
        out = np.zeros(b, dtype=numba.boolean)
        K_plus1 = K + 1
        for i in prange(b):
            t0 = _find_bin_numba(edges2d[0], K_plus1, Y[i, 0])
            acc = np.empty(words, dtype=np.uint64)
            for w in range(words):
                acc[w] = bitsets[0, t0, w]
            empty = True
            for w in range(words):
                if acc[w] != 0:
                    empty = False
                    break
            if empty:
                out[i] = False
                continue
            nonzero_flag = True
            for a in range(1, M):
                ta = _find_bin_numba(edges2d[a], K_plus1, Y[i, a])
                nz = False
                for w in range(words):
                    acc[w] &= bitsets[a, ta, w]
                    if acc[w] != 0:
                        nz = True
                if not nz:
                    nonzero_flag = False
                    out[i] = False
                    break
            if not nonzero_flag:
                continue
            if _check_candidates_from_acc_numba(acc, PF, Y[i], Np):
                out[i] = True
            else:
                out[i] = False
        return out
else:
    def _check_candidates_from_acc(acc, PF, y, Np):
        words = acc.shape[0]
        M = PF.shape[1]
        for w in range(words):
            aw = int(acc[w])
            if aw == 0:
                continue
            base = w * 64
            b = 0
            while aw:
                if aw & 1:
                    idx = base + b
                    if idx < Np:
                        ok = True
                        for j in range(M):
                            if PF[idx, j] > y[j]:
                                ok = False
                                break
                        if ok:
                            return True
                aw >>= 1
                b += 1
        return False

    def dominated_mask_bitset_verify_numpy(Y, bitsets, edges2d, PF, Np, M, K, words):
        b = Y.shape[0]
        out = np.zeros(b, dtype=bool)
        for i in range(b):
            y = Y[i]
            acc = None
            dominated = False
            for a in range(M):
                edges = edges2d[a]
                t = np.searchsorted(edges, y[a], side='right') - 1
                if t < 0:
                    t = 0
                if t >= K:
                    t = K - 1
                bs = bitsets[a, t]
                if acc is None:
                    acc = bs.copy()
                else:
                    np.bitwise_and(acc, bs, out=acc)
                if not acc.any():
                    dominated = False
                    break
            else:
                if _check_candidates_from_acc(acc, PF, y, Np):
                    dominated = True
            out[i] = dominated
        return out

def dominated_mask_bitset_verify(Y, bitsets, edges2d, PF, Np, M, K, words):
    if NUMBA:
        return dominated_mask_bitset_verify_numba(Y, bitsets, edges2d, PF, Np, M, K, words)
    else:
        return dominated_mask_bitset_verify_numpy(Y, bitsets, edges2d, PF, Np, M, K, words)

# -------------------------
# QMC HV estimator (no exact) — defaults set to K=512 and N_samples=1e6
# -------------------------
def count_dominated_and_hv_high_accuracy(PF: np.ndarray,
                                         K: int = 512,
                                         S: int = 3,
                                         jitter: float = 0.0,
                                         ref_value: float = 1.1,
                                         N_samples: int = int(1e6),
                                         sample_batch: int = 50000,
                                         rng: Optional[np.random.Generator] = None,
                                         use_sobol: bool = True,
                                         verbose: bool = False) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()
    PF = np.ascontiguousarray(PF, dtype=np.float64)
    Np, M = PF.shape
    ref = np.full(M, float(ref_value), dtype=np.float64)
    box_vol = float(np.prod(ref.astype(np.float64)))

    sampler = None
    if use_sobol and HAVE_SOBOL:
        sampler = qmc.Sobol(d=M, scramble=True)
    else:
        sampler = None

    hv_sum = 0.0
    se_sum = 0.0
    total_samples = 0

    reuse_edges2d = None
    reuse_bitsets = None
    reuse_words = None
    if jitter == 0.0:
        reuse_edges2d = build_quantile_edges(PF, K, jitter=0.0)
        reuse_bitsets = build_bitsets_from_edges(PF, reuse_edges2d)
        reuse_words = reuse_bitsets.shape[2]

    for s in range(S):
        if reuse_edges2d is not None:
            edges2d = reuse_edges2d
            bitsets = reuse_bitsets
            words = reuse_words
        else:
            edges2d = build_quantile_edges(PF, K, jitter=jitter)
            bitsets = build_bitsets_from_edges(PF, edges2d)
            words = bitsets.shape[2]

        n_done = 0
        n_dom = 0
        while n_done < N_samples:
            b = min(sample_batch, N_samples - n_done)
            if sampler is not None:
                Y0 = sampler.random(n=b)
            else:
                Y0 = rng.random((b, M))
            Y = (Y0 * ref).astype(np.float64)
            mask = dominated_mask_bitset_verify(Y, bitsets, edges2d, PF, Np, M, K, words)
            n_dom += int(mask.sum())
            n_done += b
            total_samples += b
            if verbose:
                print(f"[s={s}] batch done {n_done}/{N_samples}, dom so far {n_dom}")
        frac = n_dom / n_done if n_done > 0 else 0.0
        hv = frac * box_vol
        se = math.sqrt(max(frac * (1 - frac) / n_done, 0.0)) * box_vol if n_done > 0 else 0.0
        hv_sum += hv
        se_sum += se ** 2

    hv_avg = hv_sum / S
    se_avg = math.sqrt(se_sum / S)
    return {
        "HV": hv_avg,
        "se": se_avg,
        "S": S,
        "K": K,
        "Np": Np,
        "M": M,
        "total_samples": total_samples,
        "n_D": n_dom
    }

# -------------------------
# Top-level: compute compressed + qmc only
# -------------------------
def compute_hv_qmc_only(points: np.ndarray,
                        reference: np.ndarray,
                        compress_bins: int = 128,
                        compress_target: Optional[int] = 3000,
                        K: int = 512,
                        S: int = 3,
                        jitter: float = 0.3,
                        N_samples: int = int(1e6),
                        sample_batch: int = 50000,
                        rng_seed: int = 12345,
                        use_sobol: bool = True,
                        verbose: bool = False) -> Dict[str, Any]:
    pts = np.ascontiguousarray(points, dtype=np.float64)
    n, m = pts.shape
    ref = np.asarray(reference, dtype=np.float64)
    out: Dict[str, Any] = {
        "input_k": int(n),
        "m": int(m),
        "reference": ref.tolist(),
        "hv_compressed": None,
        "compressed_k": None,
        "hv_qmc": None,
        "se_qmc": None,
        "timings": {},
    }
    t0 = time.time()
    # compression
    t1 = time.time()
    comp = compress_by_quantization(pts, bins_per_dim=compress_bins, target_k=compress_target)
    t2 = time.time()
    out["compressed_k"] = int(comp.shape[0])
    if comp.shape[0] == 0:
        out["hv_compressed"] = 0.0
    else:
        comp_res = count_dominated_and_hv_high_accuracy(comp, K=min(K, 128), S=1, jitter=0.0,
                                                       ref_value=float(ref[0]) if ref.size == 1 else float(ref[0]),
                                                       N_samples=min(20000, max(2000, comp.shape[0])),
                                                       sample_batch=5000,
                                                       rng=np.random.default_rng(rng_seed),
                                                       use_sobol=use_sobol and HAVE_SOBOL,
                                                       verbose=False)
        out["hv_compressed"] = float(comp_res["HV"])

    out["timings"]["compression_s"] = t2 - t1

    # main QMC on full set
    t3 = time.time()
    rng = np.random.default_rng(rng_seed)
    res_qmc = count_dominated_and_hv_high_accuracy(pts, K=K, S=S, jitter=jitter,
                                                   ref_value=float(ref[0]) if ref.size == 1 else float(ref[0]),
                                                   N_samples=N_samples, sample_batch=sample_batch,
                                                   rng=rng, use_sobol=use_sobol and HAVE_SOBOL, verbose=verbose)
    t4 = time.time()
    out["hv_qmc"] = float(res_qmc["HV"])
    out["se_qmc"] = float(res_qmc["se"])
    out["timings"]["qmc_s"] = t4 - t3
    out["timings"]["total_s"] = time.time() - t0
    out["n_D"] = res_qmc["n_D"]
    out["total_samples"] = res_qmc["total_samples"]
    return out
import argparse
import time
import json
import numpy as np
import os

# --- New loader to handle .npy, .npz, .csv/.txt, and legacy npz formats ---
def load_points(path):
    """
    Load points from path. Returns (pts, metadata)
      - pts: np.ndarray shape (k, m) float64
      - metadata: dict or None
    Supported:
      - .npz : looks for 'points', 'arr_0', 'X', 'points_np' keys
              other scalar / small entries are placed into metadata
      - .npy : single array saved with np.save
      - .csv / .txt: numeric table loadable with np.loadtxt
    Heuristics:
      - If a 2D array has a small first dimension (<=4) and many columns, we
        assume it was stored transposed and will transpose it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    metadata = None

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as z:
            # Try friendly keys first
            prefer_keys = ["points", "arr_0", "X", "points_np", "data"]
            found_key = None
            for k in prefer_keys:
                if k in z:
                    found_key = k
                    break
            if found_key is None:
                # pick the first array-like entry
                for k in z.files:
                    # skip obvious metadata keys that are not arrays if any
                    if isinstance(z[k], np.ndarray):
                        found_key = k
                        break
            if found_key is None:
                raise ValueError(f"No array-like dataset found inside {path}. Keys: {z.files}")

            arr = z[found_key]
            # build metadata from other entries (small/scalar items)
            metadata = {}
            for k in z.files:
                if k == found_key:
                    continue
                val = z[k]
                # try to keep only small metadata to avoid huge dumps
                try:
                    if np.shape(val) == () or (isinstance(val, np.ndarray) and val.size <= 16):
                        metadata[k] = val.tolist() if isinstance(val, np.ndarray) else val
                except Exception:
                    # fallback: skip problematic items
                    pass

    elif ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext in (".csv", ".txt"):
        arr = np.loadtxt(path, delimiter=None)
    else:
        # try to load as generic numpy array (let numpy raise if it fails)
        try:
            arr = np.load(path, allow_pickle=True)
        except Exception as e:
            raise ValueError(f"Unsupported file extension '{ext}' and numpy failed to load: {e}")

    # Validate and coerce to 2D numeric array
    if not isinstance(arr, np.ndarray):
        # maybe an object saved dict: attempt to extract 'points' key if present
        if isinstance(arr, (list, tuple)):
            arr = np.asarray(arr)
        else:
            raise ValueError(f"Loaded object is not a numpy array (type={type(arr)}).")

    if arr.ndim == 1:
        # treat as single point with m dims -> shape (1, m)
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        # try to coerce to 2D if possible
        raise ValueError(f"Loaded array must be 1D or 2D, got ndim={arr.ndim} with shape={arr.shape}.")

    # Heuristic: if first dim is "small" (e.g., number of objectives) and much smaller than second, transpose.
    k, m = arr.shape
    if k <= 4 and m > k:
        # Common case: file saved as (m, k) where m small (objectives). Transpose to (k, m).
        arr = arr.T
        k, m = arr.shape

    # Final checks: require at least 1 point and at least 1 objective
    if k < 1 or m < 1:
        raise ValueError(f"Invalid shape after loading: {arr.shape}")

    pts = np.asarray(arr, dtype=np.float64, order="C")
    return pts, (metadata if metadata else None)

def Compute_HV(PF: np.ndarray):
    """
    Fast nondominated set extraction using bitset dominance.
    Strict dominance enforced (correct ND definition).
    """
    M = PF.shape[1]
    ref = np.ones(M) * 1.1
    res = compute_hv_qmc_only(
        points=PF,
        reference=ref,
        compress_bins=512,
        compress_target=30000,
        K=512,
        S=3,
        jitter=0.3,
        N_samples=200_000,
        sample_batch=50_000,
        rng_seed=12345,
        use_sobol=True,
        verbose=False,
    )

    return res["hv_qmc"]


# -------------------------
# CLI & main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="QMC-only HV reporter (no exact HV).")
    p.add_argument("--file", "-f", type=str, default="skyline_numba_results.npz",
                   help="NPZ/NPY/CSV file containing points (rows = points, cols = objectives)")
    p.add_argument("--ref", "-r", type=float, default=1.1,
                   help="reference scalar per objective (default 1.1)")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="optional JSON output file")
    p.add_argument("--compress-bins", type=int, default=512,
                   help="bins per dim for compression")
    p.add_argument("--compress-target", type=int, default=30000,
                help="target compressed k (optional)")
    p.add_argument("--K", type=int, default=512,
                   help="quantile bins per axis for bitset (default 512)")
    p.add_argument("--S", type=int, default=3,
                   help="QMC repeats (dithers)")
    p.add_argument("--jitter", type=float, default=0.3,
                   help="edge jitter fraction (0 disables)")
    p.add_argument("--n-samples", type=int, default=1_000_000,
                   help="QMC samples per repeat (default 1,000,000)")
    p.add_argument("--batch", type=int, default=50_000,
                   help="sample batch size")
    p.add_argument("--no-sobol", action="store_true",
                   help="disable Sobol sampler (use PRNG)")
    p.add_argument("--verbose", action="store_true",
                   help="verbose progress")
    args = p.parse_args()

    pts, metadata = load_points(args.file)
    print(f"Loaded points: {pts.shape}")
    if metadata is not None:
        print("metadata:", metadata)
    m = pts.shape[1]
    ref = np.full(m, float(args.ref), dtype=np.float64)

    # The rest of your existing main flow unchanged...
    print("Config: NUMBA:", 'NUMBA' in globals(), "HAVE_SOBOL:", 'HAVE_SOBOL' in globals() and globals().get('HAVE_SOBOL') and not args.no_sobol)
    print("input_k:", int(pts.shape[0]), "m:", m)
    print("compress_bins:", args.compress_bins, "compress_target:", args.compress_target)
    print("K:", args.K, "S:", args.S, "jitter:", args.jitter)
    print("N_samples:", args.n_samples, "batch:", args.batch)

    t0 = time.time()
    # compute_hv_qmc_only should be defined elsewhere in your module
    res = compute_hv_qmc_only(pts, ref,
                              compress_bins=args.compress_bins,
                              compress_target=args.compress_target,
                              K=args.K,
                              S=args.S,
                              jitter=args.jitter,
                              N_samples=args.n_samples,
                              sample_batch=args.batch,
                              rng_seed=12345,
                              use_sobol=(not args.no_sobol) and globals().get('HAVE_SOBOL', False),
                              verbose=args.verbose)
    t1 = time.time()

    print("Results:")
    print(" input_k:", res["input_k"])
    print(" compressed_k:", res["compressed_k"])
    print(" hv_compressed (lower bound):", res["hv_compressed"])
    print(" hv_qmc estimate:", res["hv_qmc"], "±", res["se_qmc"])
    # convergence / consistency check
    if res["hv_compressed"] is not None and res["hv_qmc"] is not None and res["se_qmc"] is not None:
        lhs = res["hv_compressed"]
        rhs = res["hv_qmc"] - 2.0 * res["se_qmc"]
        converged = lhs <= rhs
        print(" convergence check (hv_compressed ≤ hv_qmc − 2·se):", converged)
        print(f"   hv_compressed = {lhs:.6e}")
        print(f"   hv_qmc − 2·se = {rhs:.6e}")
    else:
        converged = None

    print(" timings (s):", res["timings"])
    print(f"Total elapsed (script): {t1 - t0:.3f}s")

    outd = {
        "input_k": int(res["input_k"]),
        "m": int(res["m"]),
        "reference": res["reference"],
        "hv_compressed": res["hv_compressed"],
        "compressed_k": res["compressed_k"],
        "hv_qmc": res["hv_qmc"],
        "se_qmc": res["se_qmc"],
        "timings": res["timings"],
    }

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(outd, fh, indent=2)
        print("Wrote JSON to:", args.out)


if __name__ == "__main__":
    main()
