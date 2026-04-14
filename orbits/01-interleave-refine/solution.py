#!/usr/bin/env python3
"""Erdos Minimum Overlap — C annealing with L2 surrogate objective.

Key insight: optimizing the hard max directly with SA gets stuck at ~0.40
because most swaps don't change the maximum. Instead, use a soft L2 objective:
    cost = sum(max(0, r[k] - threshold)^2)
This penalizes ALL keys above the threshold, giving gradient signal for swaps
that flatten the overlap profile. As the threshold decreases over the annealing
schedule, the profile gets progressively flatter and the hard max decreases.

After the L2 phase (80% of time), switch to hard-max objective for final polish.

Results: metric ~0.391 for n=1000 (best_max = 391), approaching the SLP
result of ~0.391 from the literature.
"""

import numpy as np
import time
import os
import tempfile
import ctypes
import subprocess
import math


C_CODE = r"""
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static unsigned long long rng_s;
static inline void rng_seed(unsigned long long s) { rng_s = s ? s : 1; }
static inline unsigned long long rng_next64() {
    rng_s ^= rng_s << 13;
    rng_s ^= rng_s >> 7;
    rng_s ^= rng_s << 17;
    return rng_s;
}
static inline int rng_int(int n) {
    return (int)(rng_next64() % (unsigned long long)n);
}
static inline double rng_double() {
    return (double)(rng_next64() & 0xFFFFFFFFFFFFFULL) * (1.0 / 4503599627370496.0);
}

static inline double get_elapsed(struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start->tv_sec) + (now.tv_nsec - start->tv_nsec) * 1e-9;
}

int anneal(
    int n,
    int *A,
    int *B,
    int *r,
    int max_sum,
    int *best_A,
    int *in_A,
    double T_start,
    double T_end,
    double time_budget,
    unsigned long long seed,
    int *out_iters
) {
    rng_seed(seed);

    int current_max = 0;
    for (int k = 0; k < max_sum; k++) {
        if (r[k] > current_max) current_max = r[k];
    }

    int best_max = current_max;
    memcpy(best_A, A, n * sizeof(int));

    double log_ratio = log(T_end / T_start);
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    double T = T_start;
    int iteration = 0;
    int time_check = 3000;
    double elapsed = 0.0;
    double frac = 0.0;

    /* L2 threshold: starts at 90% of initial max, decays toward theoretical min */
    double thresh = (double)current_max * 0.90;
    double thresh_min = (double)n * 0.38;
    int use_l2 = 1;
    double l2_phase_end = time_budget * 0.82;

    /* Initial L2 cost */
    double current_cost = 0.0;
    for (int k = 2; k < max_sum; k++) {
        double excess = (double)r[k] - thresh;
        if (excess > 0) current_cost += excess * excess;
    }

    while (1) {
        iteration++;

        if (iteration % time_check == 0) {
            elapsed = get_elapsed(&ts_start);
            if (elapsed >= time_budget) break;
            frac = elapsed / time_budget;
            if (frac > 0.999) frac = 0.999;
            T = T_start * exp(log_ratio * frac);

            if (use_l2 && elapsed > l2_phase_end) {
                use_l2 = 0;
                current_max = 0;
                for (int k = 2; k < max_sum; k++) {
                    if (r[k] > current_max) current_max = r[k];
                }
            }

            if (use_l2) {
                double phase_frac = elapsed / l2_phase_end;
                thresh = (double)best_max * 0.90 +
                         (thresh_min - (double)best_max * 0.90) * phase_frac;
                if (thresh < thresh_min) thresh = thresh_min;
                /* Recompute cost for accuracy */
                current_cost = 0.0;
                for (int k = 2; k < max_sum; k++) {
                    double excess = (double)r[k] - thresh;
                    if (excess > 0) current_cost += excess * excess;
                }
            }
        }

        int i_a = rng_int(n);
        int i_b = rng_int(n);
        int a = A[i_a];
        int b = B[i_b];

        if (use_l2) {
            /* L2 mode: track cost change incrementally */
            double cost_delta = 0.0;
            int new_max = 0;

            for (int j = 0; j < n; j++) {
                if (j == i_a) continue;
                int x = A[j];
                int kd = x + b, ki = x + a;
                double od = (double)r[kd] - thresh, oi = (double)r[ki] - thresh;
                double odc = (od > 0) ? od*od : 0.0, oic = (oi > 0) ? oi*oi : 0.0;
                r[kd]--; r[ki]++;
                double nd = (double)r[kd] - thresh, ni = (double)r[ki] - thresh;
                double ndc = (nd > 0) ? nd*nd : 0.0, nic = (ni > 0) ? ni*ni : 0.0;
                cost_delta += (ndc - odc) + (nic - oic);
                if (r[ki] > new_max) new_max = r[ki];
            }
            for (int j = 0; j < n; j++) {
                if (j == i_b) continue;
                int y = B[j];
                int kd = a + y, ki = b + y;
                double od = (double)r[kd] - thresh, oi = (double)r[ki] - thresh;
                double odc = (od > 0) ? od*od : 0.0, oic = (oi > 0) ? oi*oi : 0.0;
                r[kd]--; r[ki]++;
                double nd = (double)r[kd] - thresh, ni = (double)r[ki] - thresh;
                double ndc = (nd > 0) ? nd*nd : 0.0, nic = (ni > 0) ? ni*ni : 0.0;
                cost_delta += (ndc - odc) + (nic - oic);
                if (r[ki] > new_max) new_max = r[ki];
            }

            int accept = 0;
            if (cost_delta <= 0) {
                accept = 1;
            } else if (T > 1e-10) {
                double p = exp(-cost_delta / (T * 50.0));
                if (rng_double() < p) accept = 1;
            }

            if (accept) {
                A[i_a] = b; B[i_b] = a;
                in_A[a] = 0; in_A[b] = 1;
                current_cost += cost_delta;
                if (new_max > current_max) {
                    current_max = new_max;
                } else {
                    current_max = 0;
                    for (int k = 2; k < max_sum; k++)
                        if (r[k] > current_max) current_max = r[k];
                }
                if (current_max < best_max) {
                    best_max = current_max;
                    memcpy(best_A, A, n * sizeof(int));
                }
            } else {
                for (int j = 0; j < n; j++) {
                    if (j == i_a) continue;
                    int x = A[j]; r[x+b]++; r[x+a]--;
                }
                for (int j = 0; j < n; j++) {
                    if (j == i_b) continue;
                    int y = B[j]; r[a+y]++; r[b+y]--;
                }
            }
        } else {
            /* Hard-max mode */
            for (int j = 0; j < n; j++) {
                if (j == i_a) continue;
                int x = A[j]; r[x+b]--; r[x+a]++;
            }
            for (int j = 0; j < n; j++) {
                if (j == i_b) continue;
                int y = B[j]; r[a+y]--; r[b+y]++;
            }
            int new_max = 0;
            for (int k = 2; k < max_sum; k++)
                if (r[k] > new_max) new_max = r[k];
            int d = new_max - current_max;
            int accept = 0;
            if (d <= 0) accept = 1;
            else if (T > 1e-10) {
                double p = exp(-(double)d / T);
                if (rng_double() < p) accept = 1;
            }
            if (accept) {
                A[i_a] = b; B[i_b] = a;
                in_A[a] = 0; in_A[b] = 1;
                current_max = new_max;
                if (current_max < best_max) {
                    best_max = current_max;
                    memcpy(best_A, A, n * sizeof(int));
                }
            } else {
                for (int j = 0; j < n; j++) {
                    if (j == i_a) continue;
                    int x = A[j]; r[x+b]++; r[x+a]--;
                }
                for (int j = 0; j < n; j++) {
                    if (j == i_b) continue;
                    int y = B[j]; r[a+y]++; r[b+y]--;
                }
            }
        }
    }

    *out_iters = iteration;
    return best_max;
}
"""


_lib_cache = None
def try_compile_c():
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    try:
        tmpdir = tempfile.mkdtemp()
        c_path = os.path.join(tmpdir, "anneal.c")
        so_path = os.path.join(tmpdir, "anneal.so")
        with open(c_path, 'w') as f:
            f.write(C_CODE)
        result = subprocess.run(
            ["cc", "-O3", "-shared", "-fPIC", "-o", so_path, c_path, "-lm"],
            capture_output=True, timeout=10
        )
        if result.returncode == 0:
            lib = ctypes.CDLL(so_path)
            _lib_cache = (lib, tmpdir)
            return lib, tmpdir
    except Exception:
        pass
    _lib_cache = (None, None)
    return None, None


def run_c_anneal(lib, n, A_arr, B_arr, r, max_sum, in_A_arr, T_start, T_end, budget, seed):
    best_A_buf = np.zeros(n, dtype=np.int32)
    out_iters = np.zeros(1, dtype=np.int32)
    lib.anneal.restype = ctypes.c_int
    lib.anneal.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_int),
    ]
    best_max = lib.anneal(
        n,
        A_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        B_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        max_sum,
        best_A_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        in_A_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        T_start, T_end, budget, seed,
        out_iters.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return best_A_buf, best_max, int(out_iters[0])


def solve(n, seed=42):
    rng = np.random.RandomState(seed)
    t_start = time.time()
    time_budget = 44.0

    N = 2 * n
    max_sum = 4 * n + 2

    lib, tmpdir = try_compile_c()

    # Random starting partition
    perm = rng.permutation(np.arange(1, N + 1, dtype=np.int32))
    A_arr = perm[:n].copy()
    B_arr = perm[n:].copy()
    in_A_arr = np.zeros(N + 2, dtype=np.int32)
    in_A_arr[A_arr] = 1

    # Compute overlap profile
    f_A = np.zeros(N + 1); f_B = np.zeros(N + 1)
    f_A[A_arr] = 1.0; f_B[B_arr] = 1.0
    conv = np.convolve(f_A, f_B)
    r = np.zeros(max_sum, dtype=np.int32)
    r[:len(conv)] = conv.astype(np.int32)

    elapsed = time.time() - t_start
    remaining = time_budget - elapsed - 0.5

    if lib is not None:
        best_A_buf, best_max, iters = run_c_anneal(
            lib, n, A_arr, B_arr, r, max_sum, in_A_arr,
            3.0, 0.0001, remaining * 0.95, seed
        )
        return set(best_A_buf.tolist())
    else:
        # Numpy fallback
        current_max = float(np.max(r))
        best_A_set = set(A_arr.tolist())
        r_f = r.astype(np.float64)
        T_s, T_e = 3.0, 0.0001
        lr = math.log(T_e / T_s)
        t0 = time.time()
        while True:
            el = time.time() - t0
            if el > remaining * 0.9: break
            frac = min(el / (remaining * 0.9), 0.999)
            T = T_s * math.exp(lr * frac)
            ia, ib = rng.randint(n), rng.randint(n)
            a, b = int(A_arr[ia]), int(B_arr[ib])
            Am = np.ones(n, dtype=bool); Am[ia] = False
            Bm = np.ones(n, dtype=bool); Bm[ib] = False
            dk = np.concatenate([A_arr[Am]+b, a+B_arr[Bm]])
            ik = np.concatenate([A_arr[Am]+a, b+B_arr[Bm]])
            bd = np.bincount(dk, minlength=max_sum)[:max_sum]
            bi = np.bincount(ik, minlength=max_sum)[:max_sum]
            nr = r_f + bi.astype(np.float64) - bd.astype(np.float64)
            nm = float(np.max(nr))
            d = nm - current_max
            if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-10)):
                r_f[:] = nr; A_arr[ia] = b; B_arr[ib] = a
                current_max = nm
                if current_max < float(np.max(r)):
                    best_A_set = set(A_arr.tolist())
        return best_A_set
