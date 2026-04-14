"""
Erdos Minimum Overlap — SA with sum-of-top-K objective.

The sum-of-top-K objective (K=50) avoids the "whack-a-mole" problem
by penalizing all peaks simultaneously. This reduces metric from
~0.404 (single-max SA) to ~0.393.
"""
import numpy as np
import time


def solve(n, seed=42):
    rng = np.random.default_rng(seed)
    size = 2 * n
    t_start = time.time()
    max_time = 43.5
    r_size = 2 * size + 2

    def build_r_fft(A, B):
        f_A = np.zeros(size + 1)
        f_B = np.zeros(size + 1)
        f_A[A] = 1.0
        f_B[B] = 1.0
        pad = r_size
        conv = np.fft.irfft(np.fft.rfft(f_A, n=pad) * np.fft.rfft(f_B, n=pad), n=pad)
        r = np.zeros(r_size, dtype=np.int32)
        c = np.rint(conv).astype(np.int32)
        r[:min(len(c), r_size)] = c[:r_size]
        return r

    TOP_K = 50

    perm = rng.permutation(size) + 1
    A = np.sort(perm[:n]).astype(np.int32)
    B = np.sort(perm[n:]).astype(np.int32)

    r = build_r_fft(A, B)
    current_max = int(np.max(r))
    best_max = current_max
    best_A = A.copy()

    def compute_score(r_arr):
        return int(np.sum(np.partition(r_arr, -TOP_K)[-TOP_K:]))

    current_score = compute_score(r)

    T_start = 100.0
    T_end = 0.01
    batch = 5000

    iteration = 0
    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            break

        frac = min(1.0, elapsed / max_time)
        T = T_start * ((T_end / T_start) ** frac)
        neg_inv_T = -1.0 / max(T, 1e-12)

        ai_b = rng.integers(0, n, size=batch)
        bi_b = rng.integers(0, n, size=batch)
        rands = rng.random(size=batch)

        for k in range(batch):
            ai = int(ai_b[k])
            bi = int(bi_b[k])
            a_elem = int(A[ai])
            b_elem = int(B[bi])

            s_Ab = A + b_elem
            s_Aa = A + a_elem
            s_aB = a_elem + B
            s_bB = b_elem + B

            r[s_Ab] -= 1
            r[s_Aa] += 1
            r[s_aB] -= 1
            r[s_bB] += 1
            ab_sum = a_elem + b_elem
            r[ab_sum] += 2
            r[2 * a_elem] -= 1
            r[2 * b_elem] -= 1

            new_max = int(np.max(r))
            new_score = compute_score(r)
            dd = new_score - current_score

            if dd <= 0 or rands[k] < np.exp(dd * neg_inv_T):
                A[ai] = b_elem
                B[bi] = a_elem
                current_max = new_max
                current_score = new_score
                if new_max < best_max:
                    best_max = new_max
                    best_A = A.copy()
            else:
                r[s_Ab] += 1
                r[s_Aa] -= 1
                r[s_aB] += 1
                r[s_bB] -= 1
                r[ab_sum] -= 2
                r[2 * a_elem] += 1
                r[2 * b_elem] += 1

        iteration += batch

    return set(best_A.tolist())
