"""Erdos Minimum Overlap: SA with adaptive penalty schedule.

The penalty weight increases over time:
- Early: low lambda -> explore broadly, reduce overall overlap
- Late: high lambda -> focus on flattening near the max

Penalty: sum of squared excesses above a threshold set relative to the best max.
Threshold tracks best_max - margin, where margin shrinks over time.
"""
import numpy as np
import time


def solve(n, seed=42):
    rng = np.random.RandomState(seed)
    deadline = time.time() + 44

    # Quick init selection
    best_ind = None
    best_m = float('inf')
    for p in _primes_near(2 * n, 5) + [3, 5, 7]:
        ind = _qr_init(n, p, rng)
        m = _fast_metric(ind, n)
        if m < best_m:
            best_m = m
            best_ind = ind

    best_ind = _sa(best_ind.copy(), n, rng, deadline)
    return set(np.where(best_ind[1:])[0] + 1)


def _primes_near(target, count):
    primes = []
    for off in range(500):
        for c in [target + off, target - off]:
            if c > 2 and _is_prime(c) and c not in primes:
                primes.append(c)
                if len(primes) >= count:
                    return primes
    return primes


def _is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True


def _qr_init(n, p, rng):
    qr = set()
    for x in range((p + 1) // 2):
        qr.add((x * x) % p)
    ind = np.zeros(2 * n + 1, dtype=np.float64)
    for i in range(1, 2 * n + 1):
        if i % p in qr:
            ind[i] = 1.0
    _fix(ind, n, rng)
    return ind


def _fix(ind, n, rng):
    A = np.where(ind[1:] > 0.5)[0] + 1
    B = np.where(ind[1:] < 0.5)[0] + 1
    e = len(A) - n
    if e > 0:
        ind[rng.choice(A, e, replace=False)] = 0.0
    elif e < 0:
        ind[rng.choice(B, -e, replace=False)] = 1.0


def _fast_metric(a, n):
    b = np.zeros_like(a)
    b[1:2*n+1] = 1.0 - a[1:2*n+1]
    return float(np.max(np.convolve(a, b))) / n


def _sa(ind_A, n, rng, deadline):
    alen = 2 * n + 1
    ind_B = np.zeros(alen, dtype=np.float64)
    ind_B[1:2*n+1] = 1.0 - ind_A[1:2*n+1]
    conv = np.convolve(ind_A, ind_B).astype(np.float64)
    clen = len(conv)

    Ae = (np.where(ind_A[1:] > 0.5)[0] + 1)[:n].copy()
    Be = (np.where(ind_A[1:] < 0.5)[0] + 1)[:n].copy()

    budget = deadline - time.time() - 0.3
    if budget < 0.5:
        return ind_A
    t0 = time.time()

    lo = 2
    hi = min(4 * n + 1, clen)
    slc = conv[lo:hi]  # view into conv

    cur_max = float(slc.max())
    best_max = cur_max
    best_ind = ind_A.copy()

    # Adaptive penalty parameters
    margin = 10.0
    lam = 0.02
    threshold = cur_max - margin

    # Compute initial objective
    excess = np.maximum(slc - threshold, 0.0)
    cur_penalty = float(np.sum(excess * excess))
    cur_obj = cur_max + lam * cur_penalty

    T0 = 5.0
    Tf = 0.0001
    log_ratio = np.log(Tf / T0)

    batch = 2000
    iib = batch
    update_every = 4000
    iters_since_update = 0

    while True:
        if iib >= batch:
            el = time.time() - t0
            if el > budget:
                break
            progress = min(el / budget, 1.0)
            T = T0 * np.exp(log_ratio * progress)

            # Adaptive: increase lambda and decrease margin over time
            lam = 0.01 + 0.04 * progress  # 0.01 -> 0.05
            margin = 12.0 - 8.0 * progress  # 12 -> 4

            ais = rng.randint(0, n, size=batch)
            bis = rng.randint(0, n, size=batch)
            us = rng.random(batch)
            iib = 0

        ai = ais[iib]; bi = bis[iib]; u = us[iib]; iib += 1
        a = int(Ae[ai]); b = int(Be[bi])

        eb = min(alen, clen - b)
        ea = min(alen, clen - a)

        # Apply delta in-place
        if eb > 0:
            conv[b:b+eb] += ind_B[:eb]
            conv[b:b+eb] -= ind_A[:eb]
        if ea > 0:
            conv[a:a+ea] -= ind_B[:ea]
            conv[a:a+ea] += ind_A[:ea]
        s = a + b
        if s < clen: conv[s] += 2.0
        da = 2 * a
        if da < clen: conv[da] -= 1.0
        db = 2 * b
        if db < clen: conv[db] -= 1.0

        # Compute new objective
        new_max = float(slc.max())
        excess = np.maximum(slc - threshold, 0.0)
        new_penalty = float(np.sum(excess * excess))
        new_obj = new_max + lam * new_penalty
        d = new_obj - cur_obj

        if d <= 0 or u < np.exp(-d / max(T, 1e-12)):
            ind_A[a] = 0.0; ind_A[b] = 1.0
            ind_B[a] = 1.0; ind_B[b] = 0.0
            cur_obj = new_obj
            cur_max = new_max
            cur_penalty = new_penalty
            Ae[ai] = b; Be[bi] = a
            if cur_max < best_max:
                best_max = cur_max
                best_ind = ind_A.copy()
        else:
            # Revert
            if s < clen: conv[s] -= 2.0
            if da < clen: conv[da] += 1.0
            if db < clen: conv[db] += 1.0
            if ea > 0:
                conv[a:a+ea] += ind_B[:ea]
                conv[a:a+ea] -= ind_A[:ea]
            if eb > 0:
                conv[b:b+eb] -= ind_B[:eb]
                conv[b:b+eb] += ind_A[:eb]

        # Periodically update threshold
        iters_since_update += 1
        if iters_since_update >= update_every:
            threshold = best_max - margin
            excess = np.maximum(slc - threshold, 0.0)
            cur_penalty = float(np.sum(excess * excess))
            cur_obj = cur_max + lam * cur_penalty
            iters_since_update = 0

    return best_ind
