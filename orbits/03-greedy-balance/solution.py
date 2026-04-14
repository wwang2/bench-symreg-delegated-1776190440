"""Erdos Minimum Overlap: Pure SA with maximum iteration throughput.

Clean SA loop with in-place conv updates. No frills, maximum speed.
Start from the best available initial construction and spend all time on SA.
"""
import numpy as np
import time


def solve(n, seed=42):
    rng = np.random.RandomState(seed)
    deadline = time.time() + 44

    # Find best starting construction
    best_ind = None
    best_m = float('inf')

    for p in _primes_near(2 * n, 8) + _primes_near(n, 4) + [3, 5, 7, 11]:
        ind = _qr_init(n, p, rng)
        m = _fast_metric(ind, n)
        if m < best_m:
            best_m = m
            best_ind = ind

    # All time to SA
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
    cur = float(np.max(conv))

    Ae = (np.where(ind_A[1:] > 0.5)[0] + 1)[:n].copy()
    Be = (np.where(ind_A[1:] < 0.5)[0] + 1)[:n].copy()

    budget = deadline - time.time() - 0.5
    if budget < 1:
        return ind_A
    t0 = time.time()

    T0 = 3.0
    Tf = 0.0005
    log_ratio = np.log(Tf / T0)

    best = cur
    best_ind = ind_A.copy()

    while True:
        el = time.time() - t0
        if el > budget:
            break
        T = T0 * np.exp(log_ratio * el / budget)

        ai = rng.randint(n)
        bi = rng.randint(n)
        a = int(Ae[ai])
        b = int(Be[bi])

        # In-place delta
        eb = min(alen, clen - b)
        ea = min(alen, clen - a)

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

        nm = float(np.max(conv))
        d = nm - cur

        if d <= 0 or rng.random() < np.exp(-d / T):
            ind_A[a] = 0.0; ind_A[b] = 1.0
            ind_B[a] = 1.0; ind_B[b] = 0.0
            cur = nm
            Ae[ai] = b; Be[bi] = a
            if cur < best:
                best = cur
                best_ind = ind_A.copy()
        else:
            if s < clen: conv[s] -= 2.0
            if da < clen: conv[da] += 1.0
            if db < clen: conv[db] += 1.0
            if ea > 0:
                conv[a:a+ea] += ind_B[:ea]
                conv[a:a+ea] -= ind_A[:ea]
            if eb > 0:
                conv[b:b+eb] -= ind_B[:eb]
                conv[b:b+eb] += ind_A[:eb]

    return best_ind
