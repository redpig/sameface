"""Drift statistics: G-test, Jensen-Shannon divergence, Fisher aggregation,
Benjamini-Hochberg. Pure stdlib so the harness has no dependencies."""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---- chi-square survival function via regularized upper incomplete gamma ----

def _gammainc_upper_reg(a: float, x: float) -> float:
    """Q(a, x) = Gamma(a, x) / Gamma(a). Series for x < a+1, continued fraction otherwise."""
    if x <= 0:
        return 1.0
    if x < a + 1:
        # series for P, return 1 - P
        term = 1.0 / a
        s = term
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            s += term
            if abs(term) < abs(s) * 1e-14:
                break
        p = s * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, 1.0 - p)
    # Lentz continued fraction for Q
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(stat: float, df: int) -> float:
    if df <= 0:
        return 1.0
    return _gammainc_upper_reg(df / 2.0, stat / 2.0)


# ---- per-challenge tests ----

@dataclass
class ChallengeResult:
    id: str
    family: str
    g: float
    df: int
    p: float
    jsd: float
    n_base: int
    n_run: int
    top_base: str
    top_run: str
    p_adj: float = 1.0
    significant: bool = False


def _pool(base: dict[str, int], run: dict[str, int], min_expected: float = 5.0):
    """Pool low-expected-count categories into 'pooled' so the G-test is valid."""
    labels = sorted(set(base) | set(run))
    nb, nr = sum(base.values()), sum(run.values())
    total = nb + nr
    keep, pooled_b, pooled_r = [], 0, 0
    for l in labels:
        tot_l = base.get(l, 0) + run.get(l, 0)
        exp_min = tot_l * min(nb, nr) / total if total else 0
        if exp_min >= min_expected:
            keep.append(l)
        else:
            pooled_b += base.get(l, 0)
            pooled_r += run.get(l, 0)
    b = [base.get(l, 0) for l in keep]
    r = [run.get(l, 0) for l in keep]
    if pooled_b + pooled_r > 0:
        b.append(pooled_b)
        r.append(pooled_r)
    return b, r


def g_test(base: dict[str, int], run: dict[str, int]) -> tuple[float, int, float]:
    b, r = _pool(base, run)
    k = len(b)
    if k < 2:
        return 0.0, 0, 1.0
    nb, nr = sum(b), sum(r)
    total = nb + nr
    g = 0.0
    for ob, orr in zip(b, r):
        col = ob + orr
        for o, n in ((ob, nb), (orr, nr)):
            e = col * n / total
            if o > 0 and e > 0:
                g += o * math.log(o / e)
    g *= 2
    df = k - 1
    return g, df, chi2_sf(g, df)


def jsd(base: dict[str, int], run: dict[str, int], alpha: float = 0.5) -> float:
    labels = sorted(set(base) | set(run))
    k = len(labels)
    nb, nr = sum(base.values()), sum(run.values())
    p = [(base.get(l, 0) + alpha) / (nb + alpha * k) for l in labels]
    q = [(run.get(l, 0) + alpha) / (nr + alpha * k) for l in labels]
    m = [(a + b) / 2 for a, b in zip(p, q)]
    def kl(x, y):
        return sum(xi * math.log2(xi / yi) for xi, yi in zip(x, y) if xi > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare_challenge(cid: str, family: str, base: dict[str, int], run: dict[str, int]) -> ChallengeResult:
    g, df, p = g_test(base, run)
    top = lambda d: max(d, key=d.get) if d else "-"
    return ChallengeResult(cid, family, g, df, p, jsd(base, run),
                           sum(base.values()), sum(run.values()), top(base), top(run))


# ---- aggregation ----

def benjamini_hochberg(results: list[ChallengeResult], q: float = 0.05) -> None:
    m = len(results)
    order = sorted(results, key=lambda r: r.p)
    prev = 1.0
    for rank in range(m, 0, -1):
        r = order[rank - 1]
        adj = min(prev, r.p * m / rank)
        r.p_adj = adj
        prev = adj
    for r in results:
        r.significant = r.p_adj <= q


def fisher_combined(results: list[ChallengeResult]) -> float:
    ps = [max(r.p, 1e-300) for r in results if r.df > 0]
    if not ps:
        return 1.0
    stat = -2 * sum(math.log(p) for p in ps)
    return chi2_sf(stat, 2 * len(ps))


@dataclass
class Verdict:
    changed: bool
    fisher_p: float
    n_significant: int
    n_tested: int
    mean_jsd: float
    reasons: list[str]


def verdict(results: list[ChallengeResult], fisher_alpha: float = 0.01,
            min_sig: int = 3, q: float = 0.05) -> Verdict:
    benjamini_hochberg(results, q)
    fp = fisher_combined(results)
    nsig = sum(r.significant for r in results)
    tested = sum(r.df > 0 for r in results)
    mj = sum(r.jsd for r in results) / len(results) if results else 0.0
    reasons = []
    if fp < fisher_alpha:
        reasons.append(f"combined p={fp:.2e} < {fisher_alpha}")
    if nsig >= min_sig:
        reasons.append(f"{nsig} challenges significant at q={q}")
    return Verdict(bool(reasons), fp, nsig, tested, mj, reasons)
