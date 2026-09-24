"""Likelihood models for the "100 random integers" task.

Each model gives p(n | history) for n in 1..100, where history is the numbers
already written in the same answer (every answer starts fresh). Three model
types share one interface (`fit`, `step_logprobs`):

- FeatureModel: log-linear (maximum-entropy) next-number model,
  p(n | h) proportional to exp(w . f(h, n)). The fitted weights are a readable
  fingerprint ("likes 37", "avoids repeats", "jumps far").
- MarkovModel: first-order chain on the raw numbers, smoothed toward the
  number's overall frequency. This is the model in the original disclosure.
- PPMModel: variable-order context model (PPM with Witten-Bell escapes,
  interpolated, no exclusion), contexts of up to `depth` previous numbers.

Scoring is a per-step log-likelihood ratio (LLR) of an enrolled model against
a reference (uniform, or pooled from other endpoints). `enroll` sets
thresholds from held-out baseline answers, so it does not assume the numbers
within one answer are independent. `sprt` runs the ratio until it crosses a
bound and reports how many numbers it needed; `identify` picks the most
likely enrolled model.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

N = 100
NUMS = range(1, N + 1)
LOG_UNIFORM = -math.log(N)


class UniformModel:
    kind = "uniform"

    def fit(self, seqs):
        return self

    def step_logprobs(self, seq: list[int]) -> list[float]:
        return [LOG_UNIFORM] * len(seq)


# ---------------------------------------------------------------------------
# First-order Markov chain
# ---------------------------------------------------------------------------

class MarkovModel:
    """p(n | prev) = (c(prev, n) + beta * u(n)) / (c(prev) + beta), where u is
    the add-one smoothed frequency of n. The first number of an answer uses a
    start state with the same smoothing."""
    kind = "markov"

    def __init__(self, beta: float = 50.0):
        self.beta = beta

    def fit(self, seqs):
        self.uni = Counter()
        self.trans: dict[int, Counter] = defaultdict(Counter)
        for s in seqs:
            prev = 0  # start state
            for n in s:
                self.uni[n] += 1
                self.trans[prev][n] += 1
                prev = n
        tot = sum(self.uni.values())
        self.u = {n: (self.uni[n] + 1) / (tot + N) for n in NUMS}
        self.row_tot = {k: sum(v.values()) for k, v in self.trans.items()}
        return self

    def step_logprobs(self, seq):
        out, prev = [], 0
        for n in seq:
            row = self.trans.get(prev)
            c = row[n] if row else 0
            tot = self.row_tot.get(prev, 0)
            out.append(math.log((c + self.beta * self.u[n]) / (tot + self.beta)))
            prev = n
        return out


# ---------------------------------------------------------------------------
# Variable-order model (PPM / Witten-Bell)
# ---------------------------------------------------------------------------

class PPMModel:
    """Interpolated Witten-Bell smoothing over contexts of 0..depth previous
    numbers (start-of-answer padding counts as context). Order -1 is uniform.
    p_k(n | ctx) = (c(ctx, n) + T(ctx) p_{k-1}(n | ctx')) / (c(ctx) + T(ctx)),
    T = number of distinct numbers seen after ctx (the PPM-C escape count)."""
    kind = "ppm"

    def __init__(self, depth: int = 1):
        self.depth = depth

    def _ctxs(self, hist):
        padded = (0,) * self.depth + tuple(hist)
        return [padded[len(padded) - k:] if k else () for k in range(self.depth + 1)]

    def fit(self, seqs):
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        for s in seqs:
            for t, n in enumerate(s):
                for ctx in self._ctxs(s[max(0, t - self.depth):t]):
                    self.counts[ctx][n] += 1
        self.stats = {ctx: (sum(c.values()), len(c)) for ctx, c in self.counts.items()}
        return self

    def prob(self, hist, n) -> float:
        p = 1 / N
        for ctx in self._ctxs(hist):
            st = self.stats.get(ctx)
            if not st:
                break  # longer contexts are unseen too
            tot, distinct = st
            p = (self.counts[ctx][n] + distinct * p) / (tot + distinct)
        return p

    def step_logprobs(self, seq):
        return [math.log(self.prob(seq[max(0, t - self.depth):t], n)) for t, n in enumerate(seq)]


# ---------------------------------------------------------------------------
# Log-linear feature model
# ---------------------------------------------------------------------------

def _decade(n: int) -> int:
    return min(n // 10, 9)  # 1-9 -> 0, 10-19 -> 1, ..., 90-100 -> 9


_DECADE_SIZE = [9] + [10] * 8 + [11]


def _reverse(n: int) -> int | None:
    return int(str(n)[::-1]) if 10 <= n <= 99 and n % 10 and n % 11 else None


_JUMPS = [(0, "0"), (1, "1"), (3, "2-3"), (9, "4-9"), (19, "10-19"), (39, "20-39"), (99, "40+")]


def _jump_name(d: int) -> str:
    a = abs(d)
    label = next(name for hi, name in _JUMPS if a <= hi)
    return label if d == 0 else ("+" if d > 0 else "-") + label


def static_features(n: int) -> list[str]:
    """Features of the number alone: favourites, avoided numbers, digit patterns."""
    f = [f"num={n}", f"decade={_decade(n)}", "round" if n % 10 == 0 else f"last_digit={n % 10}"]
    if n % 10 == 5:
        f.append("mult5")
    if 10 < n < 100 and n % 11 == 0:
        f.append("repdigit")
    if n < 10:
        f.append("single_digit")
    return f


def dynamic_features(hist: list[int], n: int, seen: dict[int, int], dec_count: list[int]) -> list[str]:
    """Features relating n to the answer so far. `seen` maps number -> last
    position, `dec_count` counts history per decade (both describe `hist`)."""
    t = len(hist)
    f = []
    if t == 0:
        return [f"first&decade={_decade(n)}", f"first&last_digit={n % 10}"]
    prev = hist[-1]
    f.append(f"jump={_jump_name(n - prev)}")
    if n != prev:
        if _decade(n) == _decade(prev):
            f.append("same_decade_as_prev")
        if n % 10 == prev % 10:
            f.append("same_last_digit_as_prev")
        if _reverse(prev) == n:
            f.append("reversed_prev")
    last = seen.get(n)
    if last is not None:
        ago = t - last
        f.append("seen")
        f.append("seen_ago=1-5" if ago <= 5 else "seen_ago=6-20" if ago <= 20 else "seen_ago=21+")
    # how full n's range of 10 already is, relative to an even spread
    r = dec_count[_decade(n)] - t * _DECADE_SIZE[_decade(n)] / N
    f.append("decade_fill=" + ("<<" if r < -1.5 else "<" if r < -0.5 else "=" if r <= 0.5 else ">" if r <= 1.5 else ">>"))
    if t < 5:
        f.append(f"early&decade={_decade(n)}")
    return f


@dataclass
class _Step:
    obs: int            # index 0..99 of the observed number
    dyn: list[tuple]    # per candidate: tuple of dynamic feature indices


class FeatureModel:
    kind = "feature"

    def __init__(self, l2: float = 10.0, iters: int = 150):
        self.l2, self.iters = l2, iters
        self.index: dict[str, int] = {}
        self.w: list[float] = []

    # -- feature indexing --
    def _idx(self, name: str, grow: bool) -> int | None:
        i = self.index.get(name)
        if i is None and grow:
            i = self.index[name] = len(self.index)
        return i

    def _encode(self, names, grow):
        return tuple(i for i in (self._idx(x, grow) for x in names) if i is not None)

    def _steps(self, seq, grow):
        steps, seen, dec = [], {}, [0] * 10
        for t, n in enumerate(seq):
            hist = seq[:t]
            dyn = [self._encode(dynamic_features(hist, c, seen, dec), grow) for c in NUMS]
            steps.append(_Step(n - 1, dyn))
            seen[n] = t
            dec[_decade(n)] += 1
        return steps

    def _static(self, grow):
        return [self._encode(static_features(c), grow) for c in NUMS]

    # -- likelihood --
    def _scores(self, w, static_score, step):
        return [static_score[c] + sum(w[i] for i in step.dyn[c]) for c in range(N)]

    def _objective(self, w, steps, static):
        """Negative penalized log-likelihood and its gradient."""
        static_score = [sum(w[i] for i in fs) for fs in static]
        grad = [0.0] * len(w)
        static_mass = [0.0] * N
        nll = 0.0
        for st in steps:
            s = self._scores(w, static_score, st)
            m = max(s)
            e = [math.exp(x - m) for x in s]
            z = sum(e)
            nll -= s[st.obs] - m - math.log(z)
            for c in range(N):
                p = e[c] / z
                static_mass[c] += p
                for i in st.dyn[c]:
                    grad[i] += p
            for i in st.dyn[st.obs]:
                grad[i] -= 1
        for c in range(N):
            for i in static[c]:
                grad[i] += static_mass[c]
        for st in steps:
            for i in static[st.obs]:
                grad[i] -= 1
        nll += 0.5 * self.l2 * sum(x * x for x in w)
        grad = [g + self.l2 * x for g, x in zip(grad, w)]
        return nll, grad

    def fit(self, seqs):
        self.index = {}
        static = self._static(grow=True)
        steps = [st for s in seqs for st in self._steps(s, grow=True)]
        self.w = _lbfgs(lambda w: self._objective(w, steps, static), [0.0] * len(self.index), self.iters)
        self._static_score = [sum(self.w[i] for i in fs) for fs in static]
        return self

    def step_logprobs(self, seq):
        out = []
        for st in self._steps(seq, grow=False):
            s = self._scores(self.w, self._static_score, st)
            m = max(s)
            out.append(s[st.obs] - m - math.log(sum(math.exp(x - m) for x in s)))
        return out

    def fingerprint(self, k: int = 15) -> list[tuple[str, float]]:
        """The k largest weights by magnitude: the model's most distinctive habits."""
        names = sorted(self.index, key=self.index.get)
        return sorted(zip(names, self.w), key=lambda x: -abs(x[1]))[:k]


def _lbfgs(f, x, iters: int = 150, m: int = 8, tol: float = 1e-6) -> list[float]:
    """Minimize f (returns value, gradient) with L-BFGS and a backtracking line search."""
    fx, g = f(x)
    hist: list[tuple[list[float], list[float], float]] = []
    dot = lambda a, b: sum(p * q for p, q in zip(a, b))
    for _ in range(iters):
        # two-loop recursion
        q = list(g)
        alphas = []
        for s, y, rho in reversed(hist):
            a = rho * dot(s, q)
            alphas.append(a)
            q = [qi - a * yi for qi, yi in zip(q, y)]
        if hist:
            s, y, _ = hist[-1]
            gamma = dot(s, y) / dot(y, y)
            q = [gamma * qi for qi in q]
        else:
            q = [qi / max(1.0, math.sqrt(dot(g, g))) for qi in q]
        for (s, y, rho), a in zip(hist, reversed(alphas)):
            b = rho * dot(y, q)
            q = [qi + (a - b) * si for qi, si in zip(q, s)]
        d = [-qi for qi in q]
        gd = dot(g, d)
        if gd >= 0:  # not a descent direction; restart
            hist.clear()
            d = [-gi for gi in g]
            gd = dot(g, d)
        step = 1.0
        while True:
            xn = [xi + step * di for xi, di in zip(x, d)]
            fn, gn = f(xn)
            if fn <= fx + 1e-4 * step * gd or step < 1e-10:
                break
            step *= 0.5
        s = [a - b for a, b in zip(xn, x)]
        y = [a - b for a, b in zip(gn, g)]
        sy = dot(s, y)
        if sy > 1e-12:
            hist.append((s, y, 1 / sy))
            if len(hist) > m:
                hist.pop(0)
        converged = abs(fx - fn) <= tol * max(1.0, abs(fx))
        x, fx, g = xn, fn, gn
        if converged:
            break
    return x


MODELS = {"feature": FeatureModel, "markov": MarkovModel, "ppm": PPMModel}


def make_model(kind: str, **kw):
    return UniformModel() if kind == "uniform" else MODELS[kind](**kw)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def llr_steps(model, ref, seq: list[int]) -> list[float]:
    """Per-number log-likelihood ratio log p_model - log p_ref (nats)."""
    return [a - b for a, b in zip(model.step_logprobs(seq), ref.step_logprobs(seq))]


def stream_llr(model, ref, answers: list[list[int]], k: int | None = None) -> list[float]:
    """Per-number LLR over several answers read in order, cut at k numbers."""
    out = []
    for a in answers:
        out += llr_steps(model, ref, a)
        if k is not None and len(out) >= k:
            return out[:k]
    return out


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass
class Enrollment:
    """An enrolled endpoint: a fitted model, a reference, and per-length
    thresholds on the summed LLR, set from held-out baseline answers."""
    name: str
    model: object
    ref: object
    thresholds: dict[int, float] = field(default_factory=dict)
    heldout: dict[int, list[float]] = field(default_factory=dict)

    def score(self, answers: list[list[int]], k: int | None = None) -> float:
        return sum(stream_llr(self.model, self.ref, answers, k))

    def same(self, answers: list[list[int]], k: int) -> bool:
        """True when the first k numbers look like the enrolled endpoint."""
        return self.score(answers, k) >= self.thresholds[k]


def enroll(name: str, answers: list[list[int]], kind: str = "feature", ref=None,
           lengths=(10, 20, 50, 100), folds: int = 4, false_reject: float = 0.05, **kw) -> Enrollment:
    """Fit `kind` on the baseline answers. Thresholds: split the answers into
    folds, fit on the rest, and score each held-out answer's first k numbers
    (answers joined for k > 100). A baseline has too few answers to read a 5%
    quantile directly, so the threshold is the `false_reject` quantile of a
    normal fitted to those held-out scores (a sum of many per-number LLRs)."""
    ref = ref or UniformModel()
    heldout: dict[int, list[float]] = {k: [] for k in lengths}
    folds = max(2, min(folds, len(answers)))
    for f in range(folds):
        train = [a for i, a in enumerate(answers) if i % folds != f]
        test = [a for i, a in enumerate(answers) if i % folds == f]
        m = make_model(kind, **kw).fit(train)
        for k in lengths:
            for trial in trials(test, k):
                heldout[k].append(sum(stream_llr(m, ref, trial, k)))
    z = statistics.NormalDist().inv_cdf(false_reject)
    th = {k: statistics.fmean(v) + z * (statistics.stdev(v) if len(v) > 1 else 0.0)
          for k, v in heldout.items() if v}
    return Enrollment(name, make_model(kind, **kw).fit(answers), ref, th, heldout)


def trials(answers: list[list[int]], k: int) -> list[list[list[int]]]:
    """Disjoint groups of consecutive answers holding at least k numbers each.
    Each trial starts at the beginning of an answer (a fresh history)."""
    out, cur, n = [], [], 0
    for a in answers:
        cur.append(a)
        n += len(a)
        if n >= k:
            out.append(cur)
            cur, n = [], 0
    return out


@dataclass
class SeqDecision:
    decision: str       # "same", "different" or "undecided"
    n: int              # numbers read before stopping
    llr: float


def sprt(model, ref, answers: list[list[int]], alpha: float = 0.01, beta: float = 0.01) -> SeqDecision:
    """Wald's sequential test of model vs ref on a stream of answers. Stops at
    log((1-beta)/alpha) ("same") or log(beta/(1-alpha)) ("different"). The
    numbers within an answer are not independent, so the nominal error rates
    are approximate; measure the real ones on held-out data (see seqid.py)."""
    hi, lo = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    s, t = 0.0, 0
    for a in answers:
        for x in llr_steps(model, ref, a):
            s += x
            t += 1
            if s >= hi:
                return SeqDecision("same", t, s)
            if s <= lo:
                return SeqDecision("different", t, s)
    return SeqDecision("undecided", t, s)


def identify(models: dict[str, object], answers: list[list[int]], k: int | None = None) -> tuple[str, dict[str, float]]:
    """The enrolled model under which the sample is most likely, and every
    model's total log-likelihood."""
    ll = {}
    for name, m in models.items():
        steps = []
        for a in answers:
            steps += m.step_logprobs(a)
            if k is not None and len(steps) >= k:
                break
        ll[name] = sum(steps[:k] if k is not None else steps)
    return max(ll, key=ll.get), ll


def identify_sequential(models: dict[str, object], answers: list[list[int]], margin: float = math.log(100)) -> SeqDecision:
    """Read numbers until the most likely model leads the runner-up by
    `margin` nats (a likelihood ratio of 100 by default)."""
    tot = dict.fromkeys(models, 0.0)
    t, lead = 0, 0.0
    for a in answers:
        per = {name: m.step_logprobs(a) for name, m in models.items()}
        for j in range(len(a)):
            t += 1
            for name in models:
                tot[name] += per[name][j]
            first, second = (sorted(tot.values(), reverse=True) + [-math.inf])[:2]
            lead = first - second
            if lead >= margin:
                return SeqDecision(max(tot, key=tot.get), t, lead)
    return SeqDecision("undecided", t, lead)
