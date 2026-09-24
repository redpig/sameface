"""Enrollment and identification on the "100 random integers" task.

  python -m sameface.seqid collect --model claude-sonnet-5 --answers 10 --out seqdata/sonnet.jsonl
  python -m sameface.seqid compare seqdata/*.jsonl            # cross-validated model comparison
  python -m sameface.seqid fingerprint seqdata/sonnet.jsonl   # top feature-model weights
  python -m sameface.seqid identify seqdata/*.jsonl --sample new.jsonl

Every file holds one endpoint's raw answers (one JSON object per line, with
"raw" and "nums"). Keep the raw samples: the models refit from them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

from .backend import AnthropicBackend, MockBackend
from .challenges import SEQUENCE_MAX_TOKENS, SEQUENCE_PROMPT, parse_sequence
from .seqmodel import (UniformModel, FeatureModel, enroll, identify, identify_sequential, make_model,
                       sprt, stream_llr, trials, _quantile)

LENGTHS = (5, 10, 20, 30, 50, 100, 200)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(backend, answers: int, out: str, workers: int = 4) -> list[list[int]]:
    """Sample `answers` sequence answers and append them to `out` (jsonl)."""
    def one(i):
        if hasattr(backend, "sample_raw"):
            data = backend.sample_raw(SEQUENCE_PROMPT, SEQUENCE_MAX_TOKENS)
            raw = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            return i, raw, data.get("stop_reason"), data.get("usage")
        return i, backend.sample(SEQUENCE_PROMPT, SEQUENCE_MAX_TOKENS), None, None

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    got = []
    with open(out, "a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for i, raw, stop, usage in ex.map(one, range(answers)):
            nums = parse_sequence(raw)
            if stop == "max_tokens" or len(nums) != 100:
                print(f"warning: answer {i}: {len(nums)} numbers, stop_reason={stop}", file=sys.stderr)
            f.write(json.dumps({"ts": dt.datetime.now(dt.timezone.utc).isoformat(), "endpoint": backend.name,
                                "prompt": SEQUENCE_PROMPT, "i": i, "raw": raw, "nums": nums,
                                "stop_reason": stop, "usage": usage}) + "\n")
            got.append(nums)
    return got


def load(path: str) -> tuple[str, list[list[int]]]:
    """(endpoint name, answers) from a jsonl file. Answers under 30 numbers are dropped."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    name = rows[0].get("endpoint") if rows else None
    answers = [r["nums"] if "nums" in r else parse_sequence(r["raw"]) for r in rows]
    return name or os.path.splitext(os.path.basename(path))[0], [a for a in answers if len(a) >= 30]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

# Smoothing chosen per training set by inner cross-validation (cheap models only).
GRID = {"markov": [{"beta": b} for b in (2, 10, 50, 200)],
        "ppm": [{"depth": d} for d in (0, 1, 2, 3)],
        "feature": [{"l2": 10.0}]}


def _tune(kind: str, data: dict[str, list[list[int]]]) -> dict:
    grid = GRID[kind]
    if len(grid) == 1:
        return grid[0]
    best, best_ll = grid[0], -math.inf
    for kw in grid:
        ll = 0.0
        for answers in data.values():
            for f in range(min(4, len(answers))):
                train = [a for i, a in enumerate(answers) if i % 4 != f]
                m = make_model(kind, **kw).fit(train)
                ll += sum(sum(m.step_logprobs(a)) for i, a in enumerate(answers) if i % 4 == f)
        if ll > best_ll:
            best, best_ll = kw, ll
    return best


def _fold_job(args):
    """One outer fold for one model type: fit every endpoint on its training
    answers, then score every endpoint's held-out answers."""
    kind, fold, folds, data, lengths, alpha, ref_kind = args
    train = {e: [a for i, a in enumerate(v) if i % folds != fold] for e, v in data.items()}
    test = {e: [a for i, a in enumerate(v) if i % folds == fold] for e, v in data.items()}
    kw = _tune(kind, train)
    out = {"ident": [], "ident_seq": [], "verify": [], "sprt": [], "kw": kw}
    enrolled, models = {}, {}
    for e in data:
        others = [a for o, v in train.items() if o != e for a in v]
        ref = UniformModel() if ref_kind == "uniform" or not others else make_model(kind, **kw).fit(others)
        enrolled[e] = enroll(e, train[e], kind, ref, lengths, **kw)
        models[e] = enrolled[e].model
    for e, answers in test.items():
        for k in lengths:
            for tr in trials(answers, k):
                best, _ = identify(models, tr, k)
                out["ident"].append((k, e, best == e))
        for i in range(len(answers)):
            stream = answers[i:] + answers[:i]
            d = identify_sequential(models, stream)
            out["ident_seq"].append((e, d.decision, d.n))
            for c, en in enrolled.items():
                d = sprt(en.model, en.ref, stream, alpha, alpha)
                out["sprt"].append((c, e, d.decision, d.n))
        for c, en in enrolled.items():
            for k in lengths:
                th = en.thresholds.get(k)
                for tr in trials(answers, k):
                    out["verify"].append((k, c, e, sum(stream_llr(en.model, en.ref, tr, k)), th))
    return kind, fold, out


@dataclass
class Row:
    kind: str
    k: int
    ident_acc: float
    ident_n: int
    frr: float
    far: float
    eer: float
    n_gen: int
    n_imp: int


def _eer(gen: list[float], imp: list[float]) -> float:
    """Equal error rate: the threshold where false rejects equal false accepts."""
    if not gen or not imp:
        return float("nan")
    best = 1.0
    for t in sorted(set(gen + imp)):
        frr = sum(g < t for g in gen) / len(gen)
        far = sum(i >= t for i in imp) / len(imp)
        best = min(best, max(frr, far))
    return best


def compare(data: dict[str, list[list[int]]], kinds=("feature", "markov", "ppm"), folds: int = 4,
            lengths=LENGTHS, alpha: float = 0.01, ref_kind: str = "pooled", workers: int | None = None) -> dict:
    """Cross-validated comparison. Answers are split into `folds` folds per
    endpoint; each model type is enrolled on the rest (thresholds from inner
    folds of those training answers only) and scored on the held-out fold."""
    lengths = [k for k in lengths if k <= 100 * max(1, min(len(v) for v in data.values()) // folds)]
    jobs = [(kind, f, folds, data, lengths, alpha, ref_kind) for kind in kinds for f in range(folds)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_fold_job, jobs))
    report = {"endpoints": {e: sum(map(len, v)) for e, v in data.items()}, "folds": folds,
              "alpha": alpha, "reference": ref_kind, "rows": [], "sequential": {}, "tuned": {}}
    for kind in kinds:
        outs = [o for kd, _, o in results if kd == kind]
        report["tuned"][kind] = [o["kw"] for o in outs]
        ident = [x for o in outs for x in o["ident"]]
        ver = [x for o in outs for x in o["verify"]]
        for k in lengths:
            idk = [ok for kk, _, ok in ident if kk == k]
            gen = [(s, th) for kk, c, e, s, th in ver if kk == k and c == e]
            imp = [(s, th) for kk, c, e, s, th in ver if kk == k and c != e]
            report["rows"].append(Row(
                kind, k, sum(idk) / len(idk) if idk else float("nan"), len(idk),
                sum(s < th for s, th in gen) / len(gen) if gen else float("nan"),
                sum(s >= th for s, th in imp) / len(imp) if imp else float("nan"),
                _eer([s for s, _ in gen], [s for s, _ in imp]), len(gen), len(imp)).__dict__)
        iseq = [x for o in outs for x in o["ident_seq"]]
        sp = [x for o in outs for x in o["sprt"]]
        gen = [(d, n) for c, e, d, n in sp if c == e]
        imp = [(d, n) for c, e, d, n in sp if c != e]
        decided = [(e, d, n) for e, d, n in iseq if d != "undecided"]
        report["sequential"][kind] = {
            "identify": {"trials": len(iseq), "undecided": len(iseq) - len(decided),
                         "errors": sum(d != e for e, d, _ in decided),
                         "median_n": _quantile([n for *_, n in decided], 0.5) if decided else None,
                         "p90_n": _quantile([n for *_, n in decided], 0.9) if decided else None},
            "verify_genuine": _sprt_summary(gen, "same"),
            "verify_impostor": _sprt_summary(imp, "different"),
        }
    return report


def _sprt_summary(xs, right: str) -> dict:
    decided = [(d, n) for d, n in xs if d != "undecided"]
    return {"trials": len(xs), "undecided": len(xs) - len(decided),
            "errors": sum(d != right for d, _ in decided),
            "median_n": _quantile([n for _, n in decided], 0.5) if decided else None,
            "p90_n": _quantile([n for _, n in decided], 0.9) if decided else None}


def needed(report: dict, kind: str, target: float = 0.05) -> dict:
    """Smallest tested length meeting the target error rate, or None."""
    rows = [r for r in report["rows"] if r["kind"] == kind]
    ident = next((r["k"] for r in rows if 1 - r["ident_acc"] <= target), None)
    verify = next((r["k"] for r in rows if r["frr"] <= target and r["far"] <= target), None)
    eer = next((r["k"] for r in rows if r["eer"] <= target), None)
    return {"identify": ident, "verify_threshold": verify, "verify_eer": eer}


def print_report(rep: dict) -> None:
    print("endpoints: " + ", ".join(f"{e} ({n} numbers)" for e, n in rep["endpoints"].items()))
    print(f"{rep['folds']}-fold cross-validation by answer; reference = {rep['reference']}; "
          f"thresholds at 5% false reject (normal fit to inner-fold held-out scores of the training answers)\n")
    print(f"{'model':8} {'k':>4}  {'ident acc':>9} {'(n)':>5}  {'FRR':>5} {'FAR':>5} {'EER':>5}  {'(gen/imp)':>9}")
    for r in rep["rows"]:
        print(f"{r['kind']:8} {r['k']:4d}  {r['ident_acc']:9.2f} {r['ident_n']:5d}  {r['frr']:5.2f} {r['far']:5.2f} "
              f"{r['eer']:5.2f}  {r['n_gen']:4d}/{r['n_imp']:<4d}")
    print(f"\nsequential (alpha = beta = {rep['alpha']}; identification margin = ln 100):")
    print(f"{'model':8} {'task':16} {'trials':>6} {'undecided':>9} {'errors':>6} {'median n':>8} {'p90 n':>6}")
    for kind, s in rep["sequential"].items():
        for task, v in s.items():
            med = "-" if v["median_n"] is None else f"{v['median_n']:.0f}"
            p90 = "-" if v["p90_n"] is None else f"{v['p90_n']:.0f}"
            print(f"{kind:8} {task:16} {v['trials']:6d} {v['undecided']:9d} {v['errors']:6d} {med:>8} {p90:>6}")
    print("\nnumbers needed for <= 5% error (smallest tested k; '-' = not reached):")
    for kind in rep["sequential"]:
        n = needed(rep, kind)
        fmt = lambda x: "-" if x is None else str(x)
        print(f"  {kind:8} identify {fmt(n['identify']):>4}   same/different (threshold) {fmt(n['verify_threshold']):>4}"
              f"   (EER) {fmt(n['verify_eer']):>4}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="sameface.seqid")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="sample sequence answers from one endpoint")
    c.add_argument("--model", default="claude-sonnet-5")
    c.add_argument("--mock", type=float, default=None, help="MockBackend with this bias (no API)")
    c.add_argument("--answers", type=int, default=10)
    c.add_argument("--temperature", type=float, default=1.0)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--out", required=True)
    p = sub.add_parser("compare", help="cross-validated comparison of model types")
    p.add_argument("files", nargs="+")
    p.add_argument("--kinds", default="feature,markov,ppm")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--reference", choices=("pooled", "uniform"), default="pooled")
    p.add_argument("--json", default=None, help="also write the report here")
    f = sub.add_parser("fingerprint", help="largest feature-model weights for one endpoint")
    f.add_argument("file")
    f.add_argument("--top", type=int, default=20)
    i = sub.add_parser("identify", help="which enrolled endpoint produced a sample")
    i.add_argument("files", nargs="+", help="enrolled endpoints")
    i.add_argument("--sample", required=True, help="jsonl of answers to identify")
    i.add_argument("--kind", default="feature")
    args = ap.parse_args(argv)

    if args.cmd == "collect":
        if args.mock is not None:
            backend = MockBackend(seed=int.from_bytes(os.urandom(4), "big"), bias=args.mock, name="mock")
        else:
            backend = AnthropicBackend(args.model, temperature=args.temperature)
        got = collect(backend, args.answers, args.out, args.workers)
        print(f"{args.out}: {len(got)} answers, {sum(map(len, got))} numbers")
        return 0
    if args.cmd == "compare":
        data = dict(load(p) for p in args.files)
        rep = compare(data, tuple(args.kinds.split(",")), args.folds, ref_kind=args.reference)
        print_report(rep)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(rep, fh, indent=1)
        return 0
    if args.cmd == "fingerprint":
        name, answers = load(args.file)
        m = FeatureModel().fit(answers)
        print(f"{name}: {len(answers)} answers, {sum(map(len, answers))} numbers")
        for feat, w in m.fingerprint(args.top):
            print(f"  {w:+6.2f}  {feat}")
        return 0
    if args.cmd == "identify":
        models = {name: make_model(args.kind).fit(ans) for name, ans in (load(p) for p in args.files)}
        _, sample = load(args.sample)
        best, ll = identify(models, sample)
        n = sum(map(len, sample))
        for name, v in sorted(ll.items(), key=lambda x: -x[1]):
            print(f"{'*' if name == best else ' '} {name:40} {v:10.1f}  ({v / n:+.3f} nats/number)")
        seq = identify_sequential(models, sample)
        print(f"sequential: {seq.decision} after {seq.n} numbers (lead {seq.llr:.1f} nats)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
