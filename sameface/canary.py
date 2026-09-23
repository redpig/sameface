"""CLI: baseline, run, and report.

  python -m sameface.canary baseline --model claude-sonnet-5 --secret "$CANARY_SECRET"
  python -m sameface.canary run      --model claude-sonnet-5 --secret "$CANARY_SECRET"
  python -m sameface.canary demo     # mock backends, no API calls

State lives under --state-dir (default ./canary_state):
  runs/<timestamp>.jsonl   raw samples
  baseline.json            count vectors keyed by challenge id
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .backend import AnthropicBackend, MockBackend
from .challenges import Challenge, build_battery
from .stats import compare_challenge, verdict


def epoch_now() -> str:
    y, w, _ = dt.date.today().isocalendar()
    return f"{y}-W{w:02d}"


def collect(backend, battery: list[Challenge], n: int, workers: int = 4, log=None) -> dict[str, Counter]:
    counts: dict[str, Counter] = {c.id: Counter() for c in battery}
    jobs = [(c, i) for c in battery for i in range(n)]

    def one(job):
        c, i = job
        raw = backend.sample(c.prompt)
        lab = c.canon(raw)
        return c, i, raw, (lab if c.multi else [lab])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for c, i, raw, labels in ex.map(one, jobs):
            for label in labels:
                counts[c.id][label] += 1
            if log is not None:
                log.write(json.dumps({"ts": dt.datetime.now(dt.UTC).isoformat(), "backend": backend.name,
                                      "challenge": c.id, "family": c.family, "i": i,
                                      "raw": raw, "labels": labels}) + "\n")
    return counts


def load_baseline(path: str) -> dict | None:
    return json.load(open(path)) if os.path.exists(path) else None


def save_baseline(path: str, counts: dict[str, Counter], battery, backend_name, epoch):
    json.dump({"backend": backend_name, "epoch": epoch, "created": dt.datetime.now(dt.UTC).isoformat(),
               "families": {c.id: c.family for c in battery},
               "counts": {k: dict(v) for k, v in counts.items()}}, open(path, "w"), indent=1)


def report(base: dict, counts: dict[str, Counter], battery) -> int:
    results = [compare_challenge(c.id, c.family, base["counts"].get(c.id, {}), counts[c.id]) for c in battery
               if c.id in base["counts"]]
    v = verdict(results)
    print(f"baseline: {base['backend']}  epoch {base['epoch']}  ({base['created'][:16]}Z)")
    print(f"{'challenge':16} {'family':11} {'p':>9} {'p_adj':>8} {'JSD':>6}  top(base -> run)")
    for r in sorted(results, key=lambda r: r.p):
        flag = "*" if r.significant else " "
        print(f"{flag}{r.id:15} {r.family:11} {r.p:9.3g} {r.p_adj:8.3g} {r.jsd:6.3f}  {r.top_base} -> {r.top_run}")
    print()
    print(f"combined Fisher p = {v.fisher_p:.3g}   significant: {v.n_significant}/{v.n_tested}   mean JSD = {v.mean_jsd:.3f}")
    if v.changed:
        print("VERDICT: CHANGED  (" + "; ".join(v.reasons) + ")")
        return 2
    print("VERDICT: no change detected")
    return 0


def make_backend(args, b: bool = False):
    """b=True builds the second (wrapped) surface for --diff."""
    seed = int(dt.datetime.now().timestamp()) + (1 if b else 0)
    if args.mock is not None:
        if b:
            return MockBackend(seed=seed, bias=args.mock, injected=args.system_b or "",
                               steer_to=args.mock_steer, hide=args.mock_hide)
        return MockBackend(seed=seed, bias=args.mock)
    model = (args.model_b or args.model) if b else args.model
    system = args.system_b if (b and args.system_b) else None
    kw = {"system": system} if system else {}
    return AnthropicBackend(model, temperature=args.temperature, **kw)


def diff_report(base_a, base_b, counts_a, counts_b, battery) -> int:
    """Three comparisons: A vs its baseline, B vs its baseline, and the A-B gap
    now vs the gap at baseline. Both moved -> model; only B moved -> wrapper."""
    from .stats import jsd
    ra = [compare_challenge(c.id, c.family, base_a["counts"].get(c.id, {}), counts_a[c.id]) for c in battery]
    rb = [compare_challenge(c.id, c.family, base_b["counts"].get(c.id, {}), counts_b[c.id]) for c in battery]
    va, vb = verdict(ra), verdict(rb)
    gap_then = {c.id: jsd(base_a["counts"].get(c.id, {}), base_b["counts"].get(c.id, {})) for c in battery}
    gap_now = {c.id: jsd(counts_a[c.id], counts_b[c.id]) for c in battery}
    print(f"{'challenge':16} {'A p_adj':>8} {'B p_adj':>8} {'gap then':>9} {'gap now':>8}")
    for c in battery:
        a = next(r for r in ra if r.id == c.id); b = next(r for r in rb if r.id == c.id)
        fa = "*" if a.significant else " "; fb = "*" if b.significant else " "
        print(f"{c.id:16} {fa}{a.p_adj:7.3g} {fb}{b.p_adj:7.3g} {gap_then[c.id]:9.3f} {gap_now[c.id]:8.3f}")
    print()
    print(f"A (bare):    {'CHANGED' if va.changed else 'stable'}  fisher p={va.fisher_p:.3g}  sig {va.n_significant}/{va.n_tested}")
    print(f"B (wrapped): {'CHANGED' if vb.changed else 'stable'}  fisher p={vb.fisher_p:.3g}  sig {vb.n_significant}/{vb.n_tested}")
    if va.changed and vb.changed:
        print("READ: both surfaces moved -> model or shared inference stack changed")
    elif vb.changed:
        print("READ: only the wrapped surface moved -> wrapper / injected context changed")
    elif va.changed:
        print("READ: only the bare surface moved -> unusual; check routing on A")
    else:
        print("READ: no change on either surface")
    return 2 if (va.changed or vb.changed) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sameface")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("baseline", "run"):
        s = sub.add_parser(name)
        s.add_argument("--model", default="claude-sonnet-5")
        s.add_argument("--secret", default=os.environ.get("CANARY_SECRET", "dev-secret"))
        s.add_argument("--epoch", default=None, help="defaults to current ISO week")
        s.add_argument("--n", type=int, default=20, help="samples per challenge")
        s.add_argument("--per-family", type=int, default=4)
        s.add_argument("--temperature", type=float, default=1.0)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--state-dir", default="canary_state")
        s.add_argument("--mock", type=float, default=None, help="use MockBackend with this bias (no API)")
        s.add_argument("--accept", action="store_true", help="(run) overwrite baseline with this run")
        s.add_argument("--diff", action="store_true", help="also sample a second, wrapped surface (B)")
        s.add_argument("--model-b", default=None, help="model for surface B (default: same as --model)")
        s.add_argument("--system-b", default=None, help="system prompt for B (simulates a wrapper); mock: injected text")
        s.add_argument("--mock-steer", default=None, help="mock B: option name the wrapper nudges toward")
        s.add_argument("--mock-hide", action="store_true", help="mock B: wrapper denies having instructions")
    sub.add_parser("demo")
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        return demo()

    epoch = args.epoch or epoch_now()
    battery = build_battery(args.secret, epoch, args.per_family)
    backend = make_backend(args)
    os.makedirs(os.path.join(args.state_dir, "runs"), exist_ok=True)
    bpath = os.path.join(args.state_dir, "baseline.json")
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")

    with open(os.path.join(args.state_dir, "runs", f"{stamp}.jsonl"), "w") as log:
        counts = collect(backend, battery, args.n, args.workers, log)
        counts_b = None
        if args.diff:
            backend_b = make_backend(args, b=True)
            counts_b = collect(backend_b, battery, args.n, args.workers, log)
    bpath_b = os.path.join(args.state_dir, "baseline_b.json")

    if args.cmd == "baseline":
        save_baseline(bpath, counts, battery, backend.name, epoch)
        if counts_b is not None:
            save_baseline(bpath_b, counts_b, battery, backend_b.name, epoch)
        print(f"baseline written: {bpath}{' + baseline_b.json' if counts_b else ''}  ({len(battery)} challenges x {args.n} samples)")
        return 0

    base = load_baseline(bpath)
    if base is None:
        print("no baseline; run `baseline` first", file=sys.stderr)
        return 1
    if counts_b is not None:
        base_b = load_baseline(bpath_b)
        if base_b is None:
            print("no baseline_b; run `baseline --diff` first", file=sys.stderr)
            return 1
        code = diff_report(base, base_b, counts, counts_b, battery)
        if args.accept:
            save_baseline(bpath, counts, battery, backend.name, epoch)
            save_baseline(bpath_b, counts_b, battery, backend_b.name, epoch)
            print("both baselines replaced with this run")
        return code
    if base["epoch"] != epoch:
        print(f"warning: baseline epoch {base['epoch']} != current {epoch}; wording differs, compare with care")
    code = report(base, counts, battery)
    if args.accept:
        save_baseline(bpath, counts, battery, backend.name, epoch)
        print("baseline replaced with this run")
    return code


def demo():
    """Same mock model twice (expect no change), then a biased mock (expect CHANGED)."""
    battery = build_battery("demo-secret", "2026-W39")
    base_counts = collect(MockBackend(seed=1, bias=0.0), battery, 20)
    base = {"backend": "mock", "epoch": "2026-W39", "created": dt.datetime.now(dt.UTC).isoformat(),
            "counts": {k: dict(v) for k, v in base_counts.items()}}
    print("=== same model, fresh samples ===")
    report(base, collect(MockBackend(seed=2, bias=0.0), battery, 20), battery)
    print("\n=== 'updated' model (bias 0.3) ===")
    report(base, collect(MockBackend(seed=3, bias=0.3), battery, 20), battery)
    return 0


if __name__ == "__main__":
    sys.exit(main())
