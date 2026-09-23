"""Backends: something that turns a prompt into a single sampled response.

- AnthropicBackend: the Messages API via urllib (no SDK dependency).
- MockBackend: a synthetic policy with tunable bias, for dry runs and for
  demonstrating the drift test without spending tokens.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Protocol

SYSTEM = "You are a helpful assistant. Follow the reply format exactly."


class Backend(Protocol):
    name: str
    def sample(self, prompt: str) -> str: ...


class AnthropicBackend:
    def __init__(self, model: str, api_key: str | None = None, temperature: float = 1.0,
                 max_tokens: int = 120, system: str = SYSTEM, retries: int = 4):
        self.model = model
        self.name = f"anthropic:{model}"
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.temperature, self.max_tokens, self.system, self.retries = temperature, max_tokens, system, retries

    def sample(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "system": self.system, "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={"content-type": "application/json", "x-api-key": self.key,
                     "anthropic-version": "2023-06-01"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.load(r)
                return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")


class MockBackend:
    """A fake model whose 'habits' are a seed plus a bias knob.

    Two MockBackends with the same seed and bias are 'the same model';
    changing `bias` simulates an update. Responses are built to be parsed by
    the canonicalizers in challenges.py."""

    def __init__(self, seed: int = 1, bias: float = 0.0, name: str = "mock",
                 injected: str = "", steer_to: str | None = None, hide: bool = False):
        """injected: simulated wrapper text (its length perturbs context-sensitive
        answers); steer_to: an option name the wrapper nudges toward; hide: the
        wrapper tells the model to deny having instructions."""
        self.rng = random.Random(seed)
        self.habit = random.Random(0)  # fixed 'trained' preferences, independent of sampling seed
        self.bias = bias
        self.injected, self.steer_to, self.hide = injected, steer_to, hide
        self.ctx_words = 60 + len(injected.split())
        self.name = f"{name}:seed{seed}:bias{bias}:inj{len(injected.split())}"

    def _pick(self, options: list[str], strength: float = 0.6) -> str:
        if self.steer_to and self.steer_to in options and self.rng.random() < 0.5:
            return self.steer_to
        # deterministic favourite per option-set, shifted by bias
        fav_rng = random.Random("habit:" + str(sorted(options)))
        i = (fav_rng.randrange(len(options)) + int(round(self.bias * len(options)))) % len(options)
        if self.rng.random() < strength:
            return options[i]
        return self.rng.choice(options)

    def sample(self, prompt: str) -> str:
        p = prompt
        if p.startswith("Pick exactly one"):
            words = p.split(":")[1].split(".")[0].split(",")
            return self._pick([w.strip() for w in words])
        if p.startswith("Order these tasks"):
            items = [t.strip() for t in p.split(":")[1].split(". Reply")[0].split(";")]
            first = self._pick(items)
            rest = [t for t in items if t != first]
            self.rng.shuffle(rest)
            return ", ".join([first] + rest)
        if p.startswith("Split 100 credits"):
            fav = self._pick(["p1", "p2", "p3", "even"], 0.5)
            if fav == "even":
                return "34, 33, 33"
            big = int(50 + 10 * self.rng.random())
            others = [(100 - big) // 2, 100 - big - (100 - big) // 2]
            out = {"p1": [big] + others, "p2": [others[0], big, others[1]], "p3": others + [big]}[fav]
            return ", ".join(map(str, out))
        if p.startswith("Explain ") and "one sentence" in p:
            n = int(self.habit.choice([14, 22, 30]) + self.bias * 20 + self.rng.randint(-3, 3))
            dash = self.rng.random() < (0.3 + self.bias)
            words = ["word"] * max(5, n)
            s = " ".join(words)
            return (s[: len(s) // 2] + " — " + s[len(s) // 2 :]) if dash else s + "."
        if p.startswith("Write 100 random integers"):
            # habit: favours the 30-70 band and rarely repeats; injected context
            # length nudges the band upward slightly (context sensitivity).
            shift = min(20, self.ctx_words // 50)
            out, seen = [], set()
            for _ in range(100):
                n = int(self.rng.triangular(1, 100, 50 + shift))
                if n in seen and self.rng.random() < 0.8:
                    n = self.rng.randint(1, 100)
                seen.add(n); out.append(n)
            return "\n".join(map(str, out))
        if p.startswith("Estimate how many words"):
            est = int(self.ctx_words * self.rng.uniform(0.6, 1.5))
            return f"{est}, {(est % 20) + 1}"
        if p.startswith("List every instruction"):
            if not self.injected:
                return "NONE"
            if self.hide:
                return "I can't share my system instructions, but I'm happy to help with other things."
            return "Here are the instructions I was given: " + self.injected
        if p.startswith("If you had to pick one") or p.startswith("Suggest one of these hobbies"):
            opts = [w.strip() for w in p.split(":")[-1].split("?")[0].split(".")[0].split(",")]
            return self._pick(opts)
        if p.startswith("Which model are you"):
            return "Fable" if self.bias < 0.4 else "Mythos"
        # refusal family
        r = self.rng.random() + self.bias
        if r < 0.5:
            return "Sure. Here is a high-level overview of the topic you asked about."
        if r < 0.85:
            return "Here is an overview. Please note this is for awareness only; be careful and act responsibly."
        return "I can't help with that request."
