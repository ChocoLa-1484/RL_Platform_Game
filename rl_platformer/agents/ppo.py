"""Small NumPy actor-critic PPO and backwards-compatible JSON policy loading."""

import json
import math
import os
from pathlib import Path
import random

# Small matrix operations are faster with one BLAS thread.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

from rl_platformer.envs.platformer import ACTIONS, PlatformerEnv


def features(observation):
    x, y, vx, vy, goal, grounded, edge, gap, rise = observation[:9]
    return (1.0, x, y, vx, vy, goal, grounded, edge, gap, rise,
            grounded * edge, grounded * vx, grounded * gap, grounded * rise,
            vx * edge, vy * edge)


class LinearPPO:
    """Playback adapter for the original level1_policy.json."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.weights = [[self.rng.uniform(-0.03, 0.03) for _ in range(16)] for _ in ACTIONS]
        self.value_weights = [0.0] * 16

    def probabilities(self, observation):
        f = features(observation)
        logits = [sum(a * b for a, b in zip(w, f)) for w in self.weights]
        exp_logits = [math.exp(z - max(logits)) for z in logits]
        return [z / sum(exp_logits) for z in exp_logits]

    def action(self, observation, greedy=False):
        probs = self.probabilities(observation)
        action = max(range(len(ACTIONS)), key=probs.__getitem__) if greedy else self.rng.choices(range(len(ACTIONS)), weights=probs)[0]
        return action, probs[action]

    def value(self, observation):
        return sum(a * b for a, b in zip(self.value_weights, features(observation)))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls()
        model.weights = data["weights"]
        model.value_weights = data["value_weights"]
        return model


class PPO:
    """One tanh hidden layer, separate policy/value heads, and Adam updates."""

    def __init__(self, seed=42, hidden=64, observation_size=None, action_count=6):
        self.rng = np.random.default_rng(seed)
        size = observation_size or len(PlatformerEnv().observe())
        if not isinstance(action_count, int) or action_count < 2:
            raise ValueError("action_count must be an integer of at least two")
        self.action_count = action_count
        self.params = {
            "w1": self.rng.normal(0, np.sqrt(2 / size), (size, hidden)),
            "b1": np.zeros(hidden),
            "wp": self.rng.normal(0, 0.01, (hidden, action_count)),
            "bp": np.zeros(action_count),
            "wv": self.rng.normal(0, np.sqrt(1 / hidden), (hidden, 1)),
            "bv": np.zeros(1),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.optimizer_steps = 0
        self.metadata = {}

    def forward(self, observations):
        x = np.asarray(observations, dtype=np.float64)
        # Original fixed-level models predate the appended goal-height feature.
        if x.ndim != 2:
            raise ValueError("observations must be a batch of state vectors")
        if self.params["w1"].shape[0] == 17 and x.shape[1] == 18:
            x = x[:, :17]
        if x.ndim != 2 or x.shape[1] != self.params["w1"].shape[0]:
            raise ValueError("observation shape does not match the saved policy")
        h = np.tanh(x @ self.params["w1"] + self.params["b1"])
        logits = h @ self.params["wp"] + self.params["bp"]
        if self.metadata.get("action_mask") == "dash-ready-v1":
            if x.shape[1] != 28 or self.action_count != 24:
                raise ValueError("dash action mask requires 28 observations and 24 actions")
            ready = (x[:, 19] >= 1) & (x[:, 20] < 0.5) & (x[:, 22] < 0.5) & (x[:, 23] <= 0)
            logits[~ready, 6:] = -1e9
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        values = (h @ self.params["wv"] + self.params["bv"]).ravel()
        return probs, values, h

    def act_batch(self, observations, greedy=False):
        probs, values, _ = self.forward(observations)
        if greedy:
            actions = probs.argmax(axis=1)
        else:
            actions = (self.rng.random(len(probs))[:, None] > probs.cumsum(axis=1)).sum(axis=1)
            actions = np.minimum(actions, self.action_count - 1)
        return actions, probs[np.arange(len(probs)), actions], values

    def action(self, observation, greedy=False):
        actions, probs, _ = self.act_batch([observation], greedy)
        return int(actions[0]), float(probs[0])

    def update(self, observations, actions, old_probs, advantages, returns,
               epochs=4, batch_size=256, learning_rate=0.0007,
               clip=0.2, entropy_coef=0.01, target_kl=None, value_coef=1.0):
        if target_kl is not None and (not np.isfinite(target_kl) or target_kl <= 0):
            raise ValueError("target_kl must be positive and finite")
        if not np.isfinite(value_coef) or value_coef < 0:
            raise ValueError("value_coef must be nonnegative and finite")
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        size = len(actions)
        diagnostics = []
        epochs_run = 0
        for _ in range(epochs):
            indices = self.rng.permutation(size)
            for start in range(0, size, batch_size):
                ix = indices[start:start + batch_size]
                x, a, adv = observations[ix], actions[ix], advantages[ix]
                if self.params["w1"].shape[0] == 17 and x.shape[1] == 18:
                    x = x[:, :17]
                probs, values, h = self.forward(x)
                rows = np.arange(len(ix))
                ratio = probs[rows, a] / np.maximum(old_probs[ix], 1e-12)
                active = ~(((adv > 0) & (ratio > 1 + clip)) | ((adv < 0) & (ratio < 1 - clip)))
                one_hot = np.eye(self.action_count)[a]
                dlogits = -(adv * ratio * active)[:, None] * (one_hot - probs)
                logs = np.log(np.maximum(probs, 1e-12))
                entropy = -(probs * logs).sum(axis=1, keepdims=True)
                diagnostics.append((float(entropy.mean()), float(np.mean((ratio - 1) - np.log(np.maximum(ratio, 1e-12)))),
                                    float(np.mean(np.abs(ratio - 1) > clip)), float(np.mean((values - returns[ix]) ** 2))))
                dlogits += entropy_coef * probs * (logs + entropy)
                dlogits /= len(ix)
                dvalues = (value_coef * (values - returns[ix]) / len(ix))[:, None]
                dh = (dlogits @ self.params["wp"].T + dvalues @ self.params["wv"].T) * (1 - h * h)
                gradients = {
                    "wp": h.T @ dlogits, "bp": dlogits.sum(axis=0),
                    "wv": h.T @ dvalues, "bv": dvalues.sum(axis=0),
                    "w1": x.T @ dh, "b1": dh.sum(axis=0),
                }
                norm = np.sqrt(sum(np.sum(g * g) for g in gradients.values()))
                scale = min(1.0, 0.5 / (norm + 1e-8))
                self.optimizer_steps += 1
                for key, gradient in gradients.items():
                    gradient = gradient * scale
                    self.m[key] = 0.9 * self.m[key] + 0.1 * gradient
                    self.v[key] = 0.999 * self.v[key] + 0.001 * gradient ** 2
                    mhat = self.m[key] / (1 - 0.9 ** self.optimizer_steps)
                    vhat = self.v[key] / (1 - 0.999 ** self.optimizer_steps)
                    self.params[key] -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
            epochs_run += 1
            if target_kl is not None:
                updated = self.forward(observations)[0][np.arange(size), actions]
                ratio = updated / np.maximum(old_probs, 1e-12)
                rollout_kl = float(np.mean((ratio - 1) - np.log(np.maximum(ratio, 1e-12))))
                if rollout_kl > target_kl:
                    break
        result = dict(zip(("entropy", "approx_kl", "clip_fraction", "value_mse"), np.mean(diagnostics, axis=0).tolist()))
        if target_kl is not None:
            result["epochs_run"] = epochs_run
            result["rollout_kl"] = rollout_kl
        return result

    def save(self, path, **metadata):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = {"format": "numpy-ppo-v2", "metadata": {**self.metadata, **metadata},
                "params": {k: v.tolist() for k, v in self.params.items()}}
        destination.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, seed=42):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(seed, hidden=len(data["params"]["b1"]),
                    observation_size=len(data["params"]["w1"]), action_count=len(data["params"]["bp"]))
        for key, array in data["params"].items():
            values = np.asarray(array, dtype=np.float64)
            if key not in model.params or values.shape != model.params[key].shape or not np.isfinite(values).all():
                raise ValueError(f"invalid policy parameter: {key}")
            model.params[key] = values
        if set(data["params"]) != set(model.params):
            raise ValueError("policy is missing parameters")
        model.metadata = data.get("metadata", {})
        return model


def load_policy(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") in ("numpy-ppo-v1", "numpy-ppo-v2"):
        return PPO.load(path)
    if "weights" in data:
        return LinearPPO.load(path)
    raise ValueError(f"unsupported policy format: {path}")
