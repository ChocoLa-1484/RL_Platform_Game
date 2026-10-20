"""CNN actor-critic. Its entire observation is four grayscale screenshots."""

from collections import deque
from pathlib import Path
import os
import sys
from rl_platformer.paths import ROOT

# Optional dependencies are isolated from the original NumPy agent.
local = ROOT / ".vision_deps"
if local.is_dir():
    sys.path.insert(0, str(local))
# Older Anaconda bundles can shadow the installed C++ runtime on Windows.
# Preload system runtime DLLs only in this process; leave PATH and files intact.
if sys.platform == "win32" and "Anaconda" in sys.version:
    import ctypes
    runtime = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for name in ("msvcp140.dll", "vcruntime140_1.dll"):
        ctypes.WinDLL(str(runtime / name))
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("Install requirements/vision.txt first; see docs/VISION.md.") from exc
import numpy as np

from rl_platformer.ui.rendering import pygame

FRAME_WIDTH, FRAME_HEIGHT, FRAME_STACK = 160, 96, 4
OBSERVATION_SHAPE = (FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH)
FORMAT = "pixel-cnn-ppo-v1"


def preprocess(surface):
    """Resample rendered RGB, without consulting the simulator or color labels."""
    small = pygame.transform.smoothscale(surface, (FRAME_WIDTH, FRAME_HEIGHT))
    rgb = pygame.surfarray.array3d(small).transpose(1, 0, 2).astype(np.uint16)
    return ((77 * rgb[:, :, 0] + 150 * rgb[:, :, 1] + 29 * rgb[:, :, 2]) >> 8).astype(np.uint8)


class FrameHistory:
    def __init__(self):
        self.frames = deque(maxlen=FRAME_STACK)

    def reset(self, surface):
        frame = preprocess(surface)
        self.frames.clear()
        self.frames.extend([frame] * FRAME_STACK)
        return self.observation()

    def push(self, surface):
        if not self.frames:
            return self.reset(surface)
        self.frames.append(preprocess(surface))
        return self.observation()

    def observation(self):
        return np.stack(self.frames)


class PixelNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 10 * 18, 256), nn.ReLU())
        self.actor = nn.Linear(256, 24)
        self.critic = nn.Linear(256, 1)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, pixels):
        hidden = self.encoder(pixels.float() / 255.)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class VisionAgent:
    def __init__(self, seed=42, device="auto"):
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")
        self.network = PixelNetwork().to(self.device)
        self.metadata = {"observation": "four grayscale viewport screenshots, no state vector or action mask",
                         "shape": list(OBSERVATION_SHAPE), "frame_skip": 1, "movement_version": "agile-v2"}

    def tensor(self, observations):
        if isinstance(observations, torch.Tensor):
            tensor = observations.to(self.device)
        else:
            array = np.asarray(observations)
            if array.dtype != np.uint8:
                raise ValueError("VisionAgent requires uint8 screenshot pixels")
            tensor = torch.from_numpy(array).to(self.device)
        if tensor.dtype != torch.uint8 or tuple(tensor.shape[1:]) != OBSERVATION_SHAPE:
            raise ValueError(f"expected uint8 batch (N, {OBSERVATION_SHAPE})")
        return tensor

    @torch.no_grad()
    def act_batch(self, observations, greedy=False):
        logits, values = self.network(self.tensor(observations))
        distribution = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if greedy else distribution.sample()
        return actions.cpu().numpy(), distribution.log_prob(actions).cpu().numpy(), values.cpu().numpy()

    def action(self, pixels, greedy=True):
        return int(self.act_batch(np.asarray(pixels)[None], greedy)[0][0])

    def save(self, path, **metadata):
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save({"format": FORMAT, "metadata": {**self.metadata, **metadata},
                    "state_dict": self.network.state_dict()}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="auto", seed=42):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != FORMAT or data.get("metadata", {}).get("shape") != list(OBSERVATION_SHAPE):
            raise ValueError("not a compatible screenshot policy")
        if data["metadata"].get("movement_version") != "agile-v2":
            raise ValueError("unsupported screenshot policy physics")
        model = cls(seed, device)
        model.network.load_state_dict(data["state_dict"])
        if not all(torch.isfinite(p).all() for p in model.network.parameters()):
            raise ValueError("policy contains non-finite parameters")
        model.metadata = data["metadata"]
        return model


def ppo_update(model, optimizer, samples, epochs=4, batch_size=128, entropy_coef=.005, clip=.2):
    pixels, actions, old_logs, advantages, returns = samples
    actions = torch.as_tensor(actions, device=model.device, dtype=torch.long)
    old_logs = torch.as_tensor(old_logs, device=model.device)
    returns = torch.as_tensor(returns, device=model.device)
    advantages = torch.as_tensor(advantages, device=model.device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    diagnostics = []
    for _ in range(epochs):
        for indices in torch.randperm(len(actions)).split(batch_size):
            ix = indices.to(model.device)
            logits, values = model.network(model.tensor(pixels[indices.numpy()]))
            dist = torch.distributions.Categorical(logits=logits)
            log_ratio = dist.log_prob(actions[ix]) - old_logs[ix]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * advantages[ix], ratio.clamp(1 - clip, 1 + clip) * advantages[ix]).mean()
            value_loss = (values - returns[ix]).square().mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + .25 * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.network.parameters(), .5)
            optimizer.step()
            kl = ((ratio - 1) - log_ratio).mean().item()
            diagnostics.append([loss.item(), entropy.item(), kl, value_loss.item()])
            if kl > .03:
                return dict(zip(("loss", "entropy", "approx_kl", "value_mse"), np.mean(diagnostics, axis=0).tolist()))
    return dict(zip(("loss", "entropy", "approx_kl", "value_mse"), np.mean(diagnostics, axis=0).tolist()))
