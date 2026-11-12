"""Command-line entry point; implementation: rl_platformer.analysis.run."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from rl_platformer.analysis.run import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
