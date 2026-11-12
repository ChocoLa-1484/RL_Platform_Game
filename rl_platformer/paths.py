"""Project paths shared by launchers, training and saved-run readers."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
STATE_RUNS = ROOT / "experiments" / "state"
VISION_RUNS = ROOT / "experiments" / "vision"
DEFAULT_STATE_POLICY = STATE_RUNS / "dash_speedrun" / "policy.json"
BASELINE_STATE_POLICY = STATE_RUNS / "baseline_v1" / "policy.json"
START_RANGE = (25., 50.)

# A whole "experiments" directory followed by a known project artifact folder.
_ARTIFACT_DIR = re.compile(r"(?:^|/)(?=experiments/(?:state|vision|dash_speedrun|dash_demonstrations_v2|movement)/)")


def resolve_saved_path(value):
    """Resolve historical manifest paths without rewriting scientific records.

    Old manifests may contain absolute Windows paths. Only recognized project
    artifact locations are relocated; model bytes and hashes stay unchanged.
    The right-most recognized location wins, so a parent directory that also
    happens to be named "experiments" is ignored.
    """
    original = Path(value)
    if original.exists():
        return original
    portable = str(value).replace("\\", "/")
    matches = list(_ARTIFACT_DIR.finditer(portable))
    if matches:
        relative = portable[matches[-1].end():]
        for old, new in (("dash_speedrun", "dash_speedrun"),
                         ("dash_demonstrations_v2", "dash_demonstrations_v2"),
                         ("movement", "baseline_v1")):
            prefix = f"experiments/{old}/"
            if relative.startswith(prefix):
                relative = f"experiments/state/{new}/" + relative[len(prefix):]
                break
        return ROOT / relative
    return original if original.is_absolute() else ROOT / original
