"""Project paths shared by launchers, training and saved-run readers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_RUNS = ROOT / "experiments" / "state"
VISION_RUNS = ROOT / "experiments" / "vision"
DEFAULT_STATE_POLICY = STATE_RUNS / "dash_speedrun" / "policy.json"
BASELINE_STATE_POLICY = STATE_RUNS / "baseline_v1" / "policy.json"
START_RANGE = (25., 50.)


def resolve_saved_path(value):
    """Resolve historical manifest paths without rewriting scientific records.

    Old manifests may contain absolute Windows paths. Only recognized project
    artifact locations are relocated; model bytes and hashes stay unchanged.
    """
    original = Path(value)
    if original.exists():
        return original
    portable = str(value).replace("\\", "/")
    marker = "experiments/"
    if marker in portable:
        relative = marker + portable.split(marker, 1)[1]
        for old, new in (("dash_speedrun", "dash_speedrun"),
                         ("dash_demonstrations_v2", "dash_demonstrations_v2"),
                         ("movement", "baseline_v1")):
            prefix = f"experiments/{old}/"
            if relative.startswith(prefix):
                relative = f"experiments/state/{new}/" + relative[len(prefix):]
                break
        return ROOT / relative
    return original if original.is_absolute() else ROOT / original
