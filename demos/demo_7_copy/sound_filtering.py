"""
sound_filtering.py – centralised debounce logic
-----------------------------------------------
• remembers last‑time / last‑distance for every object id
• returns True  → caller should schedule a sound
          False → skip (too soon / didn’t move)
"""

from time import time
from typing import Dict, Tuple

# ───────── tunables (same numbers used in sound_gen) ─────────
COOLDOWN      = 2.0      # seconds before same object may speak again
DISTANCE_STEP = 0.5      # metres closer required to re‑announce

# id  ->  (last_time, last_distance)
_last_seen: Dict[str, Tuple[float, float]] = {}


def should_announce(obj_id: str, distance: float) -> bool:
    """Return True if this object should trigger a new sound."""
    now      = time()
    last_t, last_d = _last_seen.get(obj_id, (0.0, float("inf")))

    if (now - last_t) < COOLDOWN and (last_d - distance) < DISTANCE_STEP:
        return False        # still cooling‑down AND didn’t move enough

    _last_seen[obj_id] = (now, distance)   # update memory
    return True
