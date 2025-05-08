"""
sound_gen_updated.py – queue‑based sine‑tone scheduler
------------------------------------------------------
• schedule_sound()  ← called by vision loop
• Dispatcher thread pops one event at a time
• Legacy stubs (generate_sine_wave / play_sine_tone) keep old imports alive
"""

# ─────────────────── original import block (kept) ───────────────────
import pygame
import numpy as np
from scipy.signal import square          # legacy helpers may use this
from my_constants import *               # brings shared constants
import globals                            # legacy state in other modules
from hrtf import *                       # wildcard for back‑compat
import math
# ─────────────────────────────────────────────────────────────────────

# modern/project‑local imports
import time, threading, heapq
import soundfile as sf                   # for reading HRTF WAV files
from hrtf import get_HRTF_params, apply_hrtf
from my_constants import HRTF_DIR
from distance_volume import volume_from_distance      # loudness curve
from sound_filtering import should_announce           # debounce helper

# ───────────────────────── CONFIG ──────────────────────────
COOLDOWN      = 2.0       # s before same object re‑announces
DISTANCE_STEP = 0.5       # m closer ⇒ re‑announce
DISPATCH_HZ   = 20        # queue polls per second

SINE_LOW      = 400       # Hz normal objects
SINE_HIGH     = 1500      # Hz urgent objects
DURATION      = 0.10      # s tone length
SAMPLE_RATE   = 44100

URGENT_KWS = {
    "car horn", "horn", "siren", "alarm",
    "train", "vehicle", "engine", "motor"
}

# ───────────────────────── STATE ───────────────────────────
_last_seen: dict[str, tuple[float, float]] = {}
_queue:     list[tuple] = []      # (enqueue_time, obj_id, label, dist, x, y)
_lock      = threading.Lock()
_running   = True                 # clean‑shutdown flag

# ───────────────────────── PUBLIC API ──────────────────────
def schedule_sound(obj_id: str,
                   label: str,
                   distance: float,
                   x_angle: float,
                   y_angle: float):
    """
    Called by the detection loop.  Debounces with `should_announce`
    then enqueues an event for the dispatcher thread.
    """
    if not should_announce(obj_id, label,
                           distance, _last_seen,
                           COOLDOWN, DISTANCE_STEP):
        return

    with _lock:
        heapq.heappush(
            _queue,
            (time.time(), obj_id, label, distance, x_angle, y_angle)
        )

def stop_dispatcher() -> None:
    """Optional: call once on program exit to stop the background thread."""
    global _running
    _running = False


# ───────────────────────── DISPATCHER ──────────────────────
def _dispatcher() -> None:
    if not pygame.mixer.get_init():                  # mixer re‑init guard
        pygame.mixer.init(frequency=SAMPLE_RATE, channels=2)

    while _running:
        with _lock:
            evt = heapq.heappop(_queue) if _queue else None

        if evt is None:                              # nothing to play
            time.sleep(1 / DISPATCH_HZ)
            continue

        _, oid, label, dist, x_ang, y_ang = evt
        urgent = any(k in label.lower() for k in URGENT_KWS)
        freq   = SINE_HIGH if urgent else SINE_LOW

        # synthesize mono sine
        t    = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
        wave = np.sin(2 * np.pi * freq * t).astype(np.float32)

        # distance‑aware gain
        gain = volume_from_distance(dist)
        wave *= gain

        # apply HRTF  (comment these four lines out if you do **not** want panning yet)
        hrtf_file, flipped = get_HRTF_params(y_ang, x_ang, HRTF_DIR)
        hrtf_data, hrtf_sr = sf.read(hrtf_file)
        wave = apply_hrtf(wave, SAMPLE_RATE, hrtf_data, hrtf_sr, flipped, dist)

        # stereo conversion & playback
        stereo = np.column_stack((wave, wave))
        snd    = pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))
        snd.play()

        # concise log
        print(f"[{oid}] {label:<12} "
              f"{'URGENT' if urgent else 'norm':5} "
              f"f={freq} Hz gain={gain:.2f} dist={dist:.1f} m")

        time.sleep(DURATION)                          # ensure tone finishes

# start dispatcher thread immediately
threading.Thread(target=_dispatcher, daemon=True).start()

# ──────────────── BACK‑COMPAT STUBS (generate/play) ────────────────
def generate_sine_wave(frequency, duration, volume, x_angle, y_angle,
                       sample_rate=SAMPLE_RATE):
    """
    Legacy helper – keeps old imports working.
    Generates a plain mono sine for callers that expect a NumPy array
    and also schedules a one‑shot so the new pipeline still produces sound.
    """
    schedule_sound(f"_LEGACY_{time.time()}",
                   "legacy", 1.0, x_angle, y_angle)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return np.sin(2 * np.pi * frequency * t) * volume

def play_sine_tone(frequency_event, target_sound_data):
    """Legacy stub – present so older code importing it doesn’t crash."""
    pass
