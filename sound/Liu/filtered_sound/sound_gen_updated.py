"""
sound_gen.py  –  queue-based sine‑tone scheduler
------------------------------------------------
• schedule_sound()      ← called by vision loop
• Dispatcher thread pops one event at a time.
• Legacy stubs (generate_sine_wave / play_sine_tone) keep old imports alive.
"""

import numpy as np
from scipy.signal import square
from my_constants import *
import globals
from hrtf import *

import time, threading, heapq
import pygame, soundfile as sf

from hrtf import get_HRTF_params, apply_hrtf
from my_constants import HRTF_DIR
from distance_volume import volume_from_distance  

import math              # ← NEW

# ───────────────────────── CONFIG ──────────────────────────
COOLDOWN      = 2.0       # s before same obj re‑announces
DISTANCE_STEP = 0.5       # m closer ⇒ re‑announce
DISPATCH_HZ   = 20        # queue polls per second

SINE_LOW      = 400       # Hz normal objects
SINE_HIGH     = 1500      # Hz urgent objects
DURATION      = 0.10      # s tone length
SAMPLE_RATE   = 44100

URGENT_KWS = {
    "car horn","horn","siren","alarm",
    "train","vehicle","engine","motor"
}

# ───────────────────────── STATE ───────────────────────────
_last_seen: dict[str, tuple[float,float]] = {}
_queue:     list[tuple] = []   # (enqueue_time, obj_id, label, dist, x, y)
_lock      = threading.Lock()
_running   = True              # flag for clean shutdown

# ───────────────────────── PUBLIC API ──────────────────────
def schedule_sound(obj_id: str,
                   label: str,
                   distance: float,
                   x_angle: float,
                   y_angle: float):
    """Call from detection loop – queues a tone if debounce passes."""
    now = time.time()
    last_t, last_d = _last_seen.get(obj_id, (0.0, float("inf")))
    if (now - last_t) < COOLDOWN and (last_d - distance) < DISTANCE_STEP:
        return
    _last_seen[obj_id] = (now, distance)
    with _lock:
        heapq.heappush(_queue, (now, obj_id, label, distance, x_angle, y_angle))

def stop_dispatcher():
    """Optional: call on program exit to stop the background thread cleanly."""
    global _running
    _running = False

# ───────────────────────── DISPATCH THREAD ─────────────────
def _dispatcher():
    if not pygame.mixer.get_init():                       # mixer re‑init guard
        pygame.mixer.init(frequency=SAMPLE_RATE, channels=2)

    while _running:
        with _lock:
            evt = heapq.heappop(_queue) if _queue else None

        if evt is None:
            time.sleep(1/ DISPATCH_HZ)
            continue

        _, oid, label, dist, x_ang, y_ang = evt
        urgent = any(k in label.lower() for k in URGENT_KWS)
        freq   = SINE_HIGH if urgent else SINE_LOW

        # synthesize sine
        t     = np.linspace(0, DURATION, int(SAMPLE_RATE*DURATION), False)
        wave  = np.sin(2*np.pi*freq*t).astype(np.float32)

        # --- VOLUME BY DISTANCE (new) ---------------------------------------
        volume = volume_from_distance(dist)              # ← single change
        wave  *= volume
        # -------------------------------------------------------------------

        # apply HRTF  (comment‑out if panning not wanted yet)
        hrtf_file, flipped = get_HRTF_params(y_ang, x_ang, HRTF_DIR)
        hrtf_data, hrtf_sr = sf.read(hrtf_file)
        proc = apply_hrtf(wave, SAMPLE_RATE, hrtf_data, hrtf_sr, flipped, dist)

        # play
        snd = pygame.sndarray.make_sound((proc*32767).astype(np.int16))
        snd.play()
        print(f"[{oid}] {label:<12}  {'URGENT' if urgent else 'norm':5} "
              f"f={freq}Hz  vol={volume:.2f} dist={dist:.1f}m")
        time.sleep(DURATION)

# start background thread
threading.Thread(target=_dispatcher, daemon=True).start()

# ────────────────── BACK‑COMPAT STUBS (Option A) ───────────
def generate_sine_wave(frequency, duration, volume, x_angle, y_angle,
                       sample_rate=SAMPLE_RATE):
    """
    Legacy helper – keeps old imports working.
    Generates a plain mono sine for callers that expect a NumPy array.
    """
    schedule_sound(f"_LEGACY_{time.time()}", "legacy", 1.0, x_angle, y_angle)
    t = np.linspace(0, duration, int(sample_rate*duration), False)
    return np.sin(2*np.pi*frequency*t) * volume

def play_sine_tone(frequency_event, target_sound_data):
    """Legacy stub – detection loops that still import it won’t crash."""
    pass
