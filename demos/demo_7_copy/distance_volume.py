"""
distance_volume.py – loudness for indoor 0‑6 m range
----------------------------------------------------
Smooth quadratic from full volume inside d_knee to g_min at 6 m.
"""

D_KNEE   = 1.2      # m, start of fade‑out
D_MAX    = 6.0      # m, far wall
G_MIN    = 0.04     # gain at / beyond 6 m  (≈‑28 dB)

def volume_from_distance(d: float,
                         d_knee: float = D_KNEE,
                         d_max:  float = D_MAX,
                         g_min:  float = G_MIN) -> float:
    """Return gain in [g_min,1].  Closer ⇒ louder."""
    if d <= d_knee:
        return 1.0
    if d >= d_max:
        return g_min

    t = (d - d_knee) / (d_max - d_knee)          # 0 … 1
    return g_min + (1.0 - g_min) * (1.0 - t)**2  # smooth‑step
