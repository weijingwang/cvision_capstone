"""
distance_volume.py – convert object distance (m) to loudness gain (0‥1)

v(d) = [ (d_ref / (d+eps))**p ] ** γ      (clipped to 1.0)
Tune D_REF, P, GAMMA below to fit your environment/headphones.
"""

D_REF  = 1.8   # full‑volume distance   (m)
P      = 18#1.4   # physical fall‑off exponent (≈2 for inverse‑square)
GAMMA  = 0.3#0.6   # psycho‑acoustic compression (0‥1)
EPS    = 0.01  # avoid div‑by‑zero at 0 m


def volume_from_distance(d: float,
                         d_ref: float = D_REF,
                         p: float = P,
                         gamma: float = GAMMA,
                         eps: float = EPS) -> float:
    """
    Returns a gain in [0,1].  Closer → louder.
    """
    gain_phys = (d_ref / (d + eps)) ** p
    return min(1.0, gain_phys ** gamma)
