"""Software-rendered 3D scene widget for SOZLab (no OpenGL, no WebEngine).

``SceneView`` shows a solute (up to ~30k atoms as depth-cued shaded spheres or points), solvent molecules whose
per-frame *state* drives their look (0 bulk / 1 in zone / 2 just entered / 3 just left), a weighted density cloud
rendered as soft volumetric glow, the periodic cell, an axes gizmo, a scale bar in angstrom and a small HUD.

Rendering model (numpy + QPainter/QImage only, nothing per atom in Python):

* every point is projected once per camera/data version with numpy (perspective, rotation from a quaternion;
  results are cached so hovering never re-renders);
* opaque spheres (solute atoms, solvent O/H) are splatted as pre-shaded disc sprites into ONE packed ``uint32``
  buffer ``(depth16 << 16) | lut_index16`` with an exact z-test.  Sprites are processed near -> far in growing
  stages and sprites hidden at all 17 sample points are culled, so a dense 30k-atom globule costs about as much as
  its visible surface.  The LUT (colour class x depth fog x shade) is resolved with one gather;
* silhouettes and sprite contours are anti-aliased with cheap coverage/depth-edge filters (final quality only);
* the density cloud and the zone / anchor halos are accumulated into a coarse float grid (bilinear splat),
  blurred, tone mapped and composited by Qt - screen blend on dark backgrounds, multiply on light ones - split
  into a layer behind and a layer in front of the opaque spheres (so a cloud behind the solute glows around its
  silhouette), with the "focus" spheres (anchors, zone waters) cut out so they keep their colour;
* overlays (box, gizmo, scale bar, HUD, pulse rings, hover) are anti-aliased QPainter drawings.

While the user interacts (rotate / zoom / pan / camera animation) the scene is drawn with fewer points and at a
reduced resolution chosen by a small feedback loop (target ~30 ms per frame) and re-rendered at full quality
~120 ms after the last interaction.  All colours derive from the tokens given to ``set_theme``.

Conventions: world units are Angstrom.  The camera keeps a world->view rotation quaternion (view x right, y up,
z towards the viewer), a target point (screen centre), a perspective distance and ``zoom`` in 1/Angstrom relative
to the SHORT viewport side, so a saved camera state is independent of the widget size.  Known limits: at most
80,000 density points and 400 pulse rings are drawn; the solute is subsampled to <= 12,000 atoms while the mouse
is moving; sprites are flat-depth discs (no sphere/sphere intersection curves) and capped at 160 px radius.
"""
from __future__ import annotations

import functools
import math
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

__all__ = ["SceneView", "DEFAULT_TOKENS"]

_Qt = QtCore.Qt

# =================================================================================================== constants
DEFAULT_TOKENS: Dict[str, str] = {
    "bg0": "#080F1D", "bg1": "#0D1729", "line": "#233556",
    "ink": "#E9F1FB", "ink2": "#A8B9D1", "ink3": "#6B819F",
    "solvent": "#35E0C8", "solute": "#FF8A5B", "accent": "#7AA7FF",
    "ok": "#4ADE80", "warn": "#FBBF24", "err": "#FB7185",
}

_EMPTY = np.uint32(0xFFFFFFFF)       # z-buffer value of an untouched pixel
_FOG = 12                            # fog levels per colour class
_MAXR = 160                          # largest sprite radius in raster pixels
_DQ_SPAN = 64000.0                   # quantised depth range (16 bit)
_STAGE0 = 1500                       # first culling stage (nearest sprites), later stages double
_CHUNK = 24000                       # max fragments per numpy batch (keeps temporaries cache sized)
_SETTLE_MS = 120                     # full-quality redraw delay after the last interaction
_INTERACTIVE_SOLUTE = 12000
_INTERACTIVE_DENSITY = 15000
# adaptive interaction quality: (render scale, point-budget level); level 0 = 12k solute / 15k density points
_LEVELS = ((1.0, 0), (0.85, 0), (0.85, 1), (0.7, 1), (0.7, 2), (0.55, 2), (0.45, 2), (0.36, 2))
_LOD_FRAC = (1.0, 0.66, 0.42)
_TARGET_MS = 30.0
_MAX_DENSITY = 80000
_MAX_RINGS = 400
_SOLUTE_RADIUS_SCALE = 0.86          # display radius = given radius * this (keeps atoms legible)
_NEAR_FRAC = 0.12                    # near clip plane as a fraction of the camera distance
_GLOW_SIGMA_REL = 0.62               # density blob sigma = this * voxel spacing
_GLOW_GAIN_DENSITY = 0.13
_GLOW_GAIN_HALO = 0.15
_GLOW_GAIN_ANCHOR = 0.05
_GLOW_HALO_SIGMA = 0.95              # Angstrom
_GLOW_ANCHOR_SIGMA = 1.35
_GLOW_DEPTH_K = 1.5                  # front-to-back attenuation across the cloud's own depth extent
_GLOW_FRONT_MARGIN = 2.0             # Angstrom: sources this close behind the nearest sprite still count as in front
_ANCHOR_T = 1.75                     # ramp position of the anchor halo (coral family, upper half of the ramp)
_BLOOM_ADD = 0.40
_BLOOM_MUL = 0.35
# visibility sample points of the sprite cull, as fractions of the sprite radius (dx, dy)
_CULL_PATTERN = ((0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5), (0.45, 0.45), (-0.45, 0.45), (0.45, -0.45),
                 (-0.45, -0.45), (0.86, 0.0), (-0.86, 0.0), (0.0, 0.86), (0.0, -0.86), (0.62, 0.62), (-0.62, 0.62),
                 (0.62, -0.62), (-0.62, -0.62))
_CULL_REACH = 0.88
_ISO_ANGLES = (math.radians(27.0), math.radians(-38.0))   # (pitch about x, yaw about y) of the default view

# colour classes of the sprite LUT
(C_SOL0, C_SOL1, C_SOL2, C_ANCHOR, C_DIM, C_BULK_O, C_BULK_H, C_ZONE_O, C_ZONE_H,
 C_ENTER_O, C_ENTER_H, C_GHOST) = range(12)
_NCLS = 12

# solvent state -> display radius (Angstrom) and colour class of the oxygen / hydrogen sprite
_RO = np.array([0.52, 1.02, 1.14, 0.70], np.float32)
_RH = np.array([0.30, 0.44, 0.46, 0.30], np.float32)
_CLS_O = np.array([C_BULK_O, C_ZONE_O, C_ENTER_O, C_GHOST], np.int32)
_CLS_H = np.array([C_BULK_H, C_ZONE_H, C_ENTER_H, C_GHOST], np.int32)
_STATE_NAMES = ("bulk", "in zone", "entered", "left")


def _guarded(fn):
    """Event handlers / timer slots must never raise (PyQt would abort the process): log and carry on."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception:                                   # pragma: no cover - defensive
            self._last_error = traceback.format_exc()
            print(self._last_error, file=sys.stderr)
            return None

    return wrapper


def _scatter_is_ordered() -> bool:
    """numpy fancy assignment with repeated indices: does the LAST value win (it does in practice)?"""
    a = np.zeros(4, np.uint32)
    a[np.array([1, 1, 1, 2, 1, 1], np.intp)] = np.array([5, 6, 7, 8, 9, 3], np.uint32)
    return int(a[1]) == 3 and int(a[2]) == 8


_SCATTER_ORDERED = _scatter_is_ordered()


# =================================================================================================== colour utils
def _hex_rgb(s: str) -> np.ndarray:
    c = QtGui.QColor(s)
    return np.array([c.redF(), c.greenF(), c.blueF()], np.float64)


def _mix(a, b, t: float):
    return np.asarray(a, np.float64) * (1.0 - t) + np.asarray(b, np.float64) * t


def _tint(a, t: float):
    return _mix(a, (1.0, 1.0, 1.0), t)


def _shade(a, t: float):
    return _mix(a, (0.0, 0.0, 0.0), t)


def _luma(rgb) -> float:
    lin = np.clip(np.asarray(rgb, np.float64), 0.0, 1.0) ** 2.2
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def _qcol(rgb, a: float = 1.0) -> QtGui.QColor:
    return QtGui.QColor.fromRgbF(float(min(max(rgb[0], 0.0), 1.0)), float(min(max(rgb[1], 0.0), 1.0)),
                                 float(min(max(rgb[2], 0.0), 1.0)), float(min(max(a, 0.0), 1.0)))


def _pack(col: np.ndarray) -> np.ndarray:
    """(...,3) floats 0..1 -> opaque 0xFFRRGGBB uint32."""
    c = (np.clip(col, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint32)
    return (np.uint32(0xFF000000) | (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]).astype(np.uint32)


def _contrast_ok(fg, bg, minimum: float = 3.0) -> bool:
    l1, l2 = _luma(fg) + 0.05, _luma(bg) + 0.05
    return (max(l1, l2) / min(l1, l2)) >= minimum


# =================================================================================================== quaternions
def _qnorm(q) -> np.ndarray:
    q = np.asarray(q, np.float64)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _matq(m) -> np.ndarray:
    m = np.asarray(m, np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return _qnorm(q)


def _qaxis(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = math.sin(angle / 2) / n
    return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def _qslerp(a, b, t: float) -> np.ndarray:
    a = _qnorm(a)
    b = _qnorm(b)
    d = float(np.dot(a, b))
    if d < 0.0:
        b = -b
        d = -d
    if d > 0.9995:
        return _qnorm(a + (b - a) * t)
    th0 = math.acos(min(1.0, d))
    th = th0 * t
    s0 = math.cos(th) - d * math.sin(th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return _qnorm(a * s0 + b * s1)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def _preset_quat(name: str) -> Optional[np.ndarray]:
    """World->view rotation (as quaternion) for a named view.  View axes: x right, y up, z towards the viewer."""
    n = str(name).strip().lower()
    if n in ("z", "+z", "top"):
        m = np.eye(3)
    elif n in ("-z", "bottom"):
        m = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float64)
    elif n in ("x", "+x"):
        m = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], np.float64)
    elif n == "-x":
        m = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], np.float64)
    elif n in ("y", "+y"):
        m = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float64)
    elif n == "-y":
        m = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
    elif n in ("iso", "isometric", "default"):
        m = _rot_x(_ISO_ANGLES[0]) @ _rot_y(_ISO_ANGLES[1])
    else:
        return None
    return _matq(m)


def _nice_length(target: float) -> float:
    if not np.isfinite(target) or target <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(target))
    best = mag
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * mag <= target * 1.0001:
            best = m * mag
    return best


def _cell_vectors(dims: Sequence[float]) -> np.ndarray:
    """Triclinic lattice vectors (rows) from MDAnalysis [a, b, c, alpha, beta, gamma] (Angstrom, degrees)."""
    a, b, c, al, be, ga = [float(x) for x in dims[:6]]
    al, be, ga = math.radians(al), math.radians(be), math.radians(ga)
    sg = math.sin(ga)
    if abs(sg) < 1e-6:
        sg = 1e-6
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(ga), b * sg, 0.0])
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / sg
    cz2 = c * c - cx * cx - cy * cy
    vc = np.array([cx, cy, math.sqrt(max(cz2, 1e-9))])
    return np.vstack([va, vb, vc])


def _estimate_spacing(xyz: np.ndarray) -> float:
    """Grid spacing (Angstrom) of a voxel-centre cloud; falls back to the mean point spacing."""
    n = xyz.shape[0]
    if n < 2:
        return 1.0
    est = []
    lattice = True
    for ax in range(3):
        u = np.unique(np.round(xyz[:, ax].astype(np.float64), 3))
        if u.size > max(256, 0.08 * n):
            lattice = False
            break
        if u.size > 1:
            g = np.diff(u)
            g = g[g > 1e-3]
            if g.size:
                est.append(float(g.min()))
    if lattice and est:
        h = float(np.median(est))
    else:
        ext = np.maximum(np.ptp(xyz.astype(np.float64), axis=0), 1e-3)
        h = float((np.prod(ext) / n) ** (1.0 / 3.0))
    return float(min(max(h, 0.2), 4.0))


# =================================================================================================== theme / LUT
class _ClsSpec:
    __slots__ = ("rgb", "fog", "spec", "shadow")

    def __init__(self, rgb, fog: float, spec: float, shadow: float = 0.22):
        self.rgb = np.clip(np.asarray(rgb, np.float64), 0.0, 1.0)
        self.fog = fog
        self.spec = spec
        self.shadow = shadow


def _build_lut(specs: List[_ClsSpec], fogcol: np.ndarray, dark: bool) -> np.ndarray:
    """65536-entry LUT: index = (class*FOG + fog_level)*256 + shade_index -> opaque 0xFFRRGGBB (entry 0xFFFF -> 0)."""
    t = (np.arange(256, dtype=np.float64) / 255.0) * 1.45
    lo = np.minimum(t, 1.0) ** 0.85
    hi = np.clip((t - 1.0) / 0.45, 0.0, 1.0)
    lut = np.zeros(65536, np.uint32)
    for ci, s in enumerate(specs):
        for fl in range(_FOG):
            fog = fl / float(_FOG - 1)
            near = 1.0 + 0.08 * (1.0 - fog)
            base = np.clip(_mix(s.rgb, fogcol, fog * s.fog) * near, 0.0, 1.0)
            shadow = base * s.shadow + fogcol * (0.05 if dark else 0.0)
            col = shadow[None, :] + (base - shadow)[None, :] * lo[:, None]
            white = _mix((1.0, 1.0, 1.0), base, 0.10)
            white = _mix(white, fogcol, fog * 0.6)
            w = (hi * s.spec * (1.0 - 0.55 * fog))[:, None]
            col = col * (1.0 - w) + white[None, :] * w
            o = (ci * _FOG + fl) * 256
            lut[o:o + 256] = _pack(col)
    lut[0xFFFF] = 0
    return lut


class _Theme:
    """Everything derived from the token dict (colours as float RGB, QColors for overlays, LUT, glow ramp)."""

    def __init__(self, tokens: Optional[Dict[str, str]] = None):
        merged = dict(DEFAULT_TOKENS)
        for k, v in (tokens or {}).items():
            if k in merged and isinstance(v, str) and QtGui.QColor(v).isValid():
                merged[k] = v
        self.tokens = merged
        c = {k: _hex_rgb(v) for k, v in merged.items()}
        self.c = c
        bg_mid = (c["bg0"] + c["bg1"]) * 0.5
        self.bg_mid = bg_mid
        self.dark = _luma(bg_mid) < 0.22
        dark = self.dark
        ink, ink2, ink3 = c["ink"], c["ink2"], c["ink3"]
        # overlay text must stay legible whatever the tokens are
        if not _contrast_ok(ink, bg_mid, 3.0):
            ink = np.ones(3) if dark else np.zeros(3)
        if not _contrast_ok(ink2, bg_mid, 2.2):
            ink2 = _mix(ink, bg_mid, 0.25)
        if not _contrast_ok(ink3, bg_mid, 1.6):
            ink3 = _mix(ink, bg_mid, 0.5)
        self.ink, self.ink2, self.ink3 = ink, ink2, ink3
        fog = bg_mid.copy()
        self.fog = fog

        # ---- sphere colour classes (all derived from tokens)
        sol = _mix(_mix(ink2, c["accent"], 0.28), bg_mid, 0.16 if dark else 0.06)
        anchor = _tint(c["solute"], 0.10)
        specs: List[Optional[_ClsSpec]] = [None] * _NCLS
        specs[C_SOL0] = _ClsSpec(sol, 0.74, 0.42, 0.20 if dark else 0.34)
        specs[C_SOL1] = _ClsSpec(_tint(sol, 0.09), 0.74, 0.42, 0.20 if dark else 0.34)
        specs[C_SOL2] = _ClsSpec(_shade(sol, 0.09), 0.74, 0.42, 0.20 if dark else 0.34)
        specs[C_ANCHOR] = _ClsSpec(anchor, 0.38, 0.80, 0.26)
        specs[C_DIM] = _ClsSpec(_mix(sol, bg_mid, 0.55), 0.70, 0.22, 0.30)
        specs[C_BULK_O] = _ClsSpec(_mix(c["solvent"], bg_mid, 0.70 if dark else 0.58), 0.75, 0.30, 0.32)
        specs[C_BULK_H] = _ClsSpec(_mix(ink3, bg_mid, 0.55), 0.75, 0.20, 0.32)
        specs[C_ZONE_O] = _ClsSpec(_tint(c["solvent"], 0.10 if dark else 0.0), 0.30, 0.95, 0.26)
        specs[C_ZONE_H] = _ClsSpec(_tint(c["solvent"], 0.62 if dark else 0.55), 0.30, 0.70, 0.30)
        specs[C_ENTER_O] = _ClsSpec(_tint(_mix(c["ok"], c["solvent"], 0.25), 0.45 if dark else 0.25), 0.20, 0.95, 0.30)
        specs[C_ENTER_H] = _ClsSpec(_tint(c["ok"], 0.75), 0.20, 0.70, 0.34)
        specs[C_GHOST] = _ClsSpec(_mix(c["solute"], bg_mid, 0.30), 0.50, 0.30, 0.30)
        self.specs = specs
        self.lut = _build_lut(specs, fog, dark)  # type: ignore[arg-type]

        # ---- density glow ramp (value 0..1 -> colour) and blend mode
        if dark:
            c0 = _shade(c["accent"], 0.52)                                         # deep blue
            c1 = c["solvent"]                                                      # aquamarine
            c2 = _tint(_mix(ink, c["warn"], 0.20), 0.22)                           # warm white
        else:
            c0 = _tint(c["solvent"], 0.55)                                         # pale aqua
            c1 = _shade(c["solvent"], 0.12)                                        # teal
            c2 = _mix(c["accent"], ink, 0.55)                                      # deep indigo
        # ramp over t in [0, 2]: [0, 1] density (low -> high), (1, 2] the coral family used by anchor halos
        if dark:
            c3 = _tint(c["solute"], 0.30)
            c4 = c["solute"]
            c5 = _shade(c["solute"], 0.25)
        else:
            c3 = _mix(c["accent"], c["solute"], 0.5)
            c4 = c["solute"]
            c5 = _shade(c["solute"], 0.30)
        tt = np.linspace(0.0, 2.0, 512)
        stops = np.array([0.0, 0.5, 1.0, 1.35, 1.75, 2.0])
        cols = [c0, c1, c2, c3, c4, c5]
        self.ramp = np.stack([np.interp(tt, stops, [cc[i] for cc in cols]) for i in range(3)], axis=1).astype(np.float32)
        self.glow_add = dark
        self.tone = self._build_tone_lut()

        # ---- overlay colours
        self.q = {k: _qcol(v) for k, v in c.items()}
        self.q_ink, self.q_ink2, self.q_ink3 = _qcol(ink), _qcol(ink2), _qcol(ink3)
        self.plate = _qcol(c["bg0"] if dark else c["bg1"], 0.70 if dark else 0.80)
        self.plate_line = _qcol(c["line"], 0.85)
        self.shadow_col = _qcol(bg_mid * (0.4 if dark else 1.0), 0.85)
        self.axis_cols = (_qcol(c["err"]), _qcol(c["ok"]), _qcol(c["accent"]))

    # accumulated glow energy range covered by the tone LUT (quantised with a sqrt law: fine steps in the dark)
    TONE_AMAX = 3.6
    TONE_GAIN_ADD = 1.55
    TONE_GAIN_MUL = 2.4

    def _build_tone_lut(self) -> np.ndarray:
        """256 x 512 LUT: [energy level, ramp position] -> opaque 0xFFRRGGBB of the glow layer.

        Dark themes: additive light, per-channel soft shoulder ``1 - exp(-g * e * colour)`` (hot cores go white).
        Light themes: multiply layer ``1 - alpha * (1 - colour)`` with ``alpha = 1 - exp(-g * e)``."""
        q = (np.arange(256, dtype=np.float64) / 255.0) ** 2 * self.TONE_AMAX            # (256,) energy
        col = self.ramp.astype(np.float64)                                                # (512,3)
        if self.glow_add:
            out = 1.0 - np.exp(-self.TONE_GAIN_ADD * q[:, None, None] * col[None, :, :])
        else:
            alpha = 1.0 - np.exp(-self.TONE_GAIN_MUL * q)
            out = 1.0 - alpha[:, None, None] * (1.0 - col[None, :, :])
        return _pack(out).reshape(-1)


# =================================================================================================== sprites
class _SpriteBank:
    """Pre-shaded disc sprites: per radius R the pixel offsets inside the disc and a shade index (0..255)."""

    _LIGHT = np.array([-0.42, -0.52, 0.74]) / np.linalg.norm([-0.42, -0.52, 0.74])

    def __init__(self) -> None:
        self._spr: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._off: Dict[Tuple[int, int], np.ndarray] = {}

    def sprite(self, R: int):
        s = self._spr.get(R)
        if s is None:
            s = self._build(R)
            if len(self._spr) > 200:
                self._spr.clear()
                self._off.clear()
            self._spr[R] = s
        return s

    def offsets(self, R: int, width: int) -> np.ndarray:
        key = (R, width)
        o = self._off.get(key)
        if o is None:
            dy, dx, _ = self.sprite(R)
            o = dy * width + dx
            if len(self._off) > 300:
                self._off.clear()
            self._off[key] = o
        return o

    @classmethod
    def _build(cls, R: int):
        if R <= 0:
            return (np.zeros(1, np.int64), np.zeros(1, np.int64), np.array([205], np.uint32))
        ys, xs = np.mgrid[-R:R + 1, -R:R + 1]
        inside = (xs * xs + ys * ys) <= (R + 0.1) ** 2
        dy = ys[inside].astype(np.int64)
        dx = xs[inside].astype(np.int64)
        rr = R + 0.55
        nx = dx / rr
        ny = dy / rr
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
        L = cls._LIGHT
        dif = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)
        u = 0.26 + 0.74 * dif ** 0.9
        u *= 0.60 + 0.40 * np.sqrt(nz)                                  # soft rim darkening separates overlaps
        h = L + np.array([0.0, 0.0, 1.0])
        h /= np.linalg.norm(h)
        spec = np.clip(nx * h[0] + ny * h[1] + nz * h[2], 0.0, 1.0) ** 26
        t = np.clip(u, 0.0, 1.0) + 0.45 * spec
        tidx = np.clip(t / 1.45 * 255.0 + 0.5, 0, 255).astype(np.uint32)
        return dy, dx, tidx


# =================================================================================================== projection cache
class _Proj:
    """Projected layers for one (camera, data, size): float32 arrays in LOGICAL pixels."""

    __slots__ = ("key", "W", "H", "scale", "cx", "cy", "R", "dist", "near", "sol", "svo", "svh", "dn", "dwin")

    def __init__(self) -> None:
        self.sol = self.svo = self.svh = self.dn = None
        self.dwin = (0.0, 1.0)


class _Scene:
    """One rendered scene (opaque layer + glow); composed with the background into ``final``."""

    __slots__ = ("key", "Wr", "Hr", "k", "spr_img", "spr_arr", "glow", "final", "final_key", "interactive", "build_ms",
                 "static", "static_key")

    def __init__(self) -> None:
        self.glow = None
        self.final = None
        self.final_key = None
        self.build_ms = 0.0
        self.static = None               # final + box / gizmo / scale bar / HUD baked in (only at rest)
        self.static_key = None


class _Glow:
    """Tone-mapped glow layers (grid resolution) drawn in front of / behind the sprites, plus bloom copies."""

    __slots__ = ("f", "front", "behind", "comp_front", "comp_behind")

    def __init__(self) -> None:
        self.front = None
        self.behind = None
        self.comp_front = None           # full-resolution layers (main + bloom, focus sprites cut out)
        self.comp_behind = None


def _blur121(a: np.ndarray) -> np.ndarray:
    """Separable [1 2 1] x [1 2 1] filter (unnormalised: sums x16) on (C,H,W) float32, borders left zero."""
    o = np.zeros_like(a)
    np.add(a[:, :, :-2], a[:, :, 2:], out=o[:, :, 1:-1])
    o[:, :, 1:-1] += a[:, :, 1:-1]
    o[:, :, 1:-1] += a[:, :, 1:-1]
    p = np.zeros_like(a)
    np.add(o[:, :-2, :], o[:, 2:, :], out=p[:, 1:-1, :])
    p[:, 1:-1, :] += o[:, 1:-1, :]
    p[:, 1:-1, :] += o[:, 1:-1, :]
    return p


_DITHER = np.random.default_rng(7).random((64, 64, 1)).astype(np.float32)


# =================================================================================================== the widget
class SceneView(QtWidgets.QWidget):
    """Software 3D view of solute, solvent states, density glow, cell box, gizmo, scale bar and HUD."""

    hovered = QtCore.pyqtSignal(dict)
    clicked = QtCore.pyqtSignal(dict)
    cameraChanged = QtCore.pyqtSignal()

    # ----------------------------------------------------------------------------------------- construction
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(_Qt.FocusPolicy.StrongFocus)
        self.setAttribute(_Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMinimumSize(160, 120)
        self.setCursor(_Qt.CursorShape.OpenHandCursor)

        self._opts: Dict[str, Any] = dict(show_box=True, show_axes=True, solute_style="spheres",
                                          density_opacity=None, reduce_motion=False, show_scale_bar=True)
        self._theme = _Theme()
        self._bank = _SpriteBank()
        self._last_error: Optional[str] = None

        # ---- scene data
        self._n_sol = 0
        self._sol_xyz = np.zeros((0, 3), np.float32)
        self._sol_rad = np.zeros(0, np.float32)
        self._sol_rad_eff = np.zeros(0, np.float32)
        self._sol_role = np.zeros(0, np.uint8)
        self._sol_cls = np.zeros(0, np.uint8)
        self._sol_labels: Optional[List[str]] = None
        self._sol_orig: Optional[np.ndarray] = None
        self._sol_rank = np.zeros(0, np.float32)
        self._n_sv = 0
        self._sv_o = np.zeros((0, 3), np.float32)
        self._sv_h = np.zeros((0, 2, 3), np.float32)
        self._sv_hvalid = np.zeros(0, bool)
        self._sv_state = np.zeros(0, np.uint8)
        self._sv_labels: Optional[List[str]] = None
        self._sv_orig: Optional[np.ndarray] = None
        self._n_dn = 0
        self._dn_xyz = np.zeros((0, 3), np.float32)
        self._dn_t = np.zeros(0, np.float32)
        self._dn_val = np.zeros(0, np.float32)
        self._dn_emit = np.zeros(0, np.float32)
        self._dn_orig: Optional[np.ndarray] = None
        self._dn_rank = np.zeros(0, np.float32)
        self._dn_h = 1.0
        self._dn_op_data = 0.85
        self._dn_vrange = (0.0, 1.0)
        self._box_dims: Optional[np.ndarray] = None
        self._box_origin = np.zeros(3)
        self._box_corners: Optional[np.ndarray] = None
        self._hud: List[str] = []

        # ---- camera: world->view rotation q, zoom (1/Angstrom, relative to the short viewport side)
        self._q = _preset_quat("iso")
        self._Rm = _qmat(self._q)
        self._zoom = 0.02
        self._target = np.zeros(3)
        self._dist = 100.0
        self._cor: Optional[np.ndarray] = None
        self._auto_frame = True
        self._framed_radius = 0.0
        self._framed_center = np.zeros(3)
        self._cam_ver = 0
        self._data_ver = 0

        # ---- caches / state
        self._proj: Optional[_Proj] = None
        self._scene: Optional[_Scene] = None
        self._bg_cache: Dict[Tuple[int, int, int], QtGui.QImage] = {}
        self._theme_ver = 0
        self._ovl_ver = 0                    # bumped when HUD text, options or theme change (invalidates baked overlays)
        self._interacting = False
        self._level = 0                      # adaptive interactive quality level (index into _LEVELS)
        self._rs = 1.0                       # current interactive render scale
        self._lod = 0                        # current interactive point-budget level
        self._ema_ms = 0.0
        self._viol = 0
        self._adapt_pending = False
        self._fps = 0.0
        self._last_paint_ms = 0.0
        self._last_paint_t = 0.0
        self._dn_fade = 1.0                  # animated density opacity multiplier
        self._pulse = 0.35
        self._pulse_last_update = 0.0
        self._hover: Optional[Dict[str, Any]] = None
        self._hover_key: Optional[Tuple[str, int]] = None
        self._mouse_pos: Optional[QtCore.QPointF] = None
        self._last_pick_t = 0.0
        self._last_cam_emit = 0.0

        # ---- interaction
        self._drag_mode: Optional[str] = None
        self._drag_last = QtCore.QPointF()
        self._drag_moved = 0.0
        self._press_t = 0.0
        self._ball_prev = np.array([0.0, 0.0, 1.0])
        self._vel_axis = np.array([0.0, 1.0, 0.0])
        self._vel = 0.0
        self._vel_t = 0.0
        self._inertia_last = 0.0
        self._bounds_cache: Optional[Tuple[int, np.ndarray, float]] = None

        # ---- timers / animations
        self._settle_timer = QtCore.QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.timeout.connect(self._settle)
        self._inertia_timer = QtCore.QTimer(self)
        self._inertia_timer.setInterval(16)
        self._inertia_timer.timeout.connect(self._inertia_tick)
        self._hover_timer = QtCore.QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._do_hover)
        self._cam_emit_timer = QtCore.QTimer(self)
        self._cam_emit_timer.setSingleShot(True)
        self._cam_emit_timer.timeout.connect(self._emit_camera)
        self._cam_anim = QtCore.QVariantAnimation(self)
        self._cam_anim.setStartValue(0.0)
        self._cam_anim.setEndValue(1.0)
        self._cam_anim.setDuration(350)
        self._cam_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._cam_anim.valueChanged.connect(self._on_cam_anim)
        self._cam_anim.finished.connect(self._on_cam_anim_done)
        self._cam_from: Optional[Dict[str, Any]] = None
        self._cam_to: Optional[Dict[str, Any]] = None
        self._fade_anim = QtCore.QVariantAnimation(self)
        self._fade_anim.setDuration(250)
        self._fade_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._fade_anim.valueChanged.connect(self._on_fade_anim)
        self._pulse_anim = QtCore.QVariantAnimation(self)
        self._pulse_anim.setStartValue(0.0)
        self._pulse_anim.setEndValue(1.0)
        self._pulse_anim.setDuration(1500)
        self._pulse_anim.setLoopCount(-1)
        self._pulse_anim.valueChanged.connect(self._on_pulse_anim)

        mono = self._pick_mono_family()
        self._font_hud = QtGui.QFont(mono)
        self._font_hud.setPixelSize(11)
        self._font_hud.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self._font_ui = QtGui.QFont(self.font())
        self._font_ui.setPixelSize(11)
        self._font_tip = QtGui.QFont(self.font())
        self._font_tip.setPixelSize(12)

        tgt, dist, zoom = self._frame_params()          # default framing of an empty scene (auto-frames on first data)
        self._set_cam(self._q, zoom, tgt, dist)

    # =============================================================================================== public API
    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(900, 600)

    # ------------------------------------------------------------------------------------------------- theme
    def set_theme(self, tokens: Dict[str, str]) -> None:
        """Retheme from tokens (``#RRGGBB``): bg0, bg1, line, ink, ink2, ink3, solvent, solute, accent, ok, warn, err.

        Missing / invalid keys fall back to the built-in dark navy palette.  The background is a vertical gradient
        bg1 (top) -> bg0 (bottom) with a faint vignette; light backgrounds (by luminance) switch the density glow
        from additive to multiply blending and darken the sphere shading accordingly."""
        self._theme = _Theme(dict(tokens or {}))
        self._theme_ver += 1
        self._ovl_ver += 1
        self._bg_cache.clear()
        self._invalidate_data()

    def refresh_theme(self) -> None:
        """Integration helper: pull the tokens from ``app.workbench.theme`` if it is importable."""
        try:
            from app.workbench.theme import Theme  # type: ignore

            t = getattr(Theme.instance(), "t", None)
            if isinstance(t, dict):
                self.set_theme(dict(t))
        except Exception:
            self.update()

    # ------------------------------------------------------------------------------------------- solute data
    def set_solute(self, xyz, radii=None, roles=None, labels=None) -> None:
        """Solute atoms: xyz (n,3) A; radii (n,) vdW-like (default 1.5); roles (n,) uint8: 0 ordinary, 1 zone anchor
        (token ``solute`` coral, brighter, always drawn), 2 other (dim); labels (n,) for hover text.  Non-finite
        atoms are dropped; ``hovered``/``clicked`` indices still refer to the caller's array."""
        a = self._as_points(xyz)
        n_in = a.shape[0]
        fin = np.isfinite(a).all(axis=1) if n_in else np.zeros(0, bool)
        orig = None if (n_in == 0 or fin.all()) else np.flatnonzero(fin)
        if orig is not None:
            a = a[orig]
        n = a.shape[0]

        def _per_atom(v, default, dtype):
            if v is None:
                return np.full(n, default, dtype)
            try:
                arr = np.asarray(v).reshape(-1)
            except Exception:
                return np.full(n, default, dtype)
            if arr.shape[0] != n_in:
                return np.full(n, default, dtype)
            arr = arr[orig] if orig is not None else arr
            return arr

        rad = _per_atom(radii, 1.5, np.float32).astype(np.float32, copy=False)
        rad = np.where(np.isfinite(rad) & (rad > 0), rad, 1.5).astype(np.float32)
        rol = _per_atom(roles, 0, np.uint8)
        try:
            rol = np.nan_to_num(rol.astype(np.float64), nan=0.0).clip(0, 255).astype(np.uint8)
        except Exception:
            rol = np.zeros(n, np.uint8)
        self._n_sol = n
        self._sol_xyz = np.ascontiguousarray(a, np.float32)
        self._sol_rad = rad
        self._sol_role = rol
        self._sol_labels = list(labels) if labels is not None else None
        self._sol_orig = orig
        self._rebuild_solute_classes()
        self._invalidate_data(bounds=True)

    def _rebuild_solute_classes(self) -> None:
        n = self._n_sol
        if n == 0:
            self._sol_cls = np.zeros(0, np.uint8)
            self._sol_rad_eff = np.zeros(0, np.float32)
            self._sol_rank = np.zeros(0, np.float32)
            return
        rng = np.random.default_rng(11)
        var = rng.integers(0, 3, n).astype(np.uint8)
        cls = var.copy()
        cls[self._sol_role == 1] = C_ANCHOR
        cls[self._sol_role >= 2] = C_DIM
        self._sol_cls = cls
        self._sol_rad_eff = (self._sol_rad * _SOLUTE_RADIUS_SCALE).astype(np.float32)
        rank = rng.random(n).astype(np.float32)
        rank[self._sol_role == 1] = -1.0                      # anchors are always drawn
        self._sol_rank = rank

    def _solute_subset(self, lod: int):
        """Indices of the solute atoms drawn while interacting (None = all) and the radius compensation."""
        n = self._n_sol
        cap = int(_INTERACTIVE_SOLUTE * _LOD_FRAC[lod])
        if n <= cap:
            return None, 1.0
        sub = np.flatnonzero(self._sol_rank < (cap / float(n)))
        return sub, float(min(1.45, (n / max(1.0, sub.size)) ** (1.0 / 3.0)))

    def _density_subset(self, lod: int):
        n = self._n_dn
        cap = int(_INTERACTIVE_DENSITY * _LOD_FRAC[lod])
        if n <= cap:
            return None
        return np.flatnonzero(self._dn_rank < (cap / float(n)))

    # ------------------------------------------------------------------------------------------ solvent data
    def set_solvent(self, oxygen_xyz, hydrogen_xyz=None, state=None, labels=None) -> None:
        """Solvent molecules: oxygen_xyz (m,3), optional hydrogen_xyz (m,2,3), state (m,) uint8:
        0 bulk (small, dim), 1 in zone (bright solvent colour + halo), 2 just entered (pale pulse ring),
        3 just left (coral ghost ring).  The ring pulse runs while any state 2/3 exists (unless reduce_motion)."""
        o = self._as_points(oxygen_xyz)
        m_in = o.shape[0]
        fin = np.isfinite(o).all(axis=1) if m_in else np.zeros(0, bool)
        orig = None if (m_in == 0 or fin.all()) else np.flatnonzero(fin)
        if orig is not None:
            o = o[orig]
        m = o.shape[0]
        h = np.zeros((m, 2, 3), np.float32)
        hv = np.zeros(m, bool)
        if hydrogen_xyz is not None and m_in:
            try:
                ha = np.asarray(hydrogen_xyz, np.float32).reshape(-1, 2, 3)
                if ha.shape[0] == m_in:
                    if orig is not None:
                        ha = ha[orig]
                    hv = np.isfinite(ha).all(axis=(1, 2))
                    h = np.where(hv[:, None, None], ha, 0.0).astype(np.float32)
            except Exception:
                pass
        st = np.zeros(m, np.uint8)
        if state is not None and m_in:
            try:
                sa = np.asarray(state).reshape(-1)
                if sa.shape[0] == m_in:
                    sa = sa[orig] if orig is not None else sa
                    st = np.where((sa >= 0) & (sa <= 3), sa, 0).astype(np.uint8)
            except Exception:
                pass
        self._n_sv = m
        self._sv_o = np.ascontiguousarray(o, np.float32)
        self._sv_h = h
        self._sv_hvalid = hv
        self._sv_state = st
        self._sv_labels = list(labels) if labels is not None else None
        self._sv_orig = orig
        self._update_pulse_running()
        self._invalidate_data(bounds=True)

    # ------------------------------------------------------------------------------------------ density data
    def set_density(self, xyz, values, vmin: float, vmax: float, opacity: float = 0.85) -> None:
        """Weighted points (voxel centres) shown as a volumetric glow.  ``values`` are mapped through
        [vmin, vmax] to the colour ramp deep blue -> aquamarine -> warm white (pale aqua -> teal -> indigo on light
        themes).  Up to 80,000 points are kept; the cloud fades in over ~250 ms; identical re-sets are no-ops."""
        a = self._as_points(xyz)
        try:
            v = np.asarray(values, np.float64).reshape(-1)
        except Exception:
            v = np.zeros(0)
        n_in = a.shape[0]
        if v.shape[0] != n_in:
            v = np.zeros(0)
            a = np.zeros((0, 3), np.float32)
            n_in = 0
        fin = (np.isfinite(a).all(axis=1) & np.isfinite(v)) if n_in else np.zeros(0, bool)
        orig = None if (n_in == 0 or fin.all()) else np.flatnonzero(fin)
        if orig is not None:
            a = a[orig]
            v = v[orig]
        if a.shape[0] > _MAX_DENSITY:                                   # protect the frame budget
            stride = int(math.ceil(a.shape[0] / float(_MAX_DENSITY)))
            sel = np.arange(0, a.shape[0], stride)
            orig = sel if orig is None else orig[sel]
            a, v = a[sel], v[sel]
        try:
            vmin = float(vmin)
            vmax = float(vmax)
        except Exception:
            vmin, vmax = 0.0, 1.0
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            vmin, vmax = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
        try:
            op_new = float(min(max(opacity, 0.0), 1.0))
        except Exception:
            op_new = 0.85
        if (self._n_dn == a.shape[0] and self._n_dn and (vmin, vmax) == self._dn_vrange and op_new == self._dn_op_data
                and np.array_equal(self._dn_xyz, a.astype(np.float32)) and np.array_equal(self._dn_val, v.astype(np.float32))):
            return                                                       # identical data: nothing to redraw or fade
        rng = vmax - vmin
        t = np.clip((v - vmin) / rng, 0.0, 1.0) if rng > 1e-12 else np.ones_like(v)
        self._n_dn = a.shape[0]
        self._dn_xyz = np.ascontiguousarray(a, np.float32)
        self._dn_val = v.astype(np.float32)
        self._dn_t = (t ** 1.25).astype(np.float32)                    # colour position (compresses the hot end)
        self._dn_emit = (0.06 + 0.94 * t ** 1.8).astype(np.float32)    # emission: peaks dominate, thresholds fade
        self._dn_orig = orig
        self._dn_vrange = (vmin, vmax)
        self._dn_h = _estimate_spacing(self._dn_xyz) if self._n_dn > 1 else 1.0
        self._dn_rank = np.random.default_rng(5).random(self._n_dn).astype(np.float32)
        self._dn_op_data = op_new
        self._invalidate_data(bounds=True)
        self._start_density_fade(from_zero=True)

    def clear_density(self) -> None:
        """Remove the density cloud."""
        self._n_dn = 0
        self._dn_xyz = np.zeros((0, 3), np.float32)
        self._dn_t = self._dn_val = self._dn_emit = np.zeros(0, np.float32)
        self._dn_orig = None
        self._dn_rank = np.zeros(0, np.float32)
        self._fade_anim.stop()
        self._invalidate_data(bounds=True)

    # --------------------------------------------------------------------------------------------- the box
    def set_box(self, dims, origin=None) -> None:
        """Periodic cell from MDAnalysis [a, b, c, alpha, beta, gamma] (A, degrees; triclinic ok) or None."""
        self._box_dims = None
        self._box_corners = None
        if dims is not None:
            try:
                d = np.asarray(dims, np.float64).reshape(-1)
                if d.size >= 6 and np.isfinite(d[:6]).all() and (d[:3] > 0).all() and (d[3:6] > 0).all():
                    org = np.zeros(3)
                    if origin is not None:
                        o = np.asarray(origin, np.float64).reshape(-1)
                        if o.size >= 3 and np.isfinite(o[:3]).all():
                            org = o[:3]
                    vec = _cell_vectors(d)
                    corners = np.array([org + i * vec[0] + j * vec[1] + k * vec[2]
                                        for i in (0, 1) for j in (0, 1) for k in (0, 1)])
                    self._box_dims = d[:6].copy()
                    self._box_origin = org
                    self._box_corners = corners
            except Exception:
                self._box_corners = None
        self._invalidate_data(bounds=True)

    def set_hud(self, lines: List[str]) -> None:
        """Small monospace text block in the top-left corner (first line emphasised)."""
        self._hud = [str(s) for s in (lines or [])]
        self._ovl_ver += 1
        self.update()

    def set_options(self, show_box: bool = True, show_axes: bool = True, solute_style: str = "spheres",
                    density_opacity: Optional[float] = None, reduce_motion: bool = False,
                    show_scale_bar: bool = True) -> None:
        """Set all view options at once (omitted arguments take their defaults).  ``solute_style``: spheres |
        points | ribbon-points; ``density_opacity`` None keeps the value given to ``set_density``;
        ``reduce_motion`` jumps instead of animating and freezes the ring pulse."""
        style = str(solute_style).lower() if solute_style else "spheres"
        if style not in ("spheres", "points", "ribbon-points"):
            style = "spheres"
        old_eff = self._dn_opacity()
        style_changed = style != self._opts["solute_style"]
        self._opts.update(show_box=bool(show_box), show_axes=bool(show_axes), solute_style=style,
                          reduce_motion=bool(reduce_motion), show_scale_bar=bool(show_scale_bar),
                          density_opacity=None if density_opacity is None else float(min(max(density_opacity, 0.0), 1.0)))
        self._ovl_ver += 1
        if self._opts["reduce_motion"]:
            self._cam_anim.stop()
            self._fade_anim.stop()
            self._dn_fade = 1.0
            self._inertia_timer.stop()
        self._update_pulse_running()
        if style_changed:
            self._invalidate_data()
        new_eff = self._dn_opacity()
        if self._n_dn and abs(new_eff - old_eff) > 1e-4 and not self._opts["reduce_motion"]:
            # cross-fade the glow from its old to its new strength
            self._dn_fade = old_eff / new_eff if new_eff > 1e-6 else 0.0
            self._start_density_fade(from_zero=False)
        else:
            self.update()

    # ------------------------------------------------------------------------------------------------ camera
    def reset_view(self, animate: bool = True) -> None:
        """Frame solute + solvent + density in the default isometric orientation, together with the cell unless the
        cell is more than ~2.4x larger than the data (a huge periodic box would shrink the molecules to specks)."""
        q = _preset_quat("iso")
        tgt, dist, zoom = self._frame_params(q)
        self._framed_center, self._framed_radius = self._bounds_of(self._framing_layers())
        self._auto_frame = True            # keeps following the data until the user (or a camera call) moves the view
        self._cor = None
        self._goto(dict(q=q, zoom=zoom, target=tgt, dist=dist), animate)

    def set_view_preset(self, name: str, animate: bool = True) -> None:
        """Look down an axis: "x" | "y" | "z" (z = standard top view: x right, y up) | "iso" (also "-x", "-y", "-z")."""
        q = _preset_quat(name)
        if q is None:
            return
        self._auto_frame = False
        cur = self._cam_dict()
        cur["q"] = q
        self._goto(cur, animate)

    def focus_on(self, xyz, animate: bool = True) -> None:
        """Centre the view on the centroid of ``xyz``; the zoom is kept unless the points would leave the screen."""
        a = self._as_points(xyz)
        if a.shape[0]:
            a = a[np.isfinite(a).all(axis=1)]
        if a.shape[0] == 0:
            return
        c = a.astype(np.float64).mean(axis=0)
        spread = float(np.linalg.norm(a.astype(np.float64) - c, axis=1).max()) if a.shape[0] > 1 else 0.0
        cur = self._cam_dict()
        cur["target"] = c
        if spread > 1e-6:
            cur["zoom"] = min(cur["zoom"], 0.9 / spread)
        self._auto_frame = False
        self._goto(cur, animate)

    def set_center_of_rotation(self, xyz) -> None:
        """Rotate about the centroid of ``xyz`` (it stays fixed on screen); None = the screen centre."""
        if xyz is None:
            self._cor = None
            return
        a = self._as_points(xyz)
        if a.shape[0]:
            a = a[np.isfinite(a).all(axis=1)]
        self._cor = a.astype(np.float64).mean(axis=0) if a.shape[0] else None

    def camera_state(self) -> Dict[str, Any]:
        """JSON-able view: quat [w,x,y,z] (world->view rotation), zoom (1/A: a point 1/zoom A from the centre lands
        on the edge of the short viewport side, so it is resolution independent), center [x,y,z] (A), distance (A,
        perspective), cor (centre of rotation or None)."""
        return {"quat": [float(x) for x in self._q], "zoom": float(self._zoom),
                "center": [float(x) for x in self._target], "distance": float(self._dist),
                "cor": None if self._cor is None else [float(x) for x in self._cor]}

    def set_camera_state(self, state: Dict[str, Any], animate: bool = False) -> None:
        """Restore a state from ``camera_state()`` (missing / invalid fields keep their current value)."""
        if not isinstance(state, dict):
            return
        cur = self._cam_dict()
        try:
            if state.get("quat") is not None:
                q = np.asarray(state["quat"], np.float64).reshape(4)
                if np.isfinite(q).all() and np.linalg.norm(q) > 1e-9:
                    cur["q"] = _qnorm(q)
            z = state.get("zoom")
            if z is not None and np.isfinite(float(z)) and float(z) > 0:
                cur["zoom"] = float(z)
            c = state.get("center")
            if c is not None:
                c = np.asarray(c, np.float64).reshape(3)
                if np.isfinite(c).all():
                    cur["target"] = c
            d = state.get("distance")
            if d is not None and np.isfinite(float(d)) and float(d) > 0:
                cur["dist"] = float(d)
            if "cor" in state:
                cc = state.get("cor")
                if cc is None:
                    self._cor = None
                else:
                    cc = np.asarray(cc, np.float64).reshape(3)
                    if np.isfinite(cc).all():
                        self._cor = cc
        except Exception:
            return
        self._auto_frame = False
        self._goto(cur, animate)

    def is_animating(self) -> bool:
        return self._cam_anim.state() == QtCore.QAbstractAnimation.State.Running

    def project_points(self, xyz) -> Tuple[np.ndarray, np.ndarray]:
        """Screen position (logical px, widget coordinates) and camera depth (Angstrom) of world points."""
        a = np.asarray(xyz, np.float64).reshape(-1, 3)
        W, H = max(1, self.width()), max(1, self.height())
        scale = self._scale_log(W, H)
        v = (a - self._target) @ self._Rm.T
        d = self._dist - v[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ps = scale * self._dist / d
        sx = W * 0.5 + v[:, 0] * ps
        sy = H * 0.5 - v[:, 1] * ps
        bad = d <= 1e-9
        sx[bad] = np.nan
        sy[bad] = np.nan
        return np.stack([sx, sy], axis=1), d

    def pick(self, x: float, y: float) -> Dict[str, Any]:
        """Hover/click picking at widget position (x, y): payload dict or {}."""
        return self._public(self._pick(float(x), float(y)))

    @staticmethod
    def _public(hit: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in hit.items() if not k.startswith("_")}

    def set_pulse_phase(self, phase: float) -> None:
        try:
            self._pulse = float(phase) % 1.0
        except Exception:
            return
        self.update()

    def stats(self) -> Dict[str, Any]:
        """fps (EMA of recent paints), point counts (finite points only) and the last paint time in ms."""
        return {"fps": float(self._fps), "n_solute": int(self._n_sol), "n_solvent": int(self._n_sv),
                "n_density": int(self._n_dn), "last_paint_ms": float(self._last_paint_ms)}

    # ----------------------------------------------------------------------------- integration conveniences
    def set_scene(self, out: Dict[str, Any]) -> None:
        """Apply a scene dict as produced by the workbench preview service (all keys optional)."""
        try:
            sol = out.get("solute_xyz")
            if sol is not None:
                self.set_solute(sol, out.get("solute_radii"), out.get("solute_roles"), out.get("solute_labels"))
            if out.get("solvent_xyz") is not None:
                self.set_solvent(out["solvent_xyz"], out.get("solvent_h"), out.get("solvent_state"),
                                 out.get("solvent_labels"))
            box = out.get("box")
            if box is not None:
                origin = out.get("box_origin")
                if origin is None and out.get("center") is not None:
                    vec = _cell_vectors(np.asarray(box, np.float64))
                    origin = np.asarray(out["center"], np.float64) - 0.5 * vec.sum(axis=0)
                self.set_box(box, origin)
            else:
                self.set_box(None)
            if out.get("density_xyz") is not None and out.get("density_values") is not None:
                vals = np.asarray(out["density_values"], np.float64)
                if vals.size:
                    self.set_density(out["density_xyz"], vals, float(out.get("density_vmin", vals.min())),
                                     float(out.get("density_vmax", vals.max())), float(out.get("density_opacity", 0.85)))
        except Exception:
            self._last_error = traceback.format_exc()
            print(self._last_error, file=sys.stderr)

    def clear_scene(self) -> None:
        self.set_solute(None)
        self.set_solvent(None)
        self.clear_density()
        self.set_box(None)
        self._auto_frame = True
        self._framed_radius = 0.0

    # ------------------------------------------------------------------------------------------ export
    def render_to_image(self, width: int, height: int) -> QtGui.QImage:
        """Offscreen full-quality render (1.5x supersampled, without hover) of the current view at width x height."""
        w = max(1, int(width))
        h = max(1, int(height))
        ss = 1.5
        proj = self._compute_proj(w, h)
        sc = self._build_scene(w, h, ss, proj, interactive=False, aa=False)
        final = self._compose(sc, self._dn_opacity())
        p = QtGui.QPainter(final)
        p.scale(sc.Wr / float(w), sc.Hr / float(h))
        self._paint_static(p, w, h, proj)
        self._paint_dynamic(p, w, h, proj, export=True)
        p.end()
        out = final.scaled(w, h, _Qt.AspectRatioMode.IgnoreAspectRatio, _Qt.TransformationMode.SmoothTransformation)
        return out.convertToFormat(QtGui.QImage.Format.Format_RGB32)

    # =============================================================================================== helpers
    @staticmethod
    def _as_points(xyz) -> np.ndarray:
        if xyz is None:
            return np.zeros((0, 3), np.float32)
        try:
            a = np.asarray(xyz, np.float32)
            if a.size == 0:
                return np.zeros((0, 3), np.float32)
            return a.reshape(-1, 3)
        except Exception:
            return np.zeros((0, 3), np.float32)

    def _pick_mono_family(self) -> str:
        try:
            fams = set(QtGui.QFontDatabase.families())
        except Exception:
            fams = set()
        for name in ("Cascadia Mono", "Consolas", "JetBrains Mono", "SF Mono", "Menlo", "DejaVu Sans Mono",
                     "Liberation Mono", "Courier New"):
            if name in fams:
                return name
        return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont).family()

    def _dn_opacity(self) -> float:
        o = self._opts["density_opacity"]
        return float(self._dn_op_data if o is None else o)

    def _scale_log(self, W: int, H: int) -> float:
        return self._zoom * 0.5 * float(min(W, H))

    def _invalidate_data(self, bounds: bool = False) -> None:
        self._data_ver += 1
        self._proj = None
        if bounds:
            self._maybe_auto_frame()
        self._hover_dirty()
        self.update()

    def _hover_dirty(self) -> None:
        if self._mouse_pos is not None and self._drag_mode is None and not self._hover_timer.isActive():
            self._hover_timer.start(40)

    # ---- scene bounds / framing
    def _layers(self, with_box: bool = True) -> List[np.ndarray]:
        layers: List[np.ndarray] = []
        if self._n_sol:
            layers.append(self._sol_xyz)
        if self._n_sv:
            layers.append(self._sv_o)
        if self._n_dn:
            layers.append(self._dn_xyz)
        if with_box and self._box_corners is not None:
            layers.append(self._box_corners.astype(np.float32))
        return layers

    @staticmethod
    def _bounds_of(layers: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        if not layers:
            return np.zeros(3), 10.0
        lo = np.min(np.array([a.min(axis=0) for a in layers], np.float64), axis=0)
        hi = np.max(np.array([a.max(axis=0) for a in layers], np.float64), axis=0)
        c = (lo + hi) * 0.5
        r = 0.0
        for a in layers:
            r = max(r, float(np.sqrt(((a - c.astype(np.float32)) ** 2).sum(axis=1).max())))
        return c, max(r, 4.0)

    def _scene_bounds(self) -> Tuple[np.ndarray, float]:
        """Centre and radius of everything drawn (solute, solvent, density, box)."""
        return self._bounds_of(self._layers(True))

    def _framing_layers(self) -> List[np.ndarray]:
        """What a reset frames: all data, plus the cell unless it dwarfs the data (> 2.4x): a huge periodic box
        around a small region of interest would shrink the molecules to specks."""
        data = self._layers(False)
        box = self._box_corners
        if box is None:
            return data
        if not data:
            return [box.astype(np.float32)]
        d_ext = max(float(np.ptp(np.concatenate(data, axis=0), axis=0).max()), 1e-6)
        b_ext = float(np.ptp(box, axis=0).max())
        return data + [box.astype(np.float32)] if b_ext <= 2.4 * d_ext else data

    def _frame_params(self, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
        """Target, camera distance and zoom that frame the data in orientation ``q`` (default: the current one)."""
        layers = self._framing_layers()
        c, r = self._bounds_of(layers)
        dist = max(5.0 * r, 14.0)
        Rm = self._Rm if q is None else _qmat(_qnorm(q))
        ux = uy = 0.0
        for a in layers:
            if a.shape[0] > 40000:
                a = a[:: int(math.ceil(a.shape[0] / 40000.0))]
            v = (np.asarray(a, np.float64) - c) @ Rm.T
            u = dist / np.maximum(dist - v[:, 2], 1e-6)                      # perspective factor at the focal plane
            ux = max(ux, float(np.abs(v[:, 0] * u).max()))
            uy = max(uy, float(np.abs(v[:, 1] * u).max()))
        W, H = self.width(), self.height()
        if W < 80 or H < 80:
            W, H = 900, 600
        m = 0.88                                                                # share of the half viewport to fill
        if layers and (ux > 1e-6 or uy > 1e-6):
            scale = min(0.5 * W * m / max(ux, 1e-6), 0.5 * H * m / max(uy, 1e-6))
        else:
            scale = 0.5 * min(W, H) * m / max(r, 1e-6)
        zoom = 2.0 * scale / float(min(W, H))
        return c, dist, zoom

    def _maybe_auto_frame(self) -> None:
        if not self._auto_frame:
            return
        if (self._n_sol + self._n_sv + self._n_dn) == 0 and self._box_corners is None:
            return
        c, r = self._bounds_of(self._framing_layers())
        first = self._framed_radius <= 0.0
        moved = np.linalg.norm(c - self._framed_center) > 0.3 * max(r, self._framed_radius)
        grew = abs(r - self._framed_radius) > 0.3 * max(r, self._framed_radius, 1e-6)
        if first or moved or grew:
            tgt, dist, zoom = self._frame_params()
            self._set_cam(self._q, zoom, tgt, dist)
            self._framed_center, self._framed_radius = c, r

    # ---- camera plumbing
    def _cam_dict(self) -> Dict[str, Any]:
        return dict(q=self._q.copy(), zoom=self._zoom, target=self._target.copy(), dist=self._dist)

    def _set_cam(self, q, zoom, target, dist) -> None:
        self._q = _qnorm(q)
        self._Rm = _qmat(self._q)
        self._zoom = float(min(max(zoom, 1e-5), 50.0))
        self._target = np.asarray(target, np.float64).copy()
        self._dist = float(max(dist, 1.0))
        self._camera_touched()

    def _camera_touched(self) -> None:
        self._cam_ver += 1
        self._proj = None
        self._hover_dirty()
        self.update()
        self._emit_camera_throttled()

    def _goto(self, cam: Dict[str, Any], animate: bool) -> None:
        self._cam_anim.stop()
        if animate and not self._opts["reduce_motion"]:
            self._cam_from = self._cam_dict()
            self._cam_to = dict(q=_qnorm(cam["q"]), zoom=float(cam["zoom"]), target=np.asarray(cam["target"], np.float64),
                                dist=float(cam["dist"]))
            self._begin_interaction(hold=False)
            self._cam_anim.start()
        else:
            self._set_cam(cam["q"], cam["zoom"], cam["target"], cam["dist"])

    @_guarded
    def _on_cam_anim(self, v) -> None:
        a, b = self._cam_from, self._cam_to
        if a is None or b is None:
            return
        t = float(v)
        q = _qslerp(a["q"], b["q"], t)
        zoom = math.exp(math.log(max(a["zoom"], 1e-9)) * (1 - t) + math.log(max(b["zoom"], 1e-9)) * t)
        tgt = a["target"] * (1 - t) + b["target"] * t
        dist = a["dist"] * (1 - t) + b["dist"] * t
        self._set_cam(q, zoom, tgt, dist)
        self._settle_soon()

    @_guarded
    def _on_cam_anim_done(self) -> None:
        b = self._cam_to
        if b is not None:
            self._set_cam(b["q"], b["zoom"], b["target"], b["dist"])
        self._settle_soon()

    @_guarded
    def _emit_camera(self) -> None:
        self._last_cam_emit = time.perf_counter()
        self.cameraChanged.emit()

    def _emit_camera_throttled(self) -> None:
        now = time.perf_counter()
        if now - self._last_cam_emit >= 1.0 / 60.0:
            self._emit_camera()
        elif not self._cam_emit_timer.isActive():
            self._cam_emit_timer.start(max(1, int((1.0 / 60.0 - (now - self._last_cam_emit)) * 1000) + 1))

    # ---- interaction state
    def _begin_interaction(self, hold: bool = True) -> None:
        self._settle_timer.stop()
        if hold:
            self._cam_anim.stop()

    def _settle_soon(self) -> None:
        self._interacting = True
        self._settle_timer.start(_SETTLE_MS)

    @_guarded
    def _settle(self) -> None:
        if self.is_animating() or self._inertia_timer.isActive() or self._drag_mode is not None:
            self._settle_timer.start(_SETTLE_MS)
            return
        self._interacting = False
        self._rs = min(1.0, self._rs)
        self.update()

    # ---- camera operations (all in logical pixels)
    def _arcball(self, x: float, y: float) -> np.ndarray:
        W, H = max(1, self.width()), max(1, self.height())
        r = 0.5 * min(W, H) * 1.05
        px = (x - W * 0.5) / r
        py = -(y - H * 0.5) / r
        d2 = px * px + py * py
        pz = math.sqrt(1.0 - d2) if d2 <= 0.5 else 0.5 / math.sqrt(d2)
        v = np.array([px, py, pz])
        return v / np.linalg.norm(v)

    def _rotate_view(self, qd: np.ndarray) -> None:
        """Apply a rotation expressed in view space, keeping the centre of rotation fixed on screen."""
        c = self._cor if self._cor is not None else self._target
        vc = self._Rm @ (c - self._target)
        q = _qnorm(_qmul(qd, self._q))
        Rn = _qmat(q)
        tgt = c - Rn.T @ vc
        self._set_cam(q, self._zoom, tgt, self._dist)

    def _pan_px(self, dx: float, dy: float) -> None:
        W, H = max(1, self.width()), max(1, self.height())
        s = self._scale_log(W, H)
        if s <= 0:
            return
        delta = self._Rm.T @ np.array([-dx / s, dy / s, 0.0])
        self._set_cam(self._q, self._zoom, self._target + delta, self._dist)

    def _zoom_at(self, factor: float, x: float, y: float) -> None:
        W, H = max(1, self.width()), max(1, self.height())
        s0 = self._scale_log(W, H)
        z1 = min(max(self._zoom * factor, 1e-5), 50.0)
        s1 = z1 * 0.5 * min(W, H)
        ox, oy = x - W * 0.5, y - H * 0.5
        k = 1.0 / s0 - 1.0 / s1
        delta = self._Rm.T @ np.array([ox * k, -oy * k, 0.0])
        self._set_cam(self._q, z1, self._target + delta, self._dist)

    # =============================================================================================== events
    @_guarded
    def resizeEvent(self, ev) -> None:  # noqa: N802
        self._proj = None
        self._scene = None
        super().resizeEvent(ev)

    @_guarded
    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        self._update_pulse_running()

    @_guarded
    def hideEvent(self, ev) -> None:  # noqa: N802
        super().hideEvent(ev)
        if self._pulse_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self._pulse_anim.pause()

    @_guarded
    def leaveEvent(self, ev) -> None:  # noqa: N802
        self._mouse_pos = None
        if self._hover is not None or self._hover_key is not None:
            self._hover = None
            self._hover_key = None
            self.hovered.emit({})
            self.update()
        super().leaveEvent(ev)

    @_guarded
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        self.setFocus(_Qt.FocusReason.MouseFocusReason)
        pos = ev.position()
        self._mouse_pos = pos
        btn = ev.button()
        if btn == _Qt.MouseButton.LeftButton or btn == _Qt.MouseButton.MiddleButton:
            shift = bool(ev.modifiers() & _Qt.KeyboardModifier.ShiftModifier)
            self._drag_mode = "pan" if (btn == _Qt.MouseButton.MiddleButton or shift) else "rot"
            self._drag_last = pos
            self._drag_moved = 0.0
            self._press_t = time.perf_counter()
            self._ball_prev = self._arcball(pos.x(), pos.y())
            self._vel = 0.0
            self._inertia_timer.stop()
            self._begin_interaction()
            self.setCursor(_Qt.CursorShape.ClosedHandCursor if self._drag_mode == "rot" else _Qt.CursorShape.SizeAllCursor)
            self._auto_frame = False
            ev.accept()
        else:
            super().mousePressEvent(ev)

    @_guarded
    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        pos = ev.position()
        self._mouse_pos = pos
        if self._drag_mode is None:
            self._request_pick()
            return
        dx = pos.x() - self._drag_last.x()
        dy = pos.y() - self._drag_last.y()
        self._drag_moved += abs(dx) + abs(dy)
        now = time.perf_counter()
        if self._drag_mode == "pan":
            self._pan_px(dx, dy)
        else:
            v1 = self._arcball(pos.x(), pos.y())
            v0 = self._ball_prev
            axis = np.cross(v0, v1)
            sn = float(np.linalg.norm(axis))
            cs = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
            ang = math.atan2(sn, cs)
            if sn > 1e-9 and ang > 1e-6:
                axis = axis / sn
                self._rotate_view(_qaxis(axis, ang))
                dt = max(now - self._vel_t, 1e-3) if self._vel_t else 0.016
                inst = ang / min(max(dt, 0.004), 0.1)
                self._vel = 0.6 * self._vel + 0.4 * inst if self._vel else inst
                self._vel_axis = axis
            self._vel_t = now
            self._ball_prev = v1
        self._drag_last = pos
        self._settle_soon()
        ev.accept()

    @_guarded
    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        mode = self._drag_mode
        if mode is None:
            super().mouseReleaseEvent(ev)
            return
        self._drag_mode = None
        self.setCursor(_Qt.CursorShape.OpenHandCursor)
        pos = ev.position()
        click = (ev.button() == _Qt.MouseButton.LeftButton and self._drag_moved < 4.0
                 and (time.perf_counter() - self._press_t) < 0.6)
        if click:
            self.clicked.emit(self._public(self._pick(pos.x(), pos.y())))
        if (mode == "rot" and not click and not self._opts["reduce_motion"]
                and (time.perf_counter() - self._vel_t) < 0.06 and self._vel > 0.8):
            self._vel = min(self._vel, 6.0)
            self._inertia_last = time.perf_counter()
            self._inertia_timer.start()
        self._vel_t = 0.0
        if self._interacting or self._drag_moved > 0.0 or self._inertia_timer.isActive():
            self._settle_soon()                                     # a plain click leaves the picture untouched
        self._request_pick()
        ev.accept()

    @_guarded
    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._inertia_timer.stop()
        self.reset_view(animate=True)
        ev.accept()

    @_guarded
    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:  # noqa: N802
        dy = ev.angleDelta().y()
        if dy == 0:
            ev.ignore()
            return
        factor = 1.0016 ** float(dy)
        pos = ev.position()
        self._begin_interaction()
        self._auto_frame = False
        self._zoom_at(factor, pos.x(), pos.y())
        self._settle_soon()
        ev.accept()

    @_guarded
    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:  # noqa: N802
        k = ev.key()
        mods = ev.modifiers()
        step = math.radians(15.0 if mods & _Qt.KeyboardModifier.ControlModifier else
                            1.5 if mods & _Qt.KeyboardModifier.ShiftModifier else 5.0)
        K = _Qt.Key
        if k in (K.Key_Left, K.Key_Right, K.Key_Up, K.Key_Down):
            self._begin_interaction()
            self._auto_frame = False
            if k == K.Key_Left:
                qd = _qaxis((0, 1, 0), -step)
            elif k == K.Key_Right:
                qd = _qaxis((0, 1, 0), step)
            elif k == K.Key_Up:
                qd = _qaxis((1, 0, 0), -step)
            else:
                qd = _qaxis((1, 0, 0), step)
            self._rotate_view(qd)
            self._settle_soon()
        elif k == K.Key_R:
            self.reset_view(animate=True)
        elif k == K.Key_1:
            self.set_view_preset("x")
        elif k == K.Key_2:
            self.set_view_preset("y")
        elif k == K.Key_3:
            self.set_view_preset("z")
        elif k == K.Key_0:
            self.set_view_preset("iso")
        elif k in (K.Key_Plus, K.Key_Equal, K.Key_Minus, K.Key_Underscore):
            f = 1.15 if k in (K.Key_Plus, K.Key_Equal) else 1.0 / 1.15
            self._begin_interaction()
            self._auto_frame = False
            self._zoom_at(f, self.width() * 0.5, self.height() * 0.5)
            self._settle_soon()
        else:
            super().keyPressEvent(ev)
            return
        ev.accept()

    @_guarded
    def _inertia_tick(self) -> None:
        now = time.perf_counter()
        dt = min(now - (self._inertia_last or now), 0.05)
        self._inertia_last = now
        self._vel *= math.exp(-dt / 0.22)
        if self._vel < 0.12:
            self._inertia_timer.stop()
            self._vel = 0.0
            self._settle_soon()
            return
        self._rotate_view(_qaxis(self._vel_axis, self._vel * dt))
        self._settle_soon()

    # ---- animation callbacks
    def _start_density_fade(self, from_zero: bool) -> None:
        if self._opts["reduce_motion"] or self._n_dn == 0:
            self._dn_fade = 1.0
            self.update()
            return
        start = 0.0 if from_zero else float(self._dn_fade)
        self._fade_anim.stop()
        self._fade_anim.setStartValue(start)
        self._fade_anim.setEndValue(1.0)
        self._dn_fade = start
        self._fade_anim.start()

    @_guarded
    def _on_fade_anim(self, v) -> None:
        self._dn_fade = float(v)
        self.update()

    def _update_pulse_running(self) -> None:
        """Run the ring pulse only while molecules in state 2/3 exist and motion is allowed."""
        need = False
        if self._n_sv and not self._opts["reduce_motion"]:
            need = bool(((self._sv_state == 2) | (self._sv_state == 3)).any())
        State = QtCore.QAbstractAnimation.State
        st = self._pulse_anim.state()
        if need:
            if st == State.Stopped:
                self._pulse_anim.start()
            elif st == State.Paused and self.isVisible():
                self._pulse_anim.resume()
        elif st != State.Stopped:
            self._pulse_anim.stop()
        if self._opts["reduce_motion"]:
            self._pulse = 0.35

    def pulse_running(self) -> bool:
        return self._pulse_anim.state() == QtCore.QAbstractAnimation.State.Running

    @_guarded
    def _on_pulse_anim(self, v) -> None:
        now = time.perf_counter()
        if now - self._pulse_last_update < 0.030:        # ~30 Hz is plenty for a ring
            return
        self._pulse_last_update = now
        self._pulse = float(v)
        self.update()

    # =============================================================================================== projection
    def _compute_proj(self, W: int, H: int) -> _Proj:
        P = _Proj()
        P.W, P.H = W, H
        P.scale = self._scale_log(W, H)
        P.cx, P.cy = W * 0.5, H * 0.5
        P.R = self._Rm
        P.dist = self._dist
        P.near = _NEAR_FRAC * self._dist
        Rm = self._Rm.astype(np.float32)
        tv = (Rm @ self._target.astype(np.float32)).astype(np.float32)
        f = np.float32(P.scale * self._dist)
        D = np.float32(self._dist)
        cx, cy = np.float32(P.cx), np.float32(P.cy)

        def proj(xyz: np.ndarray):
            v = xyz @ Rm.T
            v -= tv
            d = D - v[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                ps = f / d
            sx = cx + v[:, 0] * ps
            sy = cy - v[:, 1] * ps
            return sx, sy, d, ps

        if self._n_sol:
            P.sol = proj(self._sol_xyz)
        if self._n_sv:
            P.svo = proj(self._sv_o)
            P.svh = proj(self._sv_h.reshape(-1, 3))
        if self._n_dn:
            P.dn = proj(self._dn_xyz)
        c, r = self._scene_bounds_cached()
        zc = float((self._Rm @ (c - self._target))[2])
        dc = self._dist - zc
        P.dwin = (dc - 1.06 * r - 1.0, dc + 1.06 * r + 1.0)
        return P

    def _scene_bounds_cached(self) -> Tuple[np.ndarray, float]:
        key = self._data_ver
        cached = self._bounds_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        c, r = self._scene_bounds()
        self._bounds_cache = (key, c, r)
        return c, r

    def _get_proj(self) -> _Proj:
        W, H = max(1, self.width()), max(1, self.height())
        P = self._proj
        if P is not None and P.W == W and P.H == H:
            return P
        P = self._compute_proj(W, H)
        self._proj = P
        return P

    # =============================================================================================== rendering
    def _bg_image(self, Wr: int, Hr: int) -> QtGui.QImage:
        key = (Wr, Hr, self._theme_ver)
        img = self._bg_cache.get(key)
        if img is not None:
            return img
        th = self._theme
        y = np.linspace(0.0, 1.0, Hr, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, Wr, dtype=np.float32)[None, :]
        top = th.c["bg1"].astype(np.float32)
        bot = th.c["bg0"].astype(np.float32)
        grad = top[None, None, :] * (1.0 - y)[..., None] + bot[None, None, :] * y[..., None]      # (Hr,1,3)
        asp = float(Wr) / float(max(Hr, 1))
        r2 = (x * 0.62 * min(asp, 1.8)) ** 2 + ((y - 0.46) * 1.55) ** 2
        v = np.clip((r2 - 0.25) / 1.1, 0.0, 1.0)
        v = v * v * (3.0 - 2.0 * v)
        if th.dark:
            fac = (1.0 - 0.30 * v)[..., None]
            rgb = grad * fac
        else:
            edge = th.c["line"].astype(np.float32)
            rgb = grad * (1.0 - 0.10 * v)[..., None] + edge[None, None, :] * (0.10 * v)[..., None] * 0.35
        rgb = rgb + (_dither_tile(Hr, Wr) - 0.5) / 255.0
        arr = _pack(rgb)
        img = QtGui.QImage(arr.data, Wr, Hr, Wr * 4, QtGui.QImage.Format.Format_RGB32).copy()
        if len(self._bg_cache) > 6:
            self._bg_cache.clear()
        self._bg_cache[key] = img
        return img

    # ---- sprite collection ---------------------------------------------------------------------------
    def _collect_sprites(self, proj: _Proj, k: float, Wr: int, Hr: int, interactive: bool):
        th = self._theme
        dlo, dhi = proj.dwin
        inv = 1.0 / max(dhi - dlo, 1e-6)
        out_cx, out_cy, out_R, out_w = [], [], [], []
        style = self._opts["solute_style"]

        def finish(sx, sy, d, rpx, cls, fogscale, rmin, rmax, extra_valid=None, fog_gamma=1.0):
            cxf = sx * k
            cyf = sy * k
            valid = (d > proj.near) & np.isfinite(cxf) & np.isfinite(cyf)
            if extra_valid is not None:
                valid &= extra_valid
            R = np.clip(np.rint(rpx), rmin, rmax)
            valid &= (cxf + R >= 0) & (cxf - R < Wr) & (cyf + R >= 0) & (cyf - R < Hr)
            if not valid.all():
                cxf, cyf, d, R, cls, fogscale = cxf[valid], cyf[valid], d[valid], R[valid], cls[valid], \
                    (fogscale[valid] if isinstance(fogscale, np.ndarray) else fogscale)
            if cxf.size == 0:
                return
            ft = np.clip((d - dlo) * inv, 0.0, 1.0)
            lvl = ((ft if fog_gamma == 1.0 else ft ** fog_gamma) * fogscale * (_FOG - 1) + 0.5).astype(np.int32)
            row = cls.astype(np.int32) * _FOG + lvl
            dq = (ft * _DQ_SPAN + 100.0).astype(np.uint32)
            out_cx.append(np.rint(cxf).astype(np.int64))
            out_cy.append(np.rint(cyf).astype(np.int64))
            out_R.append(R.astype(np.int64))
            out_w.append((dq << np.uint32(16)) + (row.astype(np.uint32) << np.uint32(8)))

        # ---- solute
        if self._n_sol and proj.sol is not None:
            sx, sy, d, ps = proj.sol
            rad = self._sol_rad_eff
            cls = self._sol_cls
            role = self._sol_role
            fogs = np.where(role == 1, 0.45, 1.0).astype(np.float32)
            comp = 1.0
            if interactive:
                sub, comp = self._solute_subset(self._lod)
                if sub is not None:
                    sx, sy, d, ps, rad, cls, fogs, role = (a[sub] for a in (sx, sy, d, ps, rad, cls, fogs, role))
            ft = np.clip((d - dlo) * inv, 0.0, 1.0)
            shrink = 1.0 - 0.07 * ft
            rpx = rad * ps * (k * comp) * shrink
            if style == "spheres":
                finish(sx, sy, d, rpx, cls, fogs, 1, _MAXR)
            else:                                                  # points: small legible dots, anchors a bit bigger
                rp = rpx * 0.20 + np.where(role == 1, 0.9 * k, 0.0)
                finish(sx, sy, d, rp, cls, fogs * 0.75, 1, 4, fog_gamma=0.55 if style == "ribbon-points" else 1.0)

        # ---- solvent
        if self._n_sv and proj.svo is not None:
            st = self._sv_state
            sx, sy, d, ps = proj.svo
            rpx = _RO[st] * ps * k
            drawn = st != 3
            fs = np.where(st == 0, 1.0, 0.5).astype(np.float32)
            rmin = np.where(st == 0, 0, 1)
            finish(sx, sy, d, rpx, _CLS_O[st], fs, rmin, _MAXR, extra_valid=drawn)
            if self._sv_hvalid.any() and proj.svh is not None:
                st2 = np.repeat(st, 2)
                hv = np.repeat(self._sv_hvalid, 2) & (st2 != 3)
                sxh, syh, dh, psh = proj.svh
                rph = _RH[st2] * psh * k
                # bulk hydrogens are skipped when they would be sub-pixel specks
                hv &= ~((st2 == 0) & (rph < 2.6))
                if hv.any():
                    fs2 = np.where(st2 == 0, 1.0, 0.5).astype(np.float32)
                    finish(sxh, syh, dh, rph, _CLS_H[st2], fs2, 0, _MAXR, extra_valid=hv)

        if not out_cx:
            e = np.zeros(0, np.int64)
            return e, e, e, np.zeros(0, np.uint32)
        if len(out_cx) == 1:
            return out_cx[0], out_cy[0], out_R[0], out_w[0]
        return (np.concatenate(out_cx), np.concatenate(out_cy), np.concatenate(out_R), np.concatenate(out_w))

    # ---- the z-buffer splat ----------------------------------------------------------------------------
    def _splat(self, cx, cy, R, w0, Wr: int, Hr: int) -> np.ndarray:
        """Packed z/colour buffer with a 1 px guard border: shape (Hr+2)*(Wr+2), stride Wr+2."""
        Wp = Wr + 2
        buf = np.full((Hr + 2) * Wp, _EMPTY, np.uint32)
        n = cx.size
        if n == 0:
            return buf
        order = np.argsort((w0 >> np.uint32(16)).astype(np.uint16), kind="stable")      # near -> far (radix sort)
        cx, cy, R, w0 = cx[order], cy[order], R[order], w0[order]
        bank = self._bank
        pos = 0
        # first stage: roughly enough of the nearest sprites to cover the image once (big sprites -> few)
        k0 = float(np.mean((R[:256].astype(np.float32) + 0.5) ** 2)) * math.pi
        stage = int(min(max(Wr * Hr / max(k0, 1.0), 24), _STAGE0))
        while pos < n:
            end = min(n, pos + stage)
            scx, scy, sR, sw = cx[pos:end], cy[pos:end], R[pos:end], w0[pos:end]
            if pos > 0:
                scx, scy, sR, sw = self._cull(buf, scx, scy, sR, sw, Wr, Hr)
            if scx.size:
                R8 = sR.astype(np.uint8)
                o2 = np.argsort(R8, kind="stable")
                scx, scy, sw, R8 = scx[o2], scy[o2], sw[o2], R8[o2]
                bc = np.bincount(R8)
                starts = np.cumsum(bc) - bc
                for rv in np.flatnonzero(bc):
                    a, b = int(starts[rv]), int(starts[rv] + bc[rv])
                    self._draw_group(buf, int(rv), scx[a:b][::-1], scy[a:b][::-1], sw[a:b][::-1], Wr, Hr, bank)
            pos = end
            stage *= 2
        return buf

    @staticmethod
    def _cull(buf, scx, scy, sR, sw, Wr: int, Hr: int):
        """Drop sprites that are hidden at all of ``_CULL_PATTERN`` (centre + 16 points inside the disc).

        A sprite is only removed when every sample point already shows a nearer pixel, so what is lost can only be
        slivers seen through gaps (measured: ~0.1-0.4 % of the covered pixels in dense scenes)."""
        big = sR >= 3
        if not big.any():
            return scx, scy, sR, sw
        Wp = Wr + 2
        hm = (sR * _CULL_REACH).astype(np.int64)
        ok = big & (scx >= hm) & (scx < Wr - hm) & (scy >= hm) & (scy < Hr - hm)
        if not ok.any():
            return scx, scy, sR, sw
        thr = (sw >> np.uint32(16)) << np.uint32(16)
        npx = buf.size
        lin = np.clip((scy + 1) * Wp + scx + 1, 0, npx - 1)
        hid = buf[lin] < thr
        rf = sR.astype(np.float32)
        for fx, fy in _CULL_PATTERN:
            off = (rf * fy).astype(np.int64) * Wp + (rf * fx).astype(np.int64)
            hid &= buf[np.clip(lin + off, 0, npx - 1)] < thr
        hid &= ok
        if not hid.any():
            return scx, scy, sR, sw
        keep = ~hid
        return scx[keep], scy[keep], sR[keep], sw[keep]

    @staticmethod
    def _draw_group(buf, Rv: int, gcx, gcy, gw, Wr: int, Hr: int, bank: _SpriteBank) -> None:
        """Draw same-radius sprites (given far -> near) with an exact depth test."""
        Wp = Wr + 2
        dy, dx, tidx = bank.sprite(Rv)
        K = dy.size
        ch = max(1, _CHUNK // K)
        inside = (gcx >= Rv) & (gcx < Wr - Rv) & (gcy >= Rv) & (gcy < Hr - Rv)
        if inside.all():
            ins_cx, ins_cy, ins_w = gcx, gcy, gw
            brd = None
        else:
            ins_cx, ins_cy, ins_w = gcx[inside], gcy[inside], gw[inside]
            nb = ~inside
            brd = (gcx[nb], gcy[nb], gw[nb])
        if ins_cx.size:
            off = bank.offsets(Rv, Wp)
            base = (ins_cy + 1) * Wp + ins_cx + 1
            for c0 in range(0, base.size, ch):
                idx = base[c0:c0 + ch, None] + off
                pk = ins_w[c0:c0 + ch, None] + tidx
                msk = pk < buf[idx]
                cnt = int(np.count_nonzero(msk))
                if cnt == 0:
                    continue
                if cnt == msk.size:
                    _scatter(buf, idx.ravel(), pk.ravel())
                else:
                    _scatter(buf, idx[msk], pk[msk])
        if brd is not None and brd[0].size:
            bcx, bcy, bw = brd
            for c0 in range(0, bcx.size, ch):
                xs = bcx[c0:c0 + ch, None] + dx
                ys = bcy[c0:c0 + ch, None] + dy
                ok = (xs >= 0) & (xs < Wr) & (ys >= 0) & (ys < Hr)
                if not ok.any():
                    continue
                idx = ((ys + 1) * Wp + xs + 1)[ok]
                pk = (bw[c0:c0 + ch, None] + tidx)[ok]
                msk = pk < buf[idx]
                if msk.any():
                    _scatter(buf, idx[msk], pk[msk])

    # ---- resolve + anti-alias ----------------------------------------------------------------------------
    def _resolve(self, buf: np.ndarray, Wr: int, Hr: int, aa: bool):
        """Premultiplied ARGB for the covered pixels (padded layout, stride Wr+2); empty pixels stay 0."""
        lut = self._theme.lut
        rgb = np.zeros(buf.size, np.uint32)
        ci = np.flatnonzero(buf != _EMPTY)
        if ci.size:
            rgb[ci] = lut.take(buf[ci] & np.uint32(0xFFFF))
            if aa:
                _aa_sprite_edges(buf, rgb, Wr)
                _aa_edges(buf, rgb, Wr, Hr)
        return rgb, ci

    @staticmethod
    def _focus_pixels(buf: np.ndarray, ci: np.ndarray) -> Optional[np.ndarray]:
        """Flat (padded) indices of pixels whose top sprite is a 'focus' object: anchors, zone / entered waters."""
        if ci.size == 0:
            return None
        rows = np.zeros(256, bool)
        for c in (C_ANCHOR, C_ZONE_O, C_ZONE_H, C_ENTER_O, C_ENTER_H):
            rows[c * _FOG:(c + 1) * _FOG] = True
        fm = rows[(buf[ci] >> np.uint32(8)) & np.uint32(0xFF)]
        return ci[fm] if fm.any() else None

    def _focus_mask(self, buf: np.ndarray, fi: Optional[np.ndarray], Wr: int, Hr: int) -> Optional[QtGui.QImage]:
        """Opaque-alpha image over the focus pixels (used to cut them out of the glow in front of the spheres)."""
        if fi is None:
            return None
        m = np.zeros(buf.size, np.uint32)
        m[fi] = np.uint32(0xFF000000)
        self._mask_keep = m                                       # keep the array alive for the QImage
        Wp = Wr + 2
        return QtGui.QImage(m[Wp + 1:].data, Wr, Hr, Wp * 4, QtGui.QImage.Format.Format_ARGB32_Premultiplied)

    def _combine_glow(self, main, bloom, f: int, Wr: int, Hr: int, mask) -> QtGui.QImage:
        """Scale the grid image to full resolution, add the bloom, cut out the focus sprites."""
        add = self._theme.glow_add
        img = QtGui.QImage(Wr, Hr, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QtGui.QPainter(img)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = QtCore.QRectF(0.0, 0.0, main.width() * f, main.height() * f)
        p.drawImage(rect, main)
        if bloom is not None:
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus if add
                                 else QtGui.QPainter.CompositionMode.CompositionMode_Multiply)
            p.setOpacity(_BLOOM_ADD if add else _BLOOM_MUL)
            p.drawImage(rect, bloom)
        if mask is not None:
            p.setOpacity(1.0)
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_DestinationOut)
            p.drawImage(0, 0, mask)
        p.end()
        return img

    # ---- the density glow --------------------------------------------------------------------------------
    def _build_glow(self, proj: _Proj, k: float, Wr: int, Hr: int, buf: np.ndarray, interactive: bool,
                    focus_idx: Optional[np.ndarray] = None):
        """Density cloud + zone/anchor halos -> soft light layers (in front of / behind the opaque spheres).

        Every source is a Gaussian blob of world size ``sigma``; blobs are accumulated at a coarse grid (bilinear
        splat), blurred, tone mapped and drawn by Qt with additive (dark) or multiply (light) blending."""
        th = self._theme
        dlo, dhi = proj.dwin
        span = max(dhi - dlo, 1e-6)
        inv = 1.0 / span
        ps_t = proj.scale * k                                    # raster px per Angstrom at the target plane
        sig_d = _GLOW_SIGMA_REL * self._dn_h                      # world sigma of a density blob
        f_min = (4.0 if interactive else 3.0) * k
        f = int(min(max(round(sig_d * ps_t * 0.85), round(f_min), 2), 12))
        cell = float(f * f)
        xs, ys, ws, ts, dqs = [], [], [], [], []

        def add(sx, sy, d, amp, tval, sigma_world, kappa=0.0, d0=0.0, dspan=1.0, boost=1.0):
            valid = (d > proj.near) & np.isfinite(sx) & np.isfinite(sy)
            fcl = np.clip((d - d0) / dspan, 0.0, 1.0)
            sp = max(sigma_world * ps_t, 0.7)
            wt = amp * np.exp(-kappa * fcl) * np.float32(boost * 2.0 * math.pi * sp * sp / cell)
            if np.ndim(tval) == 0:
                tv = np.full(sx.shape, tval, np.float32)
            else:
                tv = tval
            if not valid.all():
                sx, sy, wt, tv, d = sx[valid], sy[valid], wt[valid], tv[valid], d[valid]
            if sx.size == 0:
                return
            xs.append(sx * k)
            ys.append(sy * k)
            ws.append(wt.astype(np.float32))
            ts.append(tv.astype(np.float32))
            dqs.append(np.clip((d - dlo) * inv, 0.0, 1.0) * _DQ_SPAN + 100.0)

        if self._n_dn and proj.dn is not None:
            sx, sy, d, ps = proj.dn
            emit, tv = self._dn_emit, self._dn_t
            boost = _GLOW_GAIN_DENSITY
            sub = self._density_subset(self._lod) if interactive else None
            if sub is not None:
                sx, sy, d, emit, tv = sx[sub], sy[sub], d[sub], emit[sub], tv[sub]
                boost *= self._n_dn / float(max(sub.size, 1))
            dd = d[d > proj.near]
            if dd.size:
                d0 = float(dd.min())
                dsp = max(float(dd.max()) - d0, 1e-3)
                add(sx, sy, d, emit, tv, sig_d, _GLOW_DEPTH_K, d0, dsp, boost)
        if self._n_sv and proj.svo is not None:
            st = self._sv_state
            hm = (st == 1) | (st == 2)
            if hm.any():
                sx, sy, d, ps = (a[hm] for a in proj.svo)
                amp = np.where(st[hm] == 2, 1.0, 0.8).astype(np.float32)
                tvv = np.where(st[hm] == 2, 0.60, 0.44).astype(np.float32)
                add(sx, sy, d, amp, tvv, _GLOW_HALO_SIGMA, 0.5, dlo, span, _GLOW_GAIN_HALO)
        if self._n_sol and proj.sol is not None:
            am = np.flatnonzero(self._sol_role == 1)
            if am.size:
                if am.size > 700:
                    am = am[:: int(math.ceil(am.size / 700.0))]
                sx, sy, d, ps = (a[am] for a in proj.sol)
                add(sx, sy, d, np.float32(0.7), np.float32(_ANCHOR_T), _GLOW_ANCHOR_SIGMA, 0.35, dlo, span, _GLOW_GAIN_ANCHOR)
        if not xs:
            return None
        X = np.concatenate(xs) if len(xs) > 1 else xs[0]
        Y = np.concatenate(ys) if len(ys) > 1 else ys[0]
        Wt = np.concatenate(ws) if len(ws) > 1 else ws[0]
        T = np.concatenate(ts) if len(ts) > 1 else ts[0]
        DQ = np.concatenate(dqs) if len(dqs) > 1 else dqs[0]

        Wg = (Wr + f - 1) // f + 2          # padded grid: one guard cell on each side
        Hg = (Hr + f - 1) // f + 2
        npx = Hg * Wg
        gx = X / f - 0.5 + 1.0
        gy = Y / f - 0.5 + 1.0
        # depth of the nearest opaque sprite at the source's own pixel decides: in front of it or behind it
        Wp = Wr + 2
        ix = np.clip((X + 0.5).astype(np.int32), 0, Wr - 1)
        iy = np.clip((Y + 0.5).astype(np.int32), 0, Hr - 1)
        zpix = buf.view(np.uint16)[1::2][(iy + 1) * Wp + ix + 1]
        margin = _GLOW_FRONT_MARGIN / span * _DQ_SPAN
        behind = DQ > (zpix.astype(np.float32) + np.float32(margin))
        inb = (gx >= 0) & (gx < Wg - 1) & (gy >= 0) & (gy < Hg - 1)
        if not inb.all():
            gx, gy, Wt, T, behind = gx[inb], gy[inb], Wt[inb], T[inb], behind[inb]
        if gx.size == 0:
            return None
        any_behind = bool(behind.any())
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        fx = (gx - x0).astype(np.float32)
        fy = (gy - y0).astype(np.float32)
        base = y0 * Wg + x0 + np.where(behind, npx, 0)
        if interactive:
            idx = base + np.where(fx > 0.5, 1, 0) + np.where(fy > 0.5, Wg, 0)
            A = np.bincount(idx, weights=Wt, minlength=2 * npx)
            B = np.bincount(idx, weights=Wt * T, minlength=2 * npx)
        else:
            idx = np.concatenate((base, base + 1, base + Wg, base + Wg + 1))
            w4 = np.concatenate((Wt * (1 - fx) * (1 - fy), Wt * fx * (1 - fy), Wt * (1 - fx) * fy, Wt * fx * fy))
            A = np.bincount(idx, weights=w4, minlength=2 * npx)
            B = np.bincount(idx, weights=w4 * np.tile(T, 4), minlength=2 * npx)
        nsets = 2 if any_behind else 1
        acc = np.empty((2 * nsets, Hg, Wg), np.float32)
        for si in range(nsets):
            acc[2 * si] = A[si * npx:(si + 1) * npx].reshape(Hg, Wg)
            acc[2 * si + 1] = B[si * npx:(si + 1) * npx].reshape(Hg, Wg)
        acc = _blur121(acc)
        acc = _blur121(acc)
        acc *= np.float32(1.0 / 256.0)
        out = []
        cov = None
        if focus_idx is not None and focus_idx.size:               # interactive: cut focus sprites out at grid level
            yy = focus_idx // Wp - 1
            xx = focus_idx - (yy + 1) * Wp - 1
            cidx = np.clip(yy // f, 0, Hg - 3) * (Wg - 2) + np.clip(xx // f, 0, Wg - 3)
            cnt = np.bincount(cidx, minlength=(Hg - 2) * (Wg - 2)).reshape(Hg - 2, Wg - 2)
            cov = np.clip(cnt * (1.6 / (f * f)), 0.0, 1.0).astype(np.float32)
        for si in range(nsets):
            a = acc[2 * si, 1:-1, 1:-1]
            b = acc[2 * si + 1, 1:-1, 1:-1]
            if si == 0 and cov is not None:
                a = a * (1.0 - cov)
            tm = np.clip(b / (a + 1e-7), 0.0, 2.0)
            qa = np.sqrt(np.clip(a * np.float32(1.0 / th.TONE_AMAX), 0.0, 1.0)) * np.float32(255.0)
            noise = _dither_tile(a.shape[0], a.shape[1])[..., 0]
            qa = (qa + noise).astype(np.int32)
            np.clip(qa, 0, 255, out=qa)
            idx2 = (qa << 9) | (tm * 255.5).astype(np.int32)
            arr = np.ascontiguousarray(th.tone.take(idx2))
            img = QtGui.QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 4, QtGui.QImage.Format.Format_RGB32)
            out.append((img, arr))
        g = _Glow()
        g.f = f
        g.front = out[0]
        g.behind = out[1] if nsets > 1 else None
        return g

    # ---- scene assembly ----------------------------------------------------------------------------------
    def _build_scene(self, W: int, H: int, k: float, proj: _Proj, interactive: bool, aa: bool) -> _Scene:
        t0 = time.perf_counter()
        sc = _Scene()
        Wr = max(1, int(round(W * k)))
        Hr = max(1, int(round(H * k)))
        sc.Wr, sc.Hr, sc.k, sc.interactive = Wr, Hr, k, interactive
        cx, cy, R, w0 = self._collect_sprites(proj, k, Wr, Hr, interactive)
        buf = self._splat(cx, cy, R, w0, Wr, Hr)
        rgb, ci = self._resolve(buf, Wr, Hr, aa and not interactive)
        sc.spr_arr = rgb
        Wp = Wr + 2
        sc.spr_img = QtGui.QImage(rgb[Wp + 1:].data, Wr, Hr, Wp * 4, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        fi = self._focus_pixels(buf, ci)
        g = self._build_glow(proj, k, Wr, Hr, buf, interactive, fi if interactive else None)
        if g is not None:
            if not interactive:
                def small(src):
                    im = src[0]
                    return im.scaled(max(2, im.width() // 3), max(2, im.height() // 3),
                                     _Qt.AspectRatioMode.IgnoreAspectRatio, _Qt.TransformationMode.SmoothTransformation)

                mask = self._focus_mask(buf, fi, Wr, Hr)
                g.comp_front = self._combine_glow(g.front[0], small(g.front), g.f, Wr, Hr, mask)
                if g.behind is not None:
                    g.comp_behind = self._combine_glow(g.behind[0], small(g.behind), g.f, Wr, Hr, None)
                self._mask_keep = None
            sc.glow = g
        sc.build_ms = (time.perf_counter() - t0) * 1000.0
        return sc

    def _compose(self, sc: _Scene, op: float) -> QtGui.QImage:
        """background -> glow behind -> opaque spheres -> glow in front (screen on dark, multiply on light)."""
        base = self._bg_image(sc.Wr, sc.Hr)
        img = base.copy()
        p = QtGui.QPainter(img)
        g = sc.glow
        add = self._theme.glow_add
        mode = (QtGui.QPainter.CompositionMode.CompositionMode_Screen if add
                else QtGui.QPainter.CompositionMode.CompositionMode_Multiply)
        op = min(1.0, max(0.0, op))

        def glow_layer(main, comp):
            if op <= 0.004 or (main is None and comp is None):
                return
            p.save()
            p.setCompositionMode(mode)
            p.setOpacity(op)
            if comp is not None:
                p.drawImage(0, 0, comp)
            else:                                                     # interactive: scale the grid image directly
                p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
                p.drawImage(QtCore.QRectF(0.0, 0.0, main[0].width() * g.f, main[0].height() * g.f), main[0])
            p.restore()

        if g is not None:
            glow_layer(g.behind, g.comp_behind)
        p.drawImage(0, 0, sc.spr_img)
        if g is not None:
            glow_layer(g.front, g.comp_front)
        p.end()
        return img

    def _scene_key(self, Wr: int, Hr: int, interactive: bool):
        return (self._cam_ver, self._data_ver, self._theme_ver, Wr, Hr, interactive, self._opts["solute_style"],
                self._lod if interactive else -1)

    def _ensure_final(self, W: int, H: int, dpr: float):
        interactive = bool(self._interacting)
        if interactive:
            self._rs, self._lod = _LEVELS[self._level]
        k = dpr * (self._rs if interactive else 1.0)
        Wr = max(1, int(round(W * k)))
        Hr = max(1, int(round(H * k)))
        key = self._scene_key(Wr, Hr, interactive)
        proj = self._get_proj()
        sc = self._scene
        if sc is None or sc.key != key:
            sc = self._build_scene(W, H, k, proj, interactive, aa=True)
            sc.key = key
            self._scene = sc
            self._adapt_pending = interactive
        op = min(1.0, max(0.0, self._dn_opacity() * self._dn_fade))
        fkey = (int(round(op * 96.0)), self._theme_ver)
        if sc.final is None or sc.final_key != fkey:
            sc.final = self._compose(sc, round(op * 96.0) / 96.0)
            sc.final_key = fkey
            sc.static = None
        return sc, proj

    def _static_image(self, sc: _Scene, W: int, H: int, proj: _Proj) -> Optional[QtGui.QImage]:
        """The composed scene with the camera-dependent static overlays baked in, so that ring pulses and hover
        updates only blit one image and draw a few shapes.  Not used while interacting (overlays stay crisp)."""
        if sc.interactive:
            return None
        key = (sc.final_key, self._ovl_ver)
        if sc.static is not None and sc.static_key == key:
            return sc.static
        img = sc.final.copy()
        p = QtGui.QPainter(img)
        p.scale(img.width() / float(W), img.height() / float(H))
        self._paint_static(p, W, H, proj)
        p.end()
        sc.static, sc.static_key = img, key
        return img

    def _adapt_quality(self, ms: float) -> None:
        """Walk the interaction quality ladder so mouse-rotate frames stay near the target frame time."""
        self._ema_ms = ms if self._ema_ms <= 0 else 0.5 * self._ema_ms + 0.5 * ms
        last = len(_LEVELS) - 1
        if self._ema_ms > _TARGET_MS:
            self._viol = max(self._viol, 0) + 1
            far = self._ema_ms > 2.0 * _TARGET_MS
            if (self._viol >= 2 or far) and self._level < last:
                self._level = min(last, self._level + (1 + (self._ema_ms > 2.0 * _TARGET_MS)
                                                       + (self._ema_ms > 4.0 * _TARGET_MS)))
                self._viol = 0
                self._ema_ms = 0.0                                  # measure the new level from scratch
        elif self._ema_ms < 0.5 * _TARGET_MS:
            self._viol = min(self._viol, 0) - 1
            if self._viol <= -10 and self._level > 0:
                self._level -= 1
                self._viol = 0
                self._ema_ms = 0.0
        else:
            self._viol = 0

    # =============================================================================================== painting
    def paintEvent(self, ev) -> None:  # noqa: N802
        t0 = time.perf_counter()
        p = QtGui.QPainter(self)
        try:
            W, H = max(1, self.width()), max(1, self.height())
            dpr = float(self.devicePixelRatioF())
            sc, proj = self._ensure_final(W, H, dpr)
            baked = self._static_image(sc, W, H, proj)
            final = baked if baked is not None else sc.final
            if abs(sc.k - dpr) < 1e-6 and final.width() == int(round(W * dpr)):
                final.setDevicePixelRatio(dpr)
                p.drawImage(0, 0, final)
            else:
                p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
                p.drawImage(QtCore.QRectF(0, 0, W, H), final)
                p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
            self._set_hints(p)
            if baked is None:
                self._paint_static(p, W, H, proj)
            self._paint_dynamic(p, W, H, proj, export=False)
            self._last_error = None
        except Exception:
            self._last_error = traceback.format_exc()
            print(self._last_error, file=sys.stderr)
            p.fillRect(self.rect(), self._theme.q["bg0"])
        p.end()
        now = time.perf_counter()
        dur = (now - t0) * 1000.0
        self._last_paint_ms = dur
        if self._adapt_pending and self._interacting:
            self._adapt_pending = False
            self._adapt_quality(dur)
        dt = now - self._last_paint_t if self._last_paint_t else 1.0
        inst = 1.0 / dt if dt < 0.25 else 1000.0 / max(dur, 0.5)
        self._fps = inst if self._fps <= 0 else 0.7 * self._fps + 0.3 * inst
        self._last_paint_t = now

    # ---- overlays ----------------------------------------------------------------------------------------
    @staticmethod
    def _set_hints(p: QtGui.QPainter) -> None:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

    def _paint_static(self, p: QtGui.QPainter, W: int, H: int, proj: _Proj) -> None:
        """Overlays that only change with the camera, options or HUD text: box, axes gizmo, scale bar, HUD."""
        self._set_hints(p)
        if self._opts["show_box"] and self._box_corners is not None:
            self._draw_box(p, proj)
        if self._opts["show_axes"]:
            self._draw_gizmo(p, W, H)
        if self._opts["show_scale_bar"]:
            self._draw_scale_bar(p, W, H, proj)
        if self._hud:
            self._draw_hud(p, W, H)

    def _paint_dynamic(self, p: QtGui.QPainter, W: int, H: int, proj: _Proj, export: bool) -> None:
        """Animated / pointer dependent overlays: pulse rings, hover ring and tooltip."""
        self._set_hints(p)
        self._draw_rings(p, proj)
        if not export and self._hover is not None:
            self._draw_hover(p, W, H)
            self._draw_tooltip(p, W, H)

    def _draw_box(self, p: QtGui.QPainter, proj: _Proj) -> None:
        th = self._theme
        c = self._box_corners
        # corners ordered i,j,k in {0,1}: index = 4i + 2j + k
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        nseg = 12
        t = np.linspace(0.0, 1.0, nseg + 1)[None, :, None]
        a = np.array([c[i] for i, _ in edges])[:, None, :]
        b = np.array([c[j] for _, j in edges])[:, None, :]
        pts = (a * (1 - t) + b * t).reshape(-1, 3)
        v = (pts - self._target) @ self._Rm.T
        d = self._dist - v[:, 2]
        ok = d > proj.near
        ps = proj.scale * self._dist / np.where(ok, d, 1.0)
        sx = proj.cx + v[:, 0] * ps
        sy = proj.cy - v[:, 1] * ps
        sx = sx.reshape(12, nseg + 1)
        sy = sy.reshape(12, nseg + 1)
        d = d.reshape(12, nseg + 1)
        okm = ok.reshape(12, nseg + 1)
        dlo, dhi = proj.dwin
        dm = 0.5 * (d[:, :-1] + d[:, 1:])
        ft = np.clip((dm - dlo) / max(dhi - dlo, 1e-6), 0.0, 1.0)
        segok = okm[:, :-1] & okm[:, 1:]
        base = th.c["ink3"] if th.dark else th.c["ink2"]
        col = _mix(th.c["line"], base, 0.55)
        buckets = 4
        bi = np.minimum((ft * buckets).astype(int), buckets - 1)
        pen = QtGui.QPen()
        pen.setWidthF(1.15)
        pen.setCapStyle(_Qt.PenCapStyle.RoundCap)
        for bkt in range(buckets):
            alpha = 0.62 - 0.40 * (bkt / (buckets - 1.0))
            sel = (bi == bkt) & segok
            if not sel.any():
                continue
            ii, jj = np.nonzero(sel)
            lines = [QtCore.QLineF(float(sx[i, j]), float(sy[i, j]), float(sx[i, j + 1]), float(sy[i, j + 1]))
                     for i, j in zip(ii, jj)]
            pen.setColor(_qcol(col, alpha))
            p.setPen(pen)
            p.drawLines(lines)
        # corner dots
        v8 = (c - self._target) @ self._Rm.T
        d8 = self._dist - v8[:, 2]
        p.setPen(_Qt.PenStyle.NoPen)
        for i in range(8):
            if d8[i] <= proj.near:
                continue
            s = proj.scale * self._dist / d8[i]
            f8 = min(max((d8[i] - dlo) / max(dhi - dlo, 1e-6), 0.0), 1.0)
            p.setBrush(_qcol(col, 0.85 - 0.5 * f8))
            p.drawEllipse(QtCore.QPointF(proj.cx + v8[i, 0] * s, proj.cy - v8[i, 1] * s), 1.9, 1.9)

    def _draw_rings(self, p: QtGui.QPainter, proj: _Proj) -> None:
        if not self._n_sv or proj.svo is None:
            return
        st = self._sv_state
        ids = np.flatnonzero((st == 2) | (st == 3))
        if ids.size == 0:
            return
        sx, sy, d, ps = proj.svo
        ids = ids[d[ids] > proj.near]
        if ids.size > _MAX_RINGS:
            ids = ids[np.argsort(d[ids])[:_MAX_RINGS]]
        th = self._theme
        ph = self._pulse
        p.setBrush(_Qt.BrushStyle.NoBrush)
        pen = QtGui.QPen()
        pen.setCapStyle(_Qt.PenCapStyle.RoundCap)
        enter_col = _mix(th.c["ok"], (1, 1, 1), 0.25 if th.dark else 0.0)
        a_gain = 1.0 if th.dark else 1.35                                      # light backgrounds need more ink
        w_ring = 1.7 if th.dark else 2.1
        for i in ids:
            r0 = float(_RO[st[i]] * ps[i])
            x, y = float(sx[i]), float(sy[i])
            if st[i] == 2:
                for off in (0.0, 0.5):
                    f = (ph + off) % 1.0
                    rad = r0 * (1.25 + 2.6 * f) + 1.5
                    alpha = min(1.0, 0.95 * a_gain * (1.0 - f) ** 1.4)
                    pen.setWidthF(w_ring)
                    pen.setColor(_qcol(enter_col, alpha))
                    p.setPen(pen)
                    p.drawEllipse(QtCore.QPointF(x, y), rad, rad)
            else:
                rad = r0 * 1.75 + 2.0
                a = 0.62 + 0.28 * math.sin(2 * math.pi * ph)
                p.setBrush(_qcol(th.c["solute"], 0.12 if th.dark else 0.16))
                pen.setWidthF(1.5 if th.dark else 1.9)
                pen.setColor(_qcol(th.c["solute"], min(1.0, a * (1.0 if th.dark else 1.15))))
                p.setPen(pen)
                p.drawEllipse(QtCore.QPointF(x, y), rad, rad)
                p.setBrush(_Qt.BrushStyle.NoBrush)

    def _draw_gizmo(self, p: QtGui.QPainter, W: int, H: int) -> None:
        th = self._theme
        L = 30.0
        o = QtCore.QPointF(52.0, H - 52.0)
        if W < 160 or H < 160:
            return
        p.setPen(_Qt.PenStyle.NoPen)
        p.setBrush(_qcol(th.c["bg0"] if th.dark else th.c["bg1"], 0.34))
        p.drawEllipse(o, L + 15, L + 15)
        axes = []
        for i in range(3):
            col = self._Rm[:, i]
            axes.append((float(col[2]), float(col[0]) * L, -float(col[1]) * L, i))
        axes.sort()
        font = QtGui.QFont(self._font_ui)
        font.setPixelSize(11)
        font.setBold(True)
        p.setFont(font)
        names = ("x", "y", "z")
        pen = QtGui.QPen()
        pen.setCapStyle(_Qt.PenCapStyle.RoundCap)
        for z, dx, dy, i in axes:
            a = 0.55 + 0.45 * (z + 1.0) / 2.0
            colr = th.axis_cols[i]
            c = QtGui.QColor(colr)
            c.setAlphaF(a)
            pen.setColor(c)
            pen.setWidthF(2.0)
            p.setPen(pen)
            end = QtCore.QPointF(o.x() + dx, o.y() + dy)
            p.drawLine(o, end)
            p.setPen(_Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawEllipse(end, 2.8, 2.8)
            ln = math.hypot(dx, dy)
            ux, uy = (dx / ln, dy / ln) if ln > 1e-6 else (0.0, -1.0)
            lp = QtCore.QPointF(end.x() + ux * 9.0 - 4.0, end.y() + uy * 9.0 + 4.0)
            p.setPen(c)
            p.drawText(lp, names[i])
        p.setBrush(_qcol(th.ink2, 0.9))
        p.setPen(_Qt.PenStyle.NoPen)
        p.drawEllipse(o, 2.0, 2.0)

    def _draw_scale_bar(self, p: QtGui.QPainter, W: int, H: int, proj: _Proj) -> None:
        th = self._theme
        scale = proj.scale
        if scale <= 1e-9 or W < 200:
            return
        target = 118.0 / scale
        n = _nice_length(target)
        Lpx = n * scale
        if Lpx < 20:
            return
        x1 = W - 18.0
        x0 = x1 - Lpx
        y = H - 24.0
        p.setFont(self._font_ui)
        label = f"{n:g} Å"
        fm = QtGui.QFontMetricsF(self._font_ui)
        tw = fm.horizontalAdvance(label)
        for col, off in ((th.shadow_col, 1.0), (th.q_ink2, 0.0)):
            pen = QtGui.QPen(col)
            pen.setWidthF(2.0 if off == 0.0 else 3.0)
            pen.setCapStyle(_Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(x0, y + off), QtCore.QPointF(x1, y + off))
            pen.setWidthF(1.5 if off == 0.0 else 2.5)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(x0, y - 4 + off), QtCore.QPointF(x0, y + 4 + off))
            p.drawLine(QtCore.QPointF(x1, y - 4 + off), QtCore.QPointF(x1, y + 4 + off))
        tx = x1 - tw
        ty = y - 9.0
        p.setPen(th.shadow_col)
        p.drawText(QtCore.QPointF(tx + 1, ty + 1), label)
        p.setPen(th.q_ink2)
        p.drawText(QtCore.QPointF(tx, ty), label)

    def _draw_hud(self, p: QtGui.QPainter, W: int, H: int) -> None:
        th = self._theme
        p.setFont(self._font_hud)
        fm = QtGui.QFontMetricsF(self._font_hud)
        maxw = max(60.0, W * 0.6)
        lines = [fm.elidedText(s, _Qt.TextElideMode.ElideRight, maxw) for s in self._hud[:14]]
        if not lines:
            return
        lh = fm.height() + 1.0
        w = max(fm.horizontalAdvance(s) for s in lines) + 22.0
        h = lh * len(lines) + 14.0
        r = QtCore.QRectF(12.0, 12.0, w, h)
        p.setPen(QtGui.QPen(th.plate_line, 1.0))
        p.setBrush(th.plate)
        p.drawRoundedRect(r, 8.0, 8.0)
        y = r.top() + 7.0 + fm.ascent()
        for i, s in enumerate(lines):
            p.setPen(th.q_ink if i == 0 else th.q_ink2)
            p.drawText(QtCore.QPointF(r.left() + 11.0, y), s)
            y += lh

    def _draw_hover(self, p: QtGui.QPainter, W: int, H: int) -> None:
        hv = self._hover
        if hv is None:
            return
        th = self._theme
        x, y, r = hv["sx"], hv["sy"], hv["r"]
        kind = hv["kind"]
        col = th.c["solvent"] if kind == "solvent" else (th.c["solute"] if kind == "solute" else th.c["accent"])
        p.setBrush(_Qt.BrushStyle.NoBrush)
        pen = QtGui.QPen(th.shadow_col)
        pen.setWidthF(3.4)
        p.setPen(pen)
        p.drawEllipse(QtCore.QPointF(x, y), r, r)
        pen = QtGui.QPen(_qcol(_tint(col, 0.15)))
        pen.setWidthF(1.6)
        p.setPen(pen)
        p.drawEllipse(QtCore.QPointF(x, y), r, r)

    def _draw_tooltip(self, p: QtGui.QPainter, W: int, H: int) -> None:
        hv = self._hover
        if hv is None:
            return
        th = self._theme
        fm = QtGui.QFontMetricsF(self._font_tip)
        fm2 = QtGui.QFontMetricsF(self._font_hud)
        t1 = fm.elidedText(hv["label"], _Qt.TextElideMode.ElideRight, 260.0)
        pos = hv["pos"]
        t2 = f"{pos[0]:.1f}  {pos[1]:.1f}  {pos[2]:.1f} Å"
        w = max(fm.horizontalAdvance(t1), fm2.horizontalAdvance(t2)) + 20.0
        h = fm.height() + fm2.height() + 14.0
        mp = self._mouse_pos
        mx, my = (mp.x(), mp.y()) if mp is not None else (hv["sx"], hv["sy"])
        x = mx + 16.0
        y = my + 18.0
        if x + w > W - 6:
            x = mx - 16.0 - w
        if y + h > H - 6:
            y = my - 18.0 - h
        x = max(6.0, x)
        y = max(6.0, y)
        r = QtCore.QRectF(x, y, w, h)
        p.setPen(QtGui.QPen(th.plate_line, 1.0))
        bg = _qcol(th.c["bg0"] if th.dark else th.c["bg1"], 0.94)
        p.setBrush(bg)
        p.drawRoundedRect(r, 7.0, 7.0)
        p.setFont(self._font_tip)
        p.setPen(th.q_ink)
        p.drawText(QtCore.QPointF(x + 10.0, y + 6.0 + fm.ascent()), t1)
        p.setFont(self._font_hud)
        p.setPen(th.q_ink3)
        p.drawText(QtCore.QPointF(x + 10.0, y + 6.0 + fm.height() + 2.0 + fm2.ascent()), t2)

    # =============================================================================================== picking
    def _request_pick(self) -> None:
        now = time.perf_counter()
        if now - self._last_pick_t >= 1.0 / 60.0 and not self._hover_timer.isActive():
            self._do_hover()
        elif not self._hover_timer.isActive():
            self._hover_timer.start(17)

    @_guarded
    def _do_hover(self) -> None:
        self._last_pick_t = time.perf_counter()
        mp = self._mouse_pos
        if mp is None or self._drag_mode is not None:
            return
        hit = self._pick(mp.x(), mp.y())
        key = (hit["kind"], hit["index"]) if hit else None
        if key != self._hover_key:
            self._hover_key = key
            self._hover = hit.get("_ui") if hit else None
            self.hovered.emit(self._public(hit) if hit else {})
            self.update()
        elif hit:
            self.update()                                   # tooltip follows the cursor

    def _pick(self, x: float, y: float) -> Dict[str, Any]:
        proj = self._get_proj()
        # candidate = (score, depth, kind, array index, sx, sy, radius); score 0 = cursor inside the item's disc
        cands: List[Tuple[float, float, str, int, float, float, float]] = []

        def scan(kind: str, layer, rad_px: np.ndarray, thr_min: float, pref: float):
            sx, sy, d, ps = layer
            dx = sx - np.float32(x)
            dy = sy - np.float32(y)
            d2 = dx * dx + dy * dy
            thr = np.maximum(rad_px, thr_min)
            near = (d2 <= thr * thr) & (d > proj.near)
            ids = np.flatnonzero(near)
            if ids.size == 0:
                return
            dist = np.sqrt(d2[ids])
            inside = dist <= rad_px[ids]
            if inside.any():
                ii = ids[inside]
                jj = ii[np.argmin(d[ii])]
                cands.append((0.0, float(d[jj]), kind, int(jj), float(sx[jj]), float(sy[jj]), float(rad_px[jj])))
            else:
                jj = ids[np.argmin(dist)]
                cands.append((1.0 + pref + float(dist.min()) / max(float(thr[jj]), 1.0), float(d[jj]), kind, int(jj),
                              float(sx[jj]), float(sy[jj]), float(rad_px[jj])))

        if self._n_sv and proj.svo is not None:
            st = self._sv_state
            rad = (_RO[st] * proj.svo[3]).astype(np.float32)
            scan("solvent", proj.svo, rad, 10.0, 0.0)
        if self._n_sol and proj.sol is not None:
            rad = (self._sol_rad_eff * proj.sol[3]).astype(np.float32)
            scan("solute", proj.sol, rad, 9.0, 0.3)
        if not cands and self._n_dn and proj.dn is not None:
            rad = np.full(self._n_dn, 3.0, np.float32)
            scan("density", proj.dn, rad, 8.0, 0.6)
        if not cands:
            return {}
        # inside-disc candidates win by depth; otherwise by normalised distance
        inside = [c for c in cands if c[0] == 0.0]
        pool = inside if inside else cands
        c = min(pool, key=lambda t: (t[1] if inside else t[0]))
        _, depth, kind, i, sx, sy, r = c
        return self._payload(kind, i, sx, sy, r)

    def _payload(self, kind: str, i: int, sx: float, sy: float, r: float) -> Dict[str, Any]:
        """Public payload {kind, index, label, pos, ...} (label exactly as supplied) + private ``_ui`` for the overlay."""
        extra: Dict[str, Any] = {}
        if kind == "solvent":
            orig = int(self._sv_orig[i]) if self._sv_orig is not None else i
            lab = None
            if self._sv_labels is not None and 0 <= orig < len(self._sv_labels):
                lab = str(self._sv_labels[orig])
            st = int(self._sv_state[i])
            label = lab if lab else f"Water {orig}"
            shown = f"{label}  ·  {_STATE_NAMES[st]}"
            pos = self._sv_o[i]
            extra["state"] = st
        elif kind == "solute":
            orig = int(self._sol_orig[i]) if self._sol_orig is not None else i
            lab = None
            if self._sol_labels is not None and 0 <= orig < len(self._sol_labels):
                lab = str(self._sol_labels[orig])
            role = int(self._sol_role[i])
            label = lab if lab else f"Atom {orig}"
            shown = label + ("  ·  zone anchor" if role == 1 else "")
            pos = self._sol_xyz[i]
            extra["role"] = role
        else:
            orig = int(self._dn_orig[i]) if self._dn_orig is not None else i
            val = float(self._dn_val[i])
            label = f"Density {val:.3g}"
            shown = label
            pos = self._dn_xyz[i]
            extra["value"] = val
        pt = (float(pos[0]), float(pos[1]), float(pos[2]))
        ui = dict(kind=kind, sx=sx, sy=sy, r=max(r * 1.25 + 2.0, 6.0), label=shown, pos=pt)
        out = {"kind": kind, "index": orig, "label": label, "pos": pt}
        out.update(extra)
        out["_ui"] = ui
        return out


# =================================================================================================== free helpers
def _scatter(buf: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    if _SCATTER_ORDERED:
        buf[idx] = vals
    else:                                                          # exact but slower z-min scatter
        np.minimum.at(buf, idx, vals)


def _dither_tile(H: int, W: int) -> np.ndarray:
    reps_y = (H + 63) // 64
    reps_x = (W + 63) // 64
    t = np.tile(_DITHER, (reps_y, reps_x, 1))[:H, :W, :]
    return t


_AVG_MASK = np.uint32(0xFEFEFEFE)


def _aa_sprite_edges(buf: np.ndarray, rgb: np.ndarray, Wr: int) -> None:
    """Soften the contour where a sprite overlaps a farther one (in place, premultiplied ``rgb``).

    Sprites have a flat depth, so two neighbouring pixels with different depth belong to different sprites.  The
    pixel on the FAR side of such a boundary is averaged with its nearer neighbour, i.e. the near sprite's outline
    is anti-aliased against whatever lies behind it - the same model as the silhouette pass."""
    Wp = Wr + 2
    dq = np.ascontiguousarray(buf.view(np.uint16)[1::2])
    cov = dq != 0xFFFF
    for step in (1, Wp):
        a, b = dq[:-step], dq[step:]
        m = (a != b) & cov[:-step] & cov[step:]
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        left_far = a[idx] > b[idx]
        far = np.where(left_far, idx, idx + step)
        near = np.where(left_far, idx + step, idx)
        cf = rgb[far]
        cn = rgb[near]
        rgb[far] = ((cf & _AVG_MASK) >> np.uint32(1)) + ((cn & _AVG_MASK) >> np.uint32(1))


def _aa_edges(buf: np.ndarray, rgb: np.ndarray, Wr: int, Hr: int) -> None:
    """Anti-alias the silhouette of the opaque layer in place (padded layout, premultiplied ``rgb``).

    Coverage = pixel touched by a sprite.  Inner edge pixels keep most of their colour, the empty pixels next to
    the silhouette receive a faint copy of the neighbouring colour: a 1.5 px soft edge at negligible cost."""
    Wp = Wr + 2
    Hp = Hr + 2
    c8 = (buf != _EMPTY).view(np.uint8).reshape(Hp, Wp)
    nbp = np.zeros((Hp, Wp), np.uint8)
    inner = nbp[1:-1, 1:-1]
    np.add(c8[:-2, 1:-1], c8[2:, 1:-1], out=inner)
    inner += c8[1:-1, :-2]
    inner += c8[1:-1, 2:]
    ei = np.flatnonzero(nbp != (c8 << 2))
    if ei.size == 0:
        return
    cf = c8.reshape(-1)
    nf = nbp.reshape(-1)
    ecov = cf[ei]
    enb = nf[ei].astype(np.float32)
    col = rgb[ei]
    outer = ecov == 0
    if outer.any():
        oi = ei[outer]
        nbr = oi[:, None] + np.array([-1, 1, -Wp, Wp], np.int64)[None, :]
        cov4 = cf[nbr]
        j = cov4.argmax(axis=1)
        col = col.copy()
        col[outer] = rgb[nbr[np.arange(oi.size), j]]
    a = np.where(ecov == 1, np.where(enb == 0, 0.80, (4.0 + enb) * 0.125), enb * 0.125).astype(np.float32)
    r = ((col >> np.uint32(16)) & np.uint32(255)).astype(np.float32)
    g = ((col >> np.uint32(8)) & np.uint32(255)).astype(np.float32)
    b = (col & np.uint32(255)).astype(np.float32)
    rgb[ei] = (((a * 255.0 + 0.5).astype(np.uint32) << np.uint32(24)) | ((r * a + 0.5).astype(np.uint32) << np.uint32(16))
               | ((g * a + 0.5).astype(np.uint32) << np.uint32(8)) | (b * a + 0.5).astype(np.uint32))
