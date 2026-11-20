"""Software-rendered 3D scene widget for SOZLab (no OpenGL, no WebEngine).

``SceneView`` shows a solute (up to ~30k atoms as depth-cued shaded spheres or points), solvent molecules whose
per-frame *state* drives their look (0 bulk / 1 in zone / 2 just entered / 3 just left), a weighted density cloud
rendered as soft volumetric glow, the periodic cell, an axes gizmo, a scale bar in angstrom and a small HUD.

Rendering model (numpy + QPainter/QImage only):

* every point is projected once per camera/data version with numpy (perspective, rotation from a quaternion);
* opaque spheres (solute atoms, solvent O/H) are splatted as pre-shaded disc sprites into ONE packed ``uint32``
  buffer ``(depth16 << 16) | lut_index16`` using an exact z-test.  Sprites are processed near -> far in growing
  stages and hidden sprites are culled with a cheap 5-point visibility test, so a 30k atom globule costs about
  the same as its visible surface.  The LUT (colour class x fog level x shade) is resolved in one gather;
* silhouettes are anti-aliased with a small coverage filter (final quality only);
* the density cloud and the zone halos are accumulated into a low-resolution float grid (bilinear splat),
  blurred, tone-mapped and composited by Qt (additive on dark backgrounds, multiply on light ones), occluded
  by the opaque depth buffer;
* overlays (box, gizmo, scale bar, HUD, rings, hover) are drawn with anti-aliased QPainter calls.

While the user interacts the scene is drawn at reduced density / resolution and re-rendered at full quality
~120 ms after the last interaction.  All colours derive from the theme tokens given to ``set_theme``.
"""
from __future__ import annotations

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
_MAX_DENSITY = 80000
_MAX_RINGS = 400
_SOLUTE_RADIUS_SCALE = 0.86          # display radius = given radius * this (keeps atoms legible)
_NEAR_FRAC = 0.12                    # near clip plane as a fraction of the camera distance
_ISO_ANGLES = (math.radians(27.0), math.radians(-38.0))   # (pitch about x, yaw about y) of the default view

# colour classes of the sprite LUT
(C_SOL0, C_SOL1, C_SOL2, C_ANCHOR, C_DIM, C_BULK_O, C_BULK_H, C_ZONE_O, C_ZONE_H,
 C_ENTER_O, C_ENTER_H, C_GHOST) = range(12)
_NCLS = 12

# solvent state -> display radius (Angstrom) and colour class of the oxygen / hydrogen sprite
_RO = np.array([0.60, 1.02, 1.14, 0.70], np.float32)
_RH = np.array([0.34, 0.44, 0.46, 0.30], np.float32)
_CLS_O = np.array([C_BULK_O, C_ZONE_O, C_ENTER_O, C_GHOST], np.int32)
_CLS_H = np.array([C_BULK_H, C_ZONE_H, C_ENTER_H, C_GHOST], np.int32)
_STATE_NAMES = ("bulk", "in zone", "entered", "left")


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
        sol = _mix(ink2, bg_mid, 0.10 if dark else 0.06)
        anchor = _tint(c["solute"], 0.10)
        specs: List[Optional[_ClsSpec]] = [None] * _NCLS
        specs[C_SOL0] = _ClsSpec(sol, 0.78, 0.55, 0.20 if dark else 0.34)
        specs[C_SOL1] = _ClsSpec(_tint(sol, 0.10), 0.78, 0.55, 0.20 if dark else 0.34)
        specs[C_SOL2] = _ClsSpec(_shade(sol, 0.10), 0.78, 0.55, 0.20 if dark else 0.34)
        specs[C_ANCHOR] = _ClsSpec(anchor, 0.38, 0.80, 0.26)
        specs[C_DIM] = _ClsSpec(_mix(sol, bg_mid, 0.55), 0.70, 0.22, 0.30)
        specs[C_BULK_O] = _ClsSpec(_mix(c["solvent"], bg_mid, 0.52 if dark else 0.38), 0.55, 0.35, 0.30)
        specs[C_BULK_H] = _ClsSpec(_mix(ink3, bg_mid, 0.40), 0.55, 0.25, 0.30)
        specs[C_ZONE_O] = _ClsSpec(_tint(c["solvent"], 0.10 if dark else 0.0), 0.30, 0.95, 0.26)
        specs[C_ZONE_H] = _ClsSpec(_tint(c["solvent"], 0.62 if dark else 0.55), 0.30, 0.70, 0.30)
        specs[C_ENTER_O] = _ClsSpec(_tint(_mix(c["ok"], c["solvent"], 0.25), 0.45 if dark else 0.25), 0.20, 0.95, 0.30)
        specs[C_ENTER_H] = _ClsSpec(_tint(c["ok"], 0.75), 0.20, 0.70, 0.34)
        specs[C_GHOST] = _ClsSpec(_mix(c["solute"], bg_mid, 0.30), 0.50, 0.30, 0.30)
        self.specs = specs
        self.lut = _build_lut(specs, fog, dark)  # type: ignore[arg-type]

        # ---- density glow ramp (value 0..1 -> colour) and blend mode
        if dark:
            c0 = _shade(_mix(c["accent"], (0.10, 0.22, 0.85), 0.35), 0.42)         # deep blue
            c1 = c["solvent"]                                                      # aquamarine
            c2 = _tint(_mix(ink, c["warn"], 0.20), 0.22)                           # warm white
        else:
            c0 = _tint(c["solvent"], 0.55)                                         # pale aqua
            c1 = _shade(c["solvent"], 0.12)                                        # teal
            c2 = _mix(c["accent"], ink, 0.55)                                      # deep indigo
        tt = np.linspace(0.0, 1.0, 256)
        stops = np.array([0.0, 0.5, 1.0])
        self.ramp = np.stack([np.interp(tt, stops, [c0[i], c1[i], c2[i]]) for i in range(3)], axis=1).astype(np.float32)
        self.glow_add = dark

        # ---- overlay colours
        self.q = {k: _qcol(v) for k, v in c.items()}
        self.q_ink, self.q_ink2, self.q_ink3 = _qcol(ink), _qcol(ink2), _qcol(ink3)
        self.plate = _qcol(c["bg0"] if dark else c["bg1"], 0.70 if dark else 0.80)
        self.plate_line = _qcol(c["line"], 0.85)
        self.shadow_col = _qcol(bg_mid * (0.4 if dark else 1.0), 0.85)
        self.axis_cols = (_qcol(c["err"]), _qcol(c["ok"]), _qcol(c["accent"]))


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

    __slots__ = ("key", "Wr", "Hr", "k", "spr_img", "spr_arr", "glow_img", "glow_arr", "glow_f", "bloom_img",
                 "final", "final_key", "interactive", "build_ms")

    def __init__(self) -> None:
        self.glow_img = None
        self.glow_arr = None
        self.bloom_img = None
        self.final = None
        self.final_key = None
        self.build_ms = 0.0


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
        self._sol_sub: Optional[np.ndarray] = None
        self._sol_sub_comp = 1.0
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
        self._dn_sub: Optional[np.ndarray] = None
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
        self._interacting = False
        self._rs = 1.0                       # adaptive interactive render scale
        self._fps = 0.0
        self._last_paint_ms = 0.0
        self._last_paint_t = 0.0
        self._dn_fade = 1.0                  # animated density opacity multiplier
        self._dn_fade_target = 1.0
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
        self._drag_start = QtCore.QPointF()
        self._drag_moved = 0.0
        self._press_t = 0.0
        self._ball_prev = np.array([0.0, 0.0, 1.0])
        self._vel_axis = np.array([0.0, 1.0, 0.0])
        self._vel = 0.0
        self._vel_t = 0.0
        self._skip_click = False

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
        self._theme = _Theme(dict(tokens or {}))
        self._theme_ver += 1
        self._bg_cache.clear()
        self._rebuild_solute_classes()
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
            self._sol_sub = None
            return
        rng = np.random.default_rng(11)
        var = rng.integers(0, 3, n).astype(np.uint8)
        cls = var.copy()
        cls[self._sol_role == 1] = C_ANCHOR
        cls[self._sol_role >= 2] = C_DIM
        self._sol_cls = cls
        self._sol_rad_eff = (self._sol_rad * _SOLUTE_RADIUS_SCALE).astype(np.float32)
        if n > _INTERACTIVE_SOLUTE:
            rank = rng.random(n)
            keep = rank < (_INTERACTIVE_SOLUTE / float(n))
            keep |= (self._sol_role == 1)
            self._sol_sub = np.flatnonzero(keep)
            self._sol_sub_comp = float(min(1.35, (n / max(1.0, self._sol_sub.size)) ** (1.0 / 3.0)))
        else:
            self._sol_sub = None
            self._sol_sub_comp = 1.0

    # ------------------------------------------------------------------------------------------ solvent data
    def set_solvent(self, oxygen_xyz, hydrogen_xyz=None, state=None, labels=None) -> None:
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
        rng = vmax - vmin
        t = np.clip((v - vmin) / rng, 0.0, 1.0) if rng > 1e-12 else np.ones_like(v)
        self._n_dn = a.shape[0]
        self._dn_xyz = np.ascontiguousarray(a, np.float32)
        self._dn_val = v.astype(np.float32)
        self._dn_t = t.astype(np.float32)
        self._dn_emit = (0.10 + 0.90 * self._dn_t ** 1.15).astype(np.float32)
        self._dn_orig = orig
        self._dn_vrange = (vmin, vmax)
        self._dn_h = _estimate_spacing(self._dn_xyz) if self._n_dn > 1 else 1.0
        if self._n_dn > _INTERACTIVE_DENSITY:
            keep = np.random.default_rng(5).random(self._n_dn) < (_INTERACTIVE_DENSITY / float(self._n_dn))
            self._dn_sub = np.flatnonzero(keep)
        else:
            self._dn_sub = None
        try:
            self._dn_op_data = float(min(max(opacity, 0.0), 1.0))
        except Exception:
            self._dn_op_data = 0.85
        self._invalidate_data(bounds=True)
        self._start_density_fade(from_zero=True)

    def clear_density(self) -> None:
        self._n_dn = 0
        self._dn_xyz = np.zeros((0, 3), np.float32)
        self._dn_t = self._dn_val = self._dn_emit = np.zeros(0, np.float32)
        self._dn_orig = None
        self._dn_sub = None
        self._fade_anim.stop()
        self._invalidate_data(bounds=True)

    # --------------------------------------------------------------------------------------------- the box
    def set_box(self, dims, origin=None) -> None:
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
        self._invalidate_data(bounds=True, scene=False)

    def set_hud(self, lines: List[str]) -> None:
        self._hud = [str(s) for s in (lines or [])]
        self.update()

    def set_options(self, show_box: bool = True, show_axes: bool = True, solute_style: str = "spheres",
                    density_opacity: Optional[float] = None, reduce_motion: bool = False,
                    show_scale_bar: bool = True) -> None:
        style = str(solute_style).lower() if solute_style else "spheres"
        if style not in ("spheres", "points", "ribbon-points"):
            style = "spheres"
        old_eff = self._dn_opacity()
        style_changed = style != self._opts["solute_style"]
        self._opts.update(show_box=bool(show_box), show_axes=bool(show_axes), solute_style=style,
                          reduce_motion=bool(reduce_motion), show_scale_bar=bool(show_scale_bar),
                          density_opacity=None if density_opacity is None else float(min(max(density_opacity, 0.0), 1.0)))
        if self._opts["reduce_motion"]:
            self._cam_anim.stop()
            self._fade_anim.stop()
            self._dn_fade = 1.0
            self._inertia_timer.stop()
        self._update_pulse_running()
        if style_changed:
            self._invalidate_data(scene=True)
        new_eff = self._dn_opacity()
        if self._n_dn and abs(new_eff - old_eff) > 1e-4 and not self._opts["reduce_motion"]:
            # cross-fade the glow from its old to its new strength
            self._dn_fade = old_eff / new_eff if new_eff > 1e-6 else 0.0
            self._start_density_fade(from_zero=False)
        else:
            self.update()

    # ------------------------------------------------------------------------------------------------ camera
    def reset_view(self, animate: bool = True) -> None:
        tgt, dist, zoom = self._frame_params()
        self._auto_frame = False
        q = _preset_quat("iso")
        self._framed_center, self._framed_radius = self._scene_bounds()
        self._cor = None
        self._goto(dict(q=q, zoom=zoom, target=tgt, dist=dist), animate)

    def set_view_preset(self, name: str, animate: bool = True) -> None:
        q = _preset_quat(name)
        if q is None:
            return
        self._auto_frame = False
        cur = self._cam_dict()
        cur["q"] = q
        self._goto(cur, animate)

    def focus_on(self, xyz, animate: bool = True) -> None:
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
        if xyz is None:
            self._cor = None
            return
        a = self._as_points(xyz)
        if a.shape[0]:
            a = a[np.isfinite(a).all(axis=1)]
        self._cor = a.astype(np.float64).mean(axis=0) if a.shape[0] else None

    def camera_state(self) -> Dict[str, Any]:
        return {"quat": [float(x) for x in self._q], "zoom": float(self._zoom),
                "center": [float(x) for x in self._target], "distance": float(self._dist),
                "cor": None if self._cor is None else [float(x) for x in self._cor]}

    def set_camera_state(self, state: Dict[str, Any], animate: bool = False) -> None:
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
        return self._pick(float(x), float(y))

    def set_pulse_phase(self, phase: float) -> None:
        try:
            self._pulse = float(phase) % 1.0
        except Exception:
            return
        self.update()

    def stats(self) -> Dict[str, Any]:
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

    # ------------------------------------------------------------------------------------------ export
    def render_to_image(self, width: int, height: int) -> QtGui.QImage:
        w = max(1, int(width))
        h = max(1, int(height))
        ss = 1.5
        proj = self._compute_proj(w, h)
        sc = self._build_scene(w, h, ss, proj, interactive=False, aa=False)
        final = self._compose(sc, self._dn_opacity())
        p = QtGui.QPainter(final)
        p.scale(sc.Wr / float(w), sc.Hr / float(h))
        self._paint_overlays(p, w, h, proj, export=True)
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

    def _invalidate_data(self, bounds: bool = False, scene: bool = True) -> None:
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
    def _scene_bounds(self) -> Tuple[np.ndarray, float]:
        mins, maxs = [], []
        layers = []
        if self._n_sol:
            layers.append(self._sol_xyz)
        if self._n_sv:
            layers.append(self._sv_o)
        if self._n_dn:
            layers.append(self._dn_xyz)
        if self._box_corners is not None:
            layers.append(self._box_corners.astype(np.float32))
        if not layers:
            return np.zeros(3), 10.0
        for a in layers:
            mins.append(a.min(axis=0))
            maxs.append(a.max(axis=0))
        lo = np.min(np.array(mins, np.float64), axis=0)
        hi = np.max(np.array(maxs, np.float64), axis=0)
        c = (lo + hi) * 0.5
        r = 0.0
        for a in layers:
            r = max(r, float(np.sqrt(((a - c.astype(np.float32)) ** 2).sum(axis=1).max())))
        return c, max(r, 4.0)

    def _frame_params(self) -> Tuple[np.ndarray, float, float]:
        c, r = self._scene_bounds()
        dist = max(5.0 * r, 14.0)
        zoom = 0.90 * math.sqrt(max(dist * dist - r * r, 1e-9)) / (r * dist)
        return c, dist, zoom

    def _maybe_auto_frame(self) -> None:
        if not self._auto_frame:
            return
        c, r = self._scene_bounds()
        if (self._n_sol + self._n_sv + self._n_dn) == 0 and self._box_corners is None:
            return
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
            self._settle_soon()

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

    def _on_cam_anim_done(self) -> None:
        b = self._cam_to
        if b is not None:
            self._set_cam(b["q"], b["zoom"], b["target"], b["dist"])
        self._settle_soon()

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
        self._interacting = True
        self._settle_timer.stop()
        if hold:
            self._cam_anim.stop()

    def _settle_soon(self) -> None:
        self._interacting = True
        self._settle_timer.start(_SETTLE_MS)

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
    def resizeEvent(self, ev) -> None:  # noqa: N802
        self._proj = None
        self._scene = None
        super().resizeEvent(ev)

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        self._update_pulse_running()

    def hideEvent(self, ev) -> None:  # noqa: N802
        super().hideEvent(ev)
        self._pulse_anim.pause() if self._pulse_anim.state() == QtCore.QAbstractAnimation.State.Running else None

    def leaveEvent(self, ev) -> None:  # noqa: N802
        self._mouse_pos = None
        if self._hover is not None or self._hover_key is not None:
            self._hover = None
            self._hover_key = None
            self.hovered.emit({})
            self.update()
        super().leaveEvent(ev)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        self.setFocus(_Qt.FocusReason.MouseFocusReason)
        pos = ev.position()
        self._mouse_pos = pos
        btn = ev.button()
        if btn == _Qt.MouseButton.LeftButton or btn == _Qt.MouseButton.MiddleButton:
            shift = bool(ev.modifiers() & _Qt.KeyboardModifier.ShiftModifier)
            self._drag_mode = "pan" if (btn == _Qt.MouseButton.MiddleButton or shift) else "rot"
            self._drag_last = pos
            self._drag_start = pos
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
                inst = ang / min(dt, 0.1)
                self._vel = 0.6 * self._vel + 0.4 * inst if self._vel else inst
                self._vel_axis = axis
            self._vel_t = now
            self._ball_prev = v1
        self._drag_last = pos
        self._settle_soon()
        ev.accept()

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
        if click and not self._skip_click:
            self.clicked.emit(self._pick(pos.x(), pos.y()))
        self._skip_click = False
        if (mode == "rot" and not click and not self._opts["reduce_motion"]
                and (time.perf_counter() - self._vel_t) < 0.06 and self._vel > 0.8):
            self._vel = min(self._vel, 9.0)
            self._inertia_last = time.perf_counter()
            self._inertia_timer.start()
        self._vel_t = 0.0
        self._settle_soon()
        self._request_pick()
        ev.accept()

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._skip_click = True
        self._inertia_timer.stop()
        self.reset_view(animate=True)
        ev.accept()

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

    def _inertia_tick(self) -> None:
        now = time.perf_counter()
        dt = min(now - getattr(self, "_inertia_last", now), 0.05)
        self._inertia_last = now
        self._vel *= math.exp(-dt / 0.28)
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

    def _on_fade_anim(self, v) -> None:
        self._dn_fade = float(v)
        self.update()

    def _update_pulse_running(self) -> None:
        need = (not self._opts["reduce_motion"]) and bool(((self._sv_state == 2) | (self._sv_state == 3)).any()) \
            if self._n_sv else False
        running = self._pulse_anim.state() == QtCore.QAbstractAnimation.State.Running
        if need and self.isVisible():
            if not running:
                self._pulse_anim.start()
        elif running:
            self._pulse_anim.stop()
        elif need and not self.isVisible() and self._pulse_anim.state() == QtCore.QAbstractAnimation.State.Paused:
            pass
        if self._opts["reduce_motion"]:
            self._pulse = 0.35

    def _on_pulse_anim(self, v) -> None:
        now = time.perf_counter()
        if now - self._pulse_last_update < 0.027:        # ~35 Hz is plenty for a ring
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
        cached = getattr(self, "_bounds_cache", None)
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
        rgb = rgb + (_DITHER_TILE(Hr, Wr) - 0.5) / 255.0
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

        def finish(sx, sy, d, rpx, cls, fogscale, rmin, rmax, extra_valid=None):
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
            lvl = (ft * fogscale * (_FOG - 1) + 0.5).astype(np.int32)
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
            fogs = np.where(self._sol_role == 1, 0.45, 1.0).astype(np.float32)
            comp = 1.0
            if interactive and self._sol_sub is not None:
                sub = self._sol_sub
                sx, sy, d, ps, rad, cls, fogs = sx[sub], sy[sub], d[sub], ps[sub], rad[sub], cls[sub], fogs[sub]
                comp = self._sol_sub_comp
            ft = np.clip((d - dlo) * inv, 0.0, 1.0)
            shrink = 1.0 - 0.07 * ft
            rpx = rad * ps * (k * comp) * shrink
            if style == "spheres":
                finish(sx, sy, d, rpx, cls, fogs, 1, _MAXR)
            else:
                fs = fogs * (1.35 if style == "ribbon-points" else 1.0)
                finish(sx, sy, d, rpx * 0.34, cls, np.minimum(fs, 1.0), 0, 2)

        # ---- solvent
        if self._n_sv and proj.svo is not None:
            st = self._sv_state
            sx, sy, d, ps = proj.svo
            rpx = _RO[st] * ps * k
            drawn = st != 3
            fs = np.where(st == 0, 1.0, 0.5).astype(np.float32)
            rmin = np.where(st == 0, 0, 1)
            finish(sx, sy, d, rpx, _CLS_O[st], fs, 0, _MAXR, extra_valid=drawn)
            if self._sv_hvalid.any() and proj.svh is not None:
                st2 = np.repeat(st, 2)
                hv = np.repeat(self._sv_hvalid, 2) & (st2 != 3)
                sxh, syh, dh, psh = proj.svh
                rph = _RH[st2] * psh * k
                # bulk hydrogens are skipped when they would be sub-pixel specks
                hv &= ~((st2 == 0) & (rph < 1.0))
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
        npx = Wr * Hr
        buf = np.full(npx, _EMPTY, np.uint32)
        n = cx.size
        if n == 0:
            return buf
        order = np.argsort((w0 >> np.uint32(16)).astype(np.uint16), kind="stable")      # near -> far (radix sort)
        cx, cy, R, w0 = cx[order], cy[order], R[order], w0[order]
        bank = self._bank
        pos = 0
        stage = _STAGE0
        while pos < n:
            end = min(n, pos + stage)
            scx, scy, sR, sw = cx[pos:end], cy[pos:end], R[pos:end], w0[pos:end]
            if pos > 0:
                scx, scy, sR, sw = self._cull(buf, scx, scy, sR, sw, Wr, Hr)
            m = scx.size
            if m:
                R8 = sR.astype(np.uint8)
                o2 = np.argsort(R8, kind="stable")
                scx, scy, sR, sw, R8 = scx[o2], scy[o2], sR[o2], sw[o2], R8[o2]
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
        """Drop sprites whose centre and four inner points all lie behind already drawn nearer pixels."""
        big = sR >= 3
        if not big.any():
            return scx, scy, sR, sw
        h = (sR * 0.55).astype(np.int64)
        ok = big & (scx >= h) & (scx < Wr - h) & (scy >= h) & (scy < Hr - h)
        if not ok.any():
            return scx, scy, sR, sw
        thr = (sw >> np.uint32(16)) << np.uint32(16)
        npx = buf.size
        lin = np.clip(scy * Wr + scx, 0, npx - 1)
        hid = buf[lin] < thr
        hid &= buf[np.clip(lin + h, 0, npx - 1)] < thr
        hid &= buf[np.clip(lin - h, 0, npx - 1)] < thr
        hid &= buf[np.clip(lin + h * Wr, 0, npx - 1)] < thr
        hid &= buf[np.clip(lin - h * Wr, 0, npx - 1)] < thr
        hid &= ok
        if not hid.any():
            return scx, scy, sR, sw
        keep = ~hid
        return scx[keep], scy[keep], sR[keep], sw[keep]

    @staticmethod
    def _draw_group(buf, Rv: int, gcx, gcy, gw, Wr: int, Hr: int, bank: _SpriteBank) -> None:
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
            off = bank.offsets(Rv, Wr)
            base = ins_cy * Wr + ins_cx
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
                idx = (ys * Wr + xs)[ok]
                pk = (bw[c0:c0 + ch, None] + tidx)[ok] if tidx.ndim == 1 else None
                msk = pk < buf[idx]
                if msk.any():
                    _scatter(buf, idx[msk], pk[msk])

    # ---- resolve + anti-alias ----------------------------------------------------------------------------
    def _resolve(self, buf: np.ndarray, Wr: int, Hr: int, aa: bool) -> np.ndarray:
        lut = self._theme.lut
        rgb = lut.take(buf & np.uint32(0xFFFF))
        if aa:
            _aa_edges(buf, rgb, Wr, Hr)
        return rgb

    # ---- the density glow --------------------------------------------------------------------------------
    def _build_glow(self, proj: _Proj, k: float, Wr: int, Hr: int, buf: np.ndarray, interactive: bool):
        th = self._theme
        dlo, dhi = proj.dwin
        inv = 1.0 / max(dhi - dlo, 1e-6)
        xs, ys, ws, ts, dqs = [], [], [], [], []
        ps_t = proj.scale * k                                   # px per Angstrom at the target plane (raster)
        sigma = max(1.7 * (k if k > 1 else 1.0), 0.62 * self._dn_h * ps_t)
        if interactive:
            sigma = max(sigma, 3.0)
        f = int(min(max(round(sigma * 0.85), 2), 10))
        if interactive:
            f = max(f, 3)
        have = False
        if self._n_dn and proj.dn is not None:
            sx, sy, d, ps = proj.dn
            emit = self._dn_emit
            tv = self._dn_t
            if interactive and self._dn_sub is not None:
                sub = self._dn_sub
                sx, sy, d, emit, tv = sx[sub], sy[sub], d[sub], emit[sub], tv[sub]
            valid = (d > proj.near) & np.isfinite(sx) & np.isfinite(sy)
            ft = np.clip((d - dlo) * inv, 0.0, 1.0)
            w = emit * np.exp(-1.15 * ft)
            if interactive and self._dn_sub is not None:
                w = w * np.float32(self._n_dn / float(self._dn_sub.size))
            if not valid.all():
                sx, sy, w, tv, ft = sx[valid], sy[valid], w[valid], tv[valid], ft[valid]
            if sx.size:
                xs.append(sx * k); ys.append(sy * k); ws.append(w * self._glow_gain(f, k, proj)); ts.append(tv)
                dqs.append((ft * _DQ_SPAN + 100.0))
                have = True
        # zone / entered halos
        if self._n_sv and proj.svo is not None:
            st = self._sv_state
            hm = (st == 1) | (st == 2)
            if hm.any():
                sx, sy, d, ps = (a[hm] for a in proj.svo)
                valid = (d > proj.near) & np.isfinite(sx) & np.isfinite(sy)
                ft = np.clip((d - dlo) * inv, 0.0, 1.0)
                w = np.where(st[hm] == 2, 1.25, 0.95).astype(np.float32) * np.exp(-0.5 * ft)
                tvv = np.where(st[hm] == 2, 0.62, 0.46).astype(np.float32)
                if not valid.all():
                    sx, sy, w, tvv, ft = sx[valid], sy[valid], w[valid], tvv[valid], ft[valid]
                if sx.size:
                    xs.append(sx * k); ys.append(sy * k); ws.append(w * self._halo_gain(f, k, proj)); ts.append(tvv)
                    dqs.append((ft * _DQ_SPAN + 100.0))
                    have = True
        if not have:
            return None
        X = np.concatenate(xs) if len(xs) > 1 else xs[0]
        Y = np.concatenate(ys) if len(ys) > 1 else ys[0]
        Wt = np.concatenate(ws) if len(ws) > 1 else ws[0]
        T = np.concatenate(ts) if len(ts) > 1 else ts[0]
        DQ = np.concatenate(dqs) if len(dqs) > 1 else dqs[0]

        Wg = (Wr + f - 1) // f + 2          # padded grid: one guard cell on each side
        Hg = (Hr + f - 1) // f + 2
        # occlusion by the opaque depth buffer: block-min of the z-buffer
        gx = X / f - 0.5 + 1.0
        gy = Y / f - 0.5 + 1.0
        zb = None
        hb, wb = (Hr // f), (Wr // f)
        if hb > 0 and wb > 0:
            z16 = buf.view(np.uint16).reshape(Hr, Wr, 2)[:, :, 1]
            blk = z16[:hb * f, :wb * f].reshape(hb, f, wb, f).min(axis=(1, 3))
            if (blk != 0xFFFF).any():
                zb = blk
        if zb is not None:
            bi = np.clip((Y / f).astype(np.int32), 0, hb - 1)
            bj = np.clip((X / f).astype(np.int32), 0, wb - 1)
            zf = zb[bi, bj].astype(np.float32) * np.float32(65536.0 / 65536.0)
            # z16 holds dq of the nearest opaque sprite; margin in dq units (~2.2 Angstrom)
            span = max(dhi - dlo, 1e-6)
            margin = 2.2 / span * _DQ_SPAN
            soft = 4.5 / span * _DQ_SPAN
            over = (DQ.astype(np.float32) - zf - np.float32(margin)) / np.float32(soft)
            vis = np.where(zf >= 65535.0, 1.0, 1.0 - np.clip(over, 0.0, 1.0) * 0.88).astype(np.float32)
            Wt = Wt * vis
        inb = (gx >= 0) & (gx < Wg - 1) & (gy >= 0) & (gy < Hg - 1)
        if not inb.all():
            gx, gy, Wt, T = gx[inb], gy[inb], Wt[inb], T[inb]
        if gx.size == 0:
            return None
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        fx = (gx - x0).astype(np.float32)
        fy = (gy - y0).astype(np.float32)
        base = y0 * Wg + x0
        npx = Hg * Wg
        if interactive:
            idx = base + np.where(fx > 0.5, 1, 0) + np.where(fy > 0.5, Wg, 0)
            A = np.bincount(idx, weights=Wt, minlength=npx)
            B = np.bincount(idx, weights=Wt * T, minlength=npx)
        else:
            idx = np.concatenate((base, base + 1, base + Wg, base + Wg + 1))
            w4 = np.concatenate((Wt * (1 - fx) * (1 - fy), Wt * fx * (1 - fy), Wt * (1 - fx) * fy, Wt * fx * fy))
            A = np.bincount(idx, weights=w4, minlength=npx)
            B = np.bincount(idx, weights=w4 * np.tile(T, 4), minlength=npx)
        acc = np.empty((2, Hg, Wg), np.float32)
        acc[0] = A.reshape(Hg, Wg)
        acc[1] = B.reshape(Hg, Wg)
        acc = _blur121(acc)
        acc = _blur121(acc)
        acc *= np.float32(1.0 / 256.0)
        a = acc[0, 1:-1, 1:-1]
        b = acc[1, 1:-1, 1:-1]
        tm = np.clip(b / (a + 1e-6), 0.0, 1.0)
        ramp = th.ramp[(tm * 255.0 + 0.5).astype(np.int32)]                # (h, w, 3)
        noise = _dither_grid(a.shape[0], a.shape[1])
        if th.glow_add:
            out = 1.0 - np.exp(-1.55 * a[..., None] * ramp * 1.0)
        else:
            alpha = 1.0 - np.exp(-1.45 * a)
            out = 1.0 - alpha[..., None] * (1.0 - ramp)
        out = out * 255.0 + noise
        arr = np.empty((a.shape[0], a.shape[1], 4), np.uint8)
        arr[..., 0] = np.clip(out[..., 2], 0, 255)
        arr[..., 1] = np.clip(out[..., 1], 0, 255)
        arr[..., 2] = np.clip(out[..., 0], 0, 255)
        arr[..., 3] = 255
        arr = np.ascontiguousarray(arr)
        img = QtGui.QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 4, QtGui.QImage.Format.Format_RGB32)
        return img, arr, f

    def _glow_gain(self, f: int, k: float, proj: _Proj) -> float:
        # brightness per point so that the look is roughly independent of zoom and cloud size
        s = max(self._dn_h * proj.scale * k, 1e-3)               # lattice spacing in raster px
        cell_pts = (f / s) ** 2                                  # points per grid cell (single layer)
        return float(0.55 * min(1.0, 1.4 / max(cell_pts, 1e-3)) + 0.18)

    def _halo_gain(self, f: int, k: float, proj: _Proj) -> float:
        return float(0.9 * (f * f) / 9.0)

    # ---- scene assembly ----------------------------------------------------------------------------------
    def _build_scene(self, W: int, H: int, k: float, proj: _Proj, interactive: bool, aa: bool) -> _Scene:
        t0 = time.perf_counter()
        sc = _Scene()
        Wr = max(1, int(round(W * k)))
        Hr = max(1, int(round(H * k)))
        sc.Wr, sc.Hr, sc.k, sc.interactive = Wr, Hr, k, interactive
        cx, cy, R, w0 = self._collect_sprites(proj, k, Wr, Hr, interactive)
        buf = self._splat(cx, cy, R, w0, Wr, Hr)
        rgb = self._resolve(buf, Wr, Hr, aa and not interactive)
        sc.spr_arr = rgb
        sc.spr_img = QtGui.QImage(rgb.data, Wr, Hr, Wr * 4, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        g = self._build_glow(proj, k, Wr, Hr, buf, interactive)
        if g is not None:
            sc.glow_img, sc.glow_arr, sc.glow_f = g
            if not interactive:
                small = sc.glow_img.scaled(max(2, sc.glow_img.width() // 3), max(2, sc.glow_img.height() // 3),
                                           _Qt.AspectRatioMode.IgnoreAspectRatio, _Qt.TransformationMode.SmoothTransformation)
                sc.bloom_img = small
        sc.build_ms = (time.perf_counter() - t0) * 1000.0
        return sc

    def _compose(self, sc: _Scene, op: float) -> QtGui.QImage:
        base = self._bg_image(sc.Wr, sc.Hr)
        img = base.copy()
        p = QtGui.QPainter(img)
        p.drawImage(0, 0, sc.spr_img)
        if sc.glow_img is not None and op > 0.004:
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
            mode = (QtGui.QPainter.CompositionMode.CompositionMode_Plus if self._theme.glow_add
                    else QtGui.QPainter.CompositionMode.CompositionMode_Multiply)
            p.setCompositionMode(mode)
            f = sc.glow_f
            rect = QtCore.QRectF(-0.0, -0.0, sc.glow_img.width() * f, sc.glow_img.height() * f)
            p.setOpacity(min(1.0, op))
            p.drawImage(rect, sc.glow_img)
            if sc.bloom_img is not None:
                p.setOpacity(min(1.0, op) * (0.42 if self._theme.glow_add else 0.30))
                p.drawImage(rect, sc.bloom_img)
        p.end()
        return img

    def _scene_key(self, Wr: int, Hr: int, interactive: bool):
        return (self._cam_ver, self._data_ver, self._theme_ver, Wr, Hr, interactive, self._opts["solute_style"])

    def _ensure_final(self, W: int, H: int, dpr: float):
        interactive = bool(self._interacting)
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
            if interactive:                                    # adapt the interactive render scale
                ms = sc.build_ms
                if ms > 34.0:
                    self._rs = max(0.45, self._rs * 0.86)
                elif ms < 16.0:
                    self._rs = min(1.0, self._rs * 1.08)
        fade_q = int(round(min(1.0, max(0.0, self._dn_fade)) * 48))
        fkey = (fade_q, self._dn_opacity(), self._theme_ver)
        if sc.final is None or sc.final_key != fkey:
            sc.final = self._compose(sc, self._dn_opacity() * fade_q / 48.0)
            sc.final_key = fkey
        return sc, proj

    # =============================================================================================== painting
    def paintEvent(self, ev) -> None:  # noqa: N802
        t0 = time.perf_counter()
        p = QtGui.QPainter(self)
        try:
            W, H = max(1, self.width()), max(1, self.height())
            dpr = float(self.devicePixelRatioF())
            sc, proj = self._ensure_final(W, H, dpr)
            final = sc.final
            if abs(sc.k - dpr) < 1e-6 and final.width() == int(round(W * dpr)):
                final.setDevicePixelRatio(dpr)
                p.drawImage(0, 0, final)
            else:
                p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
                p.drawImage(QtCore.QRectF(0, 0, W, H), final)
                p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
            self._paint_overlays(p, W, H, proj, export=False)
            self._last_error = None
        except Exception:
            self._last_error = traceback.format_exc()
            print(self._last_error, file=sys.stderr)
            p.fillRect(self.rect(), self._theme.q["bg0"])
        p.end()
        now = time.perf_counter()
        dur = (now - t0) * 1000.0
        self._last_paint_ms = dur
        dt = now - self._last_paint_t if self._last_paint_t else 1.0
        inst = 1.0 / dt if dt < 0.25 else 1000.0 / max(dur, 0.5)
        self._fps = inst if self._fps <= 0 else 0.7 * self._fps + 0.3 * inst
        self._last_paint_t = now

    # ---- overlays ----------------------------------------------------------------------------------------
    def _paint_overlays(self, p: QtGui.QPainter, W: int, H: int, proj: _Proj, export: bool) -> None:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        th = self._theme
        if self._opts["show_box"] and self._box_corners is not None:
            self._draw_box(p, proj)
        self._draw_rings(p, proj)
        if not export and self._hover is not None:
            self._draw_hover(p, W, H)
        if self._opts["show_axes"]:
            self._draw_gizmo(p, W, H)
        if self._opts["show_scale_bar"]:
            self._draw_scale_bar(p, W, H, proj)
        if self._hud:
            self._draw_hud(p, W, H)
        if not export and self._hover is not None:
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
        for i in ids:
            r0 = float(_RO[st[i]] * ps[i])
            x, y = float(sx[i]), float(sy[i])
            if st[i] == 2:
                for off in (0.0, 0.5):
                    f = (ph + off) % 1.0
                    rad = r0 * (1.25 + 2.6 * f) + 1.5
                    alpha = 0.95 * (1.0 - f) ** 1.4
                    pen.setWidthF(1.7)
                    pen.setColor(_qcol(enter_col, alpha))
                    p.setPen(pen)
                    p.drawEllipse(QtCore.QPointF(x, y), rad, rad)
            else:
                rad = r0 * 1.75 + 2.0
                a = 0.62 + 0.28 * math.sin(2 * math.pi * ph)
                p.setBrush(_qcol(th.c["solute"], 0.10))
                pen.setWidthF(1.5)
                pen.setColor(_qcol(th.c["solute"], a))
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
            self.hovered.emit({k: v for k, v in hit.items() if not k.startswith("_")} if hit else {})
            self.update()
        elif hit:
            self.update()                                   # tooltip follows the cursor

    def _pick(self, x: float, y: float) -> Dict[str, Any]:
        proj = self._get_proj()
        best = None
        best_inside = None
        # candidate = (score tuple, kind, array index, sx, sy, r)
        cands: List[Tuple[float, float, str, int, float, float, float]] = []

        def scan(kind: str, layer, rad_px: np.ndarray, thr_min: float, pref: float, idx_filter=None):
            sx, sy, d, ps = layer
            dx = sx - np.float32(x)
            dy = sy - np.float32(y)
            d2 = dx * dx + dy * dy
            thr = np.maximum(rad_px, thr_min)
            near = (d2 <= thr * thr) & (d > proj.near)
            if idx_filter is not None:
                near &= idx_filter
            ids = np.flatnonzero(near)
            if ids.size == 0:
                return
            dist = np.sqrt(d2[ids])
            inside = dist <= rad_px[ids]
            for j in range(min(ids.size, 64)) if ids.size <= 64 else []:
                pass
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
        if kind == "solvent":
            orig = int(self._sv_orig[i]) if self._sv_orig is not None else i
            lab = None
            if self._sv_labels is not None and 0 <= orig < len(self._sv_labels):
                lab = str(self._sv_labels[orig])
            st = int(self._sv_state[i])
            label = (lab if lab else f"Water {orig}") + f"  ·  {_STATE_NAMES[st]}"
            pos = self._sv_o[i]
        elif kind == "solute":
            orig = int(self._sol_orig[i]) if self._sol_orig is not None else i
            lab = None
            if self._sol_labels is not None and 0 <= orig < len(self._sol_labels):
                lab = str(self._sol_labels[orig])
            role = int(self._sol_role[i])
            label = lab if lab else f"Atom {orig}" + ("  ·  zone anchor" if role == 1 else "")
            pos = self._sol_xyz[i]
        else:
            orig = int(self._dn_orig[i]) if self._dn_orig is not None else i
            label = f"Density {float(self._dn_val[i]):.3g}"
            pos = self._dn_xyz[i]
        pt = (float(pos[0]), float(pos[1]), float(pos[2]))
        ui = dict(kind=kind, sx=sx, sy=sy, r=max(r * 1.25 + 2.0, 6.0), label=label, pos=pt)
        return {"kind": kind, "index": orig, "label": label, "pos": pt, "_ui": ui}


# =================================================================================================== free helpers
def _scatter(buf: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    if _SCATTER_ORDERED:
        buf[idx] = vals
    else:                                                          # exact but slower z-min scatter
        np.minimum.at(buf, idx, vals)


def _DITHER_TILE(H: int, W: int) -> np.ndarray:
    reps_y = (H + 63) // 64
    reps_x = (W + 63) // 64
    t = np.tile(_DITHER, (reps_y, reps_x, 1))[:H, :W, :]
    return t


def _dither_grid(H: int, W: int) -> np.ndarray:
    return _DITHER_TILE(H, W)


def _aa_edges(buf: np.ndarray, rgb: np.ndarray, Wr: int, Hr: int) -> None:
    """Anti-alias the silhouette of the opaque layer (covered vs empty pixels) in place on the premultiplied rgb."""
    cov = (buf != _EMPTY).reshape(Hr, Wr)
    c8 = cov.view(np.uint8)
    if not c8.any():
        return
    p = np.zeros((Hr + 2, Wr + 2), np.uint8)
    p[1:-1, 1:-1] = c8
    nb = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    edge = np.flatnonzero(((c8 == 1) & (nb < 4)) | ((c8 == 0) & (nb > 0)))
    if edge.size == 0:
        return
    cflat = c8.reshape(-1)
    ecov = cflat[edge]
    enb = nb.reshape(-1)[edge].astype(np.float32)
    col = rgb[edge]
    outer = ecov == 0
    if outer.any():
        oi = edge[outer]
        x = oi % Wr
        cand_idx = [oi - 1, oi + 1, oi - Wr, oi + Wr]
        valid = [x > 0, x < Wr - 1, oi >= Wr, oi < (Hr - 1) * Wr]
        pick = np.zeros(oi.size, np.uint32)
        have = np.zeros(oi.size, bool)
        for ci, vm in zip(cand_idx, valid):
            ci = np.clip(ci, 0, cflat.size - 1)
            m = vm & (cflat[ci] == 1) & ~have
            if m.any():
                pick[m] = rgb[ci[m]]
                have |= m
        col = col.copy()
        col[outer] = pick
    a = np.where(ecov == 1, np.where(enb == 0, 0.80, (4.0 + enb) / 8.0), enb / 8.0).astype(np.float32)
    r = ((col >> np.uint32(16)) & np.uint32(255)).astype(np.float32)
    g = ((col >> np.uint32(8)) & np.uint32(255)).astype(np.float32)
    b = (col & np.uint32(255)).astype(np.float32)
    out = ((a * 255.0 + 0.5).astype(np.uint32) << np.uint32(24)) | ((r * a + 0.5).astype(np.uint32) << np.uint32(16)) \
        | ((g * a + 0.5).astype(np.uint32) << np.uint32(8)) | (b * a + 0.5).astype(np.uint32)
    rgb[edge] = out
