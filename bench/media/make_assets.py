"""Derive the README assets from the rendered demo video: a short looping GIF, a poster frame and the GitHub social-preview image.

    python bench/media/make_assets.py            # reads docs/media/sozlab-demo.mp4 and docs/media/timeline.json

Requires ffmpeg on PATH and Pillow.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def run(cmd):
    subprocess.run(cmd, check=True)


def font(size: int, bold=True):
    import matplotlib

    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return ImageFont.truetype(str(base / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def mono(size: int):
    import matplotlib

    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono-Bold.ttf"), size)


def hero_gif(mp4: Path, out: Path, marks: dict, width: int = 880, fps: int = 12) -> None:
    """The best feature in ~13 s: drag the distance line (zone follows in 3D and over time), then watch solvent exchange."""
    a0, a1 = marks["drag"] - 1.2, marks["drag_end"]
    b0, b1 = marks["watch"] + 2.2, marks["watch"] + 2.2 + 5.0
    graph = (f"[0:v]trim=start={a0}:end={a1},setpts=PTS-STARTPTS[a];[0:v]trim=start={b0}:end={b1},setpts=PTS-STARTPTS[b];"
             f"[a][b]concat=n=2:v=1[v];[v]fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
             f"[s0]palettegen=max_colors=160:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-filter_complex", graph, "-loop", "0", str(out)])


def poster(mp4: Path, out: Path, at: float) -> None:
    tmp = out.with_suffix(".tmp.png")
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at}", "-i", str(mp4), "-frames:v", "1", str(tmp)])
    Image.open(tmp).convert("RGB").quantize(colors=224, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, optimize=True)
    tmp.unlink()


def social_preview(frame_png: Path, out: Path) -> None:
    """1280x640, the size GitHub asks for: title and promise on the left, the real application on the right."""
    W, H = 1280, 640
    bg = Image.new("RGB", (W, H), (8, 15, 29))
    px = bg.load()
    for y in range(H):  # vertical gradient
        for x in range(0, W):
            t = (x / W * 0.45 + y / H * 0.55)
            px[x, y] = (int(8 + 6 * t), int(15 + 18 * t), int(29 + 24 * t))
    draw = ImageDraw.Draw(bg, "RGBA")
    for cx, cy, r in ((140, 560, 90), (610, 70, 60), (60, 120, 40)):  # faint "solvent" discs
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(53, 224, 200, 26))
    shot = Image.open(frame_png).convert("RGB")
    sw = 760
    sh = int(shot.height * sw / shot.width)
    shot = shot.resize((sw, sh), Image.LANCZOS)
    mask = Image.new("L", shot.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, shot.width, shot.height), radius=18, fill=255)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((470, 118, 470 + sw, 118 + sh), radius=18, fill=(0, 0, 0, 170))
    bg.paste(shadow.filter(ImageFilter.GaussianBlur(18)), (0, 12), shadow.filter(ImageFilter.GaussianBlur(18)))
    bg.paste(shot, (470, 108), mask)
    draw = ImageDraw.Draw(bg)
    draw.text((56, 118), "SOZLab", font=font(76), fill=(233, 241, 251))
    draw.text((58, 214), "See the solvent around", font=font(30, False), fill=(168, 185, 209))
    draw.text((58, 254), "your molecule.", font=font(30, False), fill=(168, 185, 209))
    draw.text((58, 330), "Occupancy, residence time and density", font=font(19, False), fill=(116, 137, 165))
    draw.text((58, 358), "with the uncertainty attached.", font=font(19, False), fill=(116, 137, 165))
    draw.rounded_rectangle((56, 440, 384, 482), radius=10, outline=(53, 224, 200), width=2)
    draw.text((72, 450), "sozlab-gui --demo", font=mono(20), fill=(53, 224, 200))
    draw.text((58, 560), "Workbench GUI  ·  CLI  ·  MIT", font=font(17, False), fill=(116, 137, 165))
    bg.save(out, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "docs" / "media"))
    args = ap.parse_args()
    d = Path(args.dir)
    mp4, tl = d / "sozlab-demo.mp4", json.load(open(d / "timeline.json"))
    marks = tl["marks"]
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required")
    hero_gif(mp4, d / "sozlab-hero.gif", marks)
    poster(mp4, d / "poster.png", marks["drag"] + 5.5)
    tmp = d / "_frame.png"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{marks['drag'] + 5.5}", "-i", str(mp4), "-frames:v", "1", str(tmp)])
    social_preview(tmp, d / "social-preview.png")
    tmp.unlink()
    for name in ("sozlab-demo.mp4", "sozlab-hero.gif", "poster.png", "social-preview.png"):
        print(f"{name}: {(d / name).stat().st_size / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
