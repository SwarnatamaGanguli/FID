"""
FILE 1 of 3: make_data.py
Creates DUMMY images. No download needed.
  - Core sets: coloured shapes, damaged copies, fake mode collapse
  - Application sets: medical scans, war/security tiles, flood tiles

Install once:   pip install clean-fid pillow numpy pandas streamlit torch
Run:            python make_data.py
"""
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 64            # image size in pixels
N = 120             # images per core folder (small on purpose: fast on a laptop CPU)
N_APP = 80           # images per application folder
OUT = "data"         # image folders go here
SAMPLES = "samples"  # small picture grids for the app

ALL_SHAPES = ["circle", "square", "triangle", "ring", "stripes"]


# ---------------------------------------------------------------- helpers
def fresh_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def build(name, fn, seed, n):
    """Make n images. fn(rng, i) must return one PIL image."""
    folder = os.path.join(OUT, name)
    fresh_folder(folder)
    rng = np.random.default_rng(seed)
    for i in range(n):
        fn(rng, i).save(os.path.join(folder, f"{i:04d}.png"))
    print(f"made {folder}  ({n} images)")


def damage_folder(src_name, new_name, fn, ext="png", **save_kwargs):
    """Copy every image from the src folder, damaged by fn."""
    src = os.path.join(OUT, src_name)
    dst = os.path.join(OUT, new_name)
    fresh_folder(dst)
    rng = np.random.default_rng(123)
    for f in sorted(os.listdir(src)):
        img = Image.open(os.path.join(src, f)).convert("RGB")
        img = fn(img, rng)
        img.save(os.path.join(dst, f.replace(".png", f".{ext}")), **save_kwargs)
    print(f"made {dst}")


def make_grid(folder_name, k=16):
    """Save a 4x4 picture grid so the app can show examples."""
    os.makedirs(SAMPLES, exist_ok=True)
    folder = os.path.join(OUT, folder_name)
    files = sorted(os.listdir(folder))[:k]
    scale = 2
    cell = SIZE * scale
    grid = Image.new("RGB", (cell * 4, cell * 4))
    for i, f in enumerate(files):
        img = Image.open(os.path.join(folder, f)).convert("RGB").resize((cell, cell))
        grid.paste(img, ((i % 4) * cell, (i // 4) * cell))
    grid.save(os.path.join(SAMPLES, f"{folder_name}.png"))


def blur(radius):
    return lambda img, rng: img.filter(ImageFilter.GaussianBlur(radius))


def noise(sigma):
    def fn(img, rng):
        arr = np.asarray(img).astype(np.float32)
        arr = arr + rng.normal(0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return fn


# ---------------------------------------------------------------- core: shapes
def make_shape(shape, rng):
    bg = tuple(int(v) for v in rng.integers(0, 90, 3))
    fg = tuple(int(v) for v in rng.integers(140, 256, 3))
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)
    cx, cy = (int(v) for v in rng.integers(22, 42, 2))
    r = int(rng.integers(10, 20))
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fg)
    elif shape == "ring":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg, width=4)
    else:  # stripes
        step = int(rng.integers(8, 14))
        vertical = bool(rng.integers(0, 2))
        for p in range(0, SIZE, step):
            box = [p, 0, p + step // 2, SIZE] if vertical else [0, p, SIZE, p + step // 2]
            d.rectangle(box, fill=fg)
    return img


def shapes_fn(shapes):
    return lambda rng, i: make_shape(shapes[i % len(shapes)], rng)


def build_core():
    build("real_a", shapes_fn(ALL_SHAPES), 1, N)
    build("real_b", shapes_fn(ALL_SHAPES), 2, N)
    build("collapse_1class", shapes_fn(["circle"]), 3, N)
    build("collapse_2class", shapes_fn(["circle", "square"]), 4, N)
    build("collapse_3class", shapes_fn(["circle", "square", "triangle"]), 5, N)

    for lvl, radius in enumerate([1, 2, 3, 4], start=1):
        damage_folder("real_b", f"blur_{lvl}", blur(radius))
    for lvl, sigma in enumerate([10, 25, 50, 80], start=1):
        damage_folder("real_b", f"noise_{lvl}", noise(sigma))
    for lvl, q in enumerate([60, 30, 15, 5], start=1):
        damage_folder("real_b", f"jpeg_{lvl}", lambda img, rng: img, ext="jpg", quality=q)


# ---------------------------------------------------------------- medical
HOSPITALS = {
    "a": dict(bright=0, contrast=1.0, noise=4, tint=(0, 0, 0)),
    "b": dict(bright=40, contrast=1.3, noise=14, tint=(0, 0, 0)),
    "c": dict(bright=-25, contrast=0.7, noise=6, tint=(0, 18, 10)),
}


def make_scan(rng, diseased, style):
    """Dummy grey 'scan': body, organ, and (if diseased) a bright lesion spot."""
    base = Image.new("L", (SIZE, SIZE), 20)
    d = ImageDraw.Draw(base)
    d.ellipse([8, 6, 56, 58], fill=110)
    d.ellipse([20, 18, 44, 46], fill=70)
    if diseased:
        x, y = (int(v) for v in rng.integers(22, 42, 2))
        r = int(rng.integers(3, 7))
        d.ellipse([x - r, y - r, x + r, y + r], fill=210)
    arr = np.asarray(base).astype(np.float32)
    arr = (arr - 100) * style["contrast"] + 100 + style["bright"]
    arr = arr + rng.normal(0, style["noise"], arr.shape)
    rgb = np.stack([arr] * 3, axis=-1) + np.array(style["tint"], dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def build_medical():
    keys = list(HOSPITALS)
    build("med_pooled", lambda rng, i: make_scan(rng, i % 2 == 0, HOSPITALS[keys[i % 3]]), 10, N_APP)
    for k, seed in zip(keys, [11, 12, 13]):
        build(f"med_hosp_{k}", lambda rng, i, k=k: make_scan(rng, i % 2 == 0, HOSPITALS[k]), seed, N_APP)
    # a "good" synthetic set: same mix as pooled, slightly soft
    build("med_synth_good",
          lambda rng, i: make_scan(rng, i % 2 == 0, HOSPITALS[keys[i % 3]]).filter(ImageFilter.GaussianBlur(0.7)),
          14, N_APP)
    # looks like clean scans but no disease spots at all
    build("med_synth_nolesion", lambda rng, i: make_scan(rng, False, HOSPITALS[keys[i % 3]]), 15, N_APP)
    # a generator that only copies its training images
    dst = os.path.join(OUT, "med_memorized")
    fresh_folder(dst)
    for f in sorted(os.listdir(os.path.join(OUT, "med_pooled"))):
        shutil.copy(os.path.join(OUT, "med_pooled", f), dst)
    print(f"made {dst}  (exact copies)")


# ---------------------------------------------------------------- war / security
def make_war_tile(rng, damage):
    """Dummy aerial tile: 4x4 buildings. damage: None, fragments, scorch, collapse."""
    ground = (int(rng.integers(80, 110)), int(rng.integers(70, 95)), int(rng.integers(50, 70)))
    img = Image.new("RGB", (SIZE, SIZE), ground)
    d = ImageDraw.Draw(img)
    for gx in range(4):
        for gy in range(4):
            x0, y0 = gx * 16 + 3, gy * 16 + 3
            x1, y1 = x0 + 10, y0 + 10
            if damage is None or rng.random() < 0.3:
                d.rectangle([x0, y0, x1, y1], fill=(190, 190, 185))
                d.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=(140, 140, 135))
            elif damage == "fragments":
                for _ in range(5):
                    fx, fy = x0 + int(rng.integers(0, 9)), y0 + int(rng.integers(0, 9))
                    d.rectangle([fx, fy, fx + 2, fy + 2], fill=(150, 145, 140))
            elif damage == "scorch":
                d.ellipse([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=(30, 28, 26))
            else:  # collapse
                d.polygon([(x0, y1), (x1, y1), (x1, y0 + 6), (x0 + 4, y1 - 3)], fill=(120, 115, 110))
    return img


WAR_KINDS = [None, "fragments", "scorch", "collapse"]


def build_war():
    build("war_real", lambda rng, i: make_war_tile(rng, WAR_KINDS[i % 4]), 21, N_APP)
    build("war_synth_diverse", lambda rng, i: make_war_tile(rng, WAR_KINDS[i % 4]), 22, N_APP)
    build("war_synth_narrow", lambda rng, i: make_war_tile(rng, "fragments"), 23, N_APP)
    build("war_synth_blurry",
          lambda rng, i: make_war_tile(rng, WAR_KINDS[i % 4]).filter(ImageFilter.GaussianBlur(1.5)),
          24, N_APP)


# ---------------------------------------------------------------- flood / disaster
FLOOD_WEEKS = [0.05, 0.12, 0.22, 0.35, 0.5, 0.65]   # water share per week


def make_flood_tile(rng, water):
    """Dummy satellite tile: green land, blue water covering about `water` of the area."""
    land = (int(rng.integers(70, 110)), int(rng.integers(110, 150)), int(rng.integers(60, 90)))
    img = Image.new("RGB", (SIZE, SIZE), land)
    mask = Image.new("L", (SIZE, SIZE), 0)
    md = ImageDraw.Draw(mask)
    tries = 0
    while np.asarray(mask).mean() / 255 < water and tries < 200:
        x, y = (int(v) for v in rng.integers(0, SIZE, 2))
        rx, ry = (int(v) for v in rng.integers(6, 16, 2))
        md.ellipse([x - rx, y - ry, x + rx, y + ry], fill=255)
        tries += 1
    blue = Image.new("RGB", (SIZE, SIZE),
                     (int(rng.integers(30, 60)), int(rng.integers(80, 110)), int(rng.integers(150, 200))))
    img.paste(blue, (0, 0), mask)
    d = ImageDraw.Draw(img)
    for _ in range(3):
        bx, by = (int(v) for v in rng.integers(2, SIZE - 8, 2))
        d.rectangle([bx, by, bx + 5, by + 5], fill=(170, 170, 165))
    return img


def build_flood():
    build("flood_base", lambda rng, i: make_flood_tile(rng, FLOOD_WEEKS[0]), 31, N_APP)
    for k, water in enumerate(FLOOD_WEEKS):
        build(f"flood_week_{k}", lambda rng, i, w=water: make_flood_tile(rng, w), 40 + k, N_APP)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    build_core()
    build_medical()
    build_war()
    build_flood()
    for name in sorted(os.listdir(OUT)):
        make_grid(name)
    print("Done. Next: python run_fid.py")


if __name__ == "__main__":
    main()
