"""
FILE 2 of 3: run_fid.py  (LOW-COMPUTE version)
Computes FID scores and saves them as CSV files in results/.

  python run_fid.py          -> runs everything (core + applications)
  python run_fid.py core     -> only the 4 basic experiments
  python run_fid.py apps     -> only the Medical / War / Flood / Detection examples

Why it is fast:
  Each image folder goes through Inception ONLY ONCE. The 2,048 numbers per image
  (called features) are kept in memory, and every FID score reuses them.
  The Inception model comes from clean-fid. The FID formula is the standard one.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from cleanfid import fid

DATA = "data"
RESULTS = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS, exist_ok=True)

MODELS = {}   # one Inception model per resize mode
FEATS = {}    # saved features per (folder, mode)


def model_for(mode):
    if mode not in MODELS:
        MODELS[mode] = fid.build_feature_extractor(mode, DEVICE, use_dataparallel=False)
    return MODELS[mode]


def feats(name, mode="clean"):
    """Inception features of a folder. Computed once, then reused."""
    key = (name, mode)
    if key not in FEATS:
        t = time.time()
        f = fid.get_folder_features(os.path.join(DATA, name), model=model_for(mode),
                                    num_workers=0, batch_size=32, device=DEVICE,
                                    mode=mode, verbose=False)
        FEATS[key] = f.astype(np.float64)
        print(f"  features  {name:<22} {len(f):>4} images  {time.time() - t:6.1f}s", flush=True)
    return FEATS[key]


def frechet(f1, f2):
    """FID between two feature sets (same result as the usual formula, but fast and safe
    when there are fewer images than 2,048)."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    a = (f1 - mu1) / np.sqrt(len(f1) - 1)
    b = (f2 - mu2) / np.sqrt(len(f2) - 1)
    trace_part = (a ** 2).sum() + (b ** 2).sum() - 2 * np.linalg.svd(a @ b.T, compute_uv=False).sum()
    return float(((mu1 - mu2) ** 2).sum() + trace_part)


def score(name_a, name_b, mode="clean", n=None):
    f1, f2 = feats(name_a, mode), feats(name_b, mode)
    if n:
        f1, f2 = f1[:n], f2[:n]
    v = frechet(f1, f2)
    print(f"  FID {v:9.2f}   {name_a}  vs  {name_b}   (mode={mode})", flush=True)
    return v


def save(rows, name):
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, name), index=False)


# ======================================================================
def run_core():
    print("\n[1] Floor check")
    floor = score("real_a", "real_b")

    print("\n[2] Damage ladder")
    rows = [{"damage": d, "level": 0, "fid": floor} for d in ["blur", "noise", "jpeg"]]
    for d in ["blur", "noise", "jpeg"]:
        for lvl in [1, 2, 3, 4]:
            rows.append({"damage": d, "level": lvl, "fid": score("real_a", f"{d}_{lvl}")})
    save(rows, "damage_ladder.csv")

    print("\n[3] Mode collapse")
    rows = [
        {"set": "All 5 shapes (real_b)", "shapes_kept": 5, "fid": floor},
        {"set": "3 shapes only", "shapes_kept": 3, "fid": score("real_a", "collapse_3class")},
        {"set": "2 shapes only", "shapes_kept": 2, "fid": score("real_a", "collapse_2class")},
        {"set": "1 shape only", "shapes_kept": 1, "fid": score("real_a", "collapse_1class")},
    ]
    save(rows, "mode_collapse.csv")

    print("\n[4] Resize modes")
    rows = []
    for mode in ["clean", "legacy_pytorch"]:
        for label, other in [("real_a vs real_b", "real_b"),
                             ("real_a vs blur_2", "blur_2"),
                             ("real_a vs noise_2", "noise_2")]:
            rows.append({"mode": mode, "pair": label, "fid": score("real_a", other, mode)})
    save(rows, "resize_mode.csv")

    print("\n[5] Sample size (no extra work: uses the first n saved features)")
    full = len(feats("real_a"))
    sizes = sorted({max(20, full // 4), max(30, full // 2), max(40, 3 * full // 4), full})
    rows = []
    for n in sizes:
        rows.append({"n_images": n, "pair": "real vs real", "fid": score("real_a", "real_b", n=n)})
        rows.append({"n_images": n, "pair": "real vs blur_2", "fid": score("real_a", "blur_2", n=n)})
    save(rows, "sample_size.csv")


# ======================================================================
def run_apps():
    if not os.path.exists(os.path.join(DATA, "med_pooled")):
        print("Application folders not found. Run: python make_data.py")
        return

    print("\n[6] Medical")
    rows = []
    for label, name in [("Hospital A", "med_hosp_a"), ("Hospital B", "med_hosp_b"), ("Hospital C", "med_hosp_c")]:
        rows.append({"group": "hospital", "label": label, "folder": name, "fid": score("med_pooled", name)})
    for label, name in [("Synthetic (good)", "med_synth_good"),
                        ("Synthetic (no lesions)", "med_synth_nolesion"),
                        ("Memorized copy", "med_memorized")]:
        rows.append({"group": "synthetic", "label": label, "folder": name, "fid": score("med_pooled", name)})
    save(rows, "apps_medical.csv")

    print("\n[7] War / security")
    rows = []
    for label, name in [("Synthetic: all damage types", "war_synth_diverse"),
                        ("Synthetic: one damage type only", "war_synth_narrow"),
                        ("Synthetic: blurry", "war_synth_blurry")]:
        rows.append({"label": label, "folder": name, "fid": score("war_real", name)})
    save(rows, "apps_war.csv")

    print("\n[8] Flood / disaster (weekly drift)")
    rows = [{"week": k, "fid": score("flood_base", f"flood_week_{k}")} for k in range(6)]
    save(rows, "apps_flood.csv")

    print("\n[9] Detection and comparison")
    if not os.path.exists(os.path.join(DATA, "real_a")):
        print("Core folders missing. Run: python make_data.py")
        return
    cands = [
        ("Real set B (reference-like)", "real_b"),
        ("Blurry", "blur_2"),
        ("Noisy (medium)", "noise_2"),
        ("Noisy (heavy)", "noise_3"),
        ("Compressed (JPEG)", "jpeg_2"),
        ("Only 2 shapes", "collapse_2class"),
    ]
    rows = []
    for label, name in cands:
        rows.append({"label": label, "folder": name,
                     "fid_to_real": score("real_a", name),
                     "fid_to_trained_generator": score("noise_1", name)})   # noise_1 = generator the detector learned
    save(rows, "apps_detection.csv")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    start = time.time()
    if which in ("all", "core"):
        run_core()
    if which in ("all", "apps"):
        run_apps()
    print(f"\nDone in {(time.time() - start) / 60:.1f} minutes. Results saved in the 'results' folder. "
          "Next: streamlit run app.py")
