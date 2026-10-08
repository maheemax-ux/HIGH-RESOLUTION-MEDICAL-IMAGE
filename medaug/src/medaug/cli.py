"""CLI:
  medaug augment --config configs/ct3d.yaml --image vol.npy [--mask seg.npy] --out out/ -n 8
  medaug split --ids ids.txt
Images: .npy (any dim) or .png/.jpg (2D). Output: .npy (image_i / mask_i) and, for 2D, a PNG preview.
"""
import argparse
import os
import numpy as np
import yaml
from .pipeline import build_pipeline
from .split import patient_split


def _load(path):
    if path.endswith(".npy"):
        return np.load(path)
    from PIL import Image
    return np.array(Image.open(path))


def cmd_augment(a):
    cfg = yaml.safe_load(open(a.config))
    pipe = build_pipeline(cfg, seed=a.seed)
    img = _load(a.image)
    mask = _load(a.mask) if a.mask else None
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.n):
        s = pipe(img, mask)
        np.save(os.path.join(a.out, f"image_{i}.npy"), s["image"])
        if "mask" in s:
            np.save(os.path.join(a.out, f"mask_{i}.npy"), s["mask"])
        if s["image"].dtype == np.uint8 and s["image"].ndim == 3:
            from PIL import Image
            Image.fromarray(s["image"]).save(os.path.join(a.out, f"image_{i}.png"))
    print(f"wrote {a.n} augmented samples to {a.out}")


def cmd_split(a):
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    for k, v in patient_split(ids, a.val, a.test, a.seed).items():
        print(f"{k}: {sorted(v)}")


def main():
    p = argparse.ArgumentParser(prog="medaug")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("augment")
    s.add_argument("--config", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--mask")
    s.add_argument("--out", required=True)
    s.add_argument("-n", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_augment)
    t = sub.add_parser("split")
    t.add_argument("--ids", required=True, help="text file, one patient id per line")
    t.add_argument("--val", type=float, default=0.15)
    t.add_argument("--test", type=float, default=0.15)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_split)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
