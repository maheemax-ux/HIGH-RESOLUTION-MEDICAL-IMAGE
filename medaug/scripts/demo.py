"""Generate synthetic data, augment, save a preview grid (no real data needed)."""
import os, sys
import numpy as np, yaml
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from medaug import build_pipeline

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "demo_output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

# --- synthetic H&E-like 2D image with circular "nuclei" and masks ---
H = W = 1024
img = np.full((H, W, 3), (235, 200, 220), np.uint8)
mask = np.zeros((H, W), np.uint8)
yy, xx = np.mgrid[:H, :W]
for _ in range(120):
    cy, cx, r = rng.integers(40, H - 40), rng.integers(40, W - 40), rng.integers(8, 18)
    m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    img[m] = (90, 60, 140)
    mask[m] = 1
img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)

pipe = build_pipeline(yaml.safe_load(open(os.path.join(ROOT, "configs/histology2d.yaml"))), seed=1)
tiles = [pipe(img, mask) for _ in range(4)]
row_img = np.concatenate([t["image"] for t in tiles], axis=1)
row_msk = np.concatenate([np.stack([t["mask"] * 255] * 3, -1) for t in tiles], axis=1)
Image.fromarray(np.concatenate([row_img, row_msk], 0)).save(os.path.join(OUT, "histology_preview.png"))

# --- synthetic 3D CT-like volume with a spherical lesion ---
vol = rng.normal(40, 15, (128, 128, 128)).astype(np.float32)
z, y, x = np.mgrid[:128, :128, :128]
lesion = ((z - 64) ** 2 + (y - 70) ** 2 + (x - 60) ** 2) < 14 ** 2
vol[lesion] += 120
seg = lesion.astype(np.uint8)
pipe3 = build_pipeline(yaml.safe_load(open(os.path.join(ROOT, "configs/ct3d.yaml"))), seed=2)
cols = []
for _ in range(4):
    s = pipe3(vol, seg)
    c = s["image"].shape[0] // 2
    sl = (s["image"][c] * 255).clip(0, 255).astype(np.uint8)
    ov = np.stack([sl] * 3, -1)
    ov[s["mask"][c] > 0, 0] = 255
    cols.append(np.kron(ov, np.ones((4, 4, 1), np.uint8)))
Image.fromarray(np.concatenate(cols, 1)).save(os.path.join(OUT, "ct_preview.png"))
print("saved previews to", OUT)
