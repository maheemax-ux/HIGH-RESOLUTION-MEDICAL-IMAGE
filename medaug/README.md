# medaug

Augmentation toolkit for **high-resolution medical images**: 2D histology / fundus and 3D CT / MRI.
Pure NumPy + SciPy (no deep-learning framework required), config-driven, seeded and tested.

## Design
| Problem | What medaug does |
|---|---|
| Images too big to augment whole | `crop(size + margin) -> transforms -> center crop(size)`; margin removes warp border artifacts |
| Small lesions lost by cropping | `min_fg_frac` retries crops until the mask keeps enough foreground |
| Masks blurring | image resampled linearly, mask with nearest neighbour; both warped by the *same* field |
| CT intensities are physical | `clip_scale` windows HU instead of arbitrary rescaling |
| Leakage | `patient_split` splits by patient *before* augmentation |
| Reproducibility | one seeded RNG per pipeline |

## Install & test
```bash
pip install -e ".[dev]"
pytest
python scripts/demo.py        # writes demo_output/*.png from synthetic data
```

## Use
```python
import yaml
from medaug import build_pipeline

pipe = build_pipeline(yaml.safe_load(open("configs/ct3d.yaml")), seed=0)
sample = pipe(volume, segmentation)       # -> {"image": (64,64,64), "mask": (64,64,64)}
```
CLI:
```bash
medaug augment --config configs/histology2d.yaml --image tile.png --mask mask.npy --out out/ -n 8
medaug split --ids patient_ids.txt
```

## Transforms (`name` in config)
Spatial (image+mask): `flip`, `rot90`, `affine`, `elastic`
Intensity (image only): `gamma`, `noise`, `rician` (MRI), `blur`, `lowres`, `bias` (MRI bias field), `clip_scale` (CT window), `hed` (H&E stain jitter, uint8 RGB)

Configs: `configs/ct3d.yaml`, `configs/mri3d.yaml`, `configs/histology2d.yaml`.

## PyTorch integration
```python
class DS(torch.utils.data.Dataset):
    def __init__(self, items, pipe): self.items, self.pipe = items, pipe
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img, msk = load(self.items[i]); s = self.pipe(img, msk)
        return torch.from_numpy(s["image"]).float()[None], torch.from_numpy(s["mask"]).long()
```
With multiple DataLoader workers, build one pipeline per worker with a different seed (`worker_init_fn`).

## Notes / limits
- Patch-origin shape inference: a trailing dim of 3-4 on a 3-D array is treated as colour channels; pass a `mask` to make dimensionality explicit.
- Do not flip left/right for lateralized anatomy; remove `flip` axes accordingly.
- Never augment val/test sets; apply only deterministic preprocessing (e.g. `clip_scale`).
- Extending: subclass `Transform`, implement `apply(sample, rng)`, add to `REGISTRY` in `pipeline.py`.
- Next steps: generative augmentation (latent diffusion) behind the same `Compose` interface; MONAI/TorchIO backends for GPU speed.
