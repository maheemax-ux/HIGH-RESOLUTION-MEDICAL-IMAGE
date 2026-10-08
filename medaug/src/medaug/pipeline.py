"""Compose transforms and build them from a YAML/dict config."""
import numpy as np
from . import transforms as T
from .stain import RandHEDJitter
from .patches import sample_patch, center_crop

REGISTRY = {
    "flip": T.RandFlip, "rot90": T.RandRot90, "affine": T.RandAffine,
    "elastic": T.RandElastic, "gamma": T.RandGamma, "noise": T.RandGaussianNoise,
    "rician": T.RandRicianNoise, "blur": T.RandGaussianBlur, "lowres": T.RandLowRes,
    "bias": T.RandBiasField, "clip_scale": T.ClipScale, "hed": RandHEDJitter,
}


class Compose:
    """crop(size+margin) -> transforms -> center crop(size). Deterministic given seed."""

    def __init__(self, transforms, patch_size=None, margin=0, min_fg_frac=0.0, seed=None):
        self.transforms = transforms
        self.patch_size, self.margin, self.min_fg_frac = patch_size, margin, min_fg_frac
        self.rng = np.random.default_rng(seed)

    def __call__(self, image, mask=None):
        if self.patch_size:
            image, mask, _ = sample_patch(image, mask, self.patch_size, self.margin,
                                          self.min_fg_frac, self.rng)
        sample = {"image": image}
        if mask is not None:
            sample["mask"] = mask
        for t in self.transforms:
            sample = t(sample, self.rng)
        if self.patch_size:
            nd = sample["mask"].ndim if "mask" in sample else (
                sample["image"].ndim - (1 if sample["image"].shape[-1] in (3, 4) and sample["image"].ndim == 3 else 0))
            sample["image"] = center_crop(sample["image"], self.patch_size, nd)
            if "mask" in sample:
                sample["mask"] = center_crop(sample["mask"], self.patch_size, nd)
        return sample


def build_pipeline(cfg, seed=None):
    steps = []
    for item in cfg["transforms"]:
        item = dict(item)
        name = item.pop("name")
        steps.append(REGISTRY[name](**item))
    c = cfg.get("crop", {})
    return Compose(steps, patch_size=c.get("size"), margin=c.get("margin", 0),
                   min_fg_frac=c.get("min_fg_frac", 0.0), seed=seed)
