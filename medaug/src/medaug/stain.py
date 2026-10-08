"""H&E stain augmentation via HED colour-space jitter (Tellez et al. style).
Input: uint8 RGB (H,W,3).
"""
import numpy as np
from .transforms import Transform

# Ruifrok & Johnston H&E-DAB stain vectors (rows: H, E, DAB)
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]])
_RGB_FROM_HED /= np.linalg.norm(_RGB_FROM_HED, axis=1, keepdims=True)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)


def rgb_to_hed(rgb_u8):
    od = -np.log((rgb_u8.astype(np.float64) + 1) / 256.0)
    return od @ _HED_FROM_RGB


def hed_to_rgb(hed):
    od = hed @ _RGB_FROM_HED
    rgb = 256.0 * np.exp(-od) - 1
    return np.clip(rgb, 0, 255).astype(np.uint8)


class RandHEDJitter(Transform):
    def __init__(self, sigma=0.05, bias=0.05, p=0.7):
        super().__init__(p)
        self.sigma, self.bias = sigma, bias

    def apply(self, s, rng):
        img = s["image"]
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[-1] == 3, "expects uint8 RGB"
        hed = rgb_to_hed(img)
        alpha = rng.uniform(1 - self.sigma, 1 + self.sigma, 3)
        beta = rng.uniform(-self.bias, self.bias, 3)
        out = dict(s)
        out["image"] = hed_to_rgb(hed * alpha + beta)
        return out
