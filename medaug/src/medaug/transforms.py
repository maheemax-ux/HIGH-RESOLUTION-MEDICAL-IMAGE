"""Transforms for 2D or 3D arrays. All take/return a sample dict:
    {"image": ndarray, "mask": ndarray (optional)}
`image` is (*spatial) or (*spatial, C) for color 2D; `mask` is (*spatial).
Spatial transforms hit image and mask identically (linear for image, nearest for mask).
Intensity transforms touch the image only.
"""
import numpy as np
from scipy import ndimage as ndi


class Transform:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, sample, rng):
        if rng.random() >= self.p:
            return sample
        return self.apply(sample, rng)

    def apply(self, sample, rng):  # pragma: no cover
        raise NotImplementedError


def _nd(sample):
    m = sample.get("mask")
    if m is not None:
        return m.ndim
    img = sample["image"]
    return img.ndim - 1 if (img.ndim == 3 and img.shape[-1] in (3, 4)) else img.ndim


def _warp(sample, coords_fn, mode="reflect"):
    """Resample image & mask with coords_fn(shape)->coordinate array (nd, *shape)."""
    nd = _nd(sample)
    img = sample["image"]
    shape = img.shape[:nd]
    coords = coords_fn(shape)
    out = dict(sample)
    if img.ndim == nd:
        out["image"] = ndi.map_coordinates(img, coords, order=1, mode=mode).astype(img.dtype)
    else:
        out["image"] = np.stack(
            [ndi.map_coordinates(img[..., c], coords, order=1, mode=mode) for c in range(img.shape[-1])],
            axis=-1).astype(img.dtype)
    if sample.get("mask") is not None:
        out["mask"] = ndi.map_coordinates(sample["mask"], coords, order=0, mode=mode).astype(sample["mask"].dtype)
    return out


class RandFlip(Transform):
    def __init__(self, axes=(0, 1), p=0.5):
        super().__init__(p)
        self.axes = axes

    def apply(self, s, rng):
        out = dict(s)
        for ax in self.axes:
            if rng.random() < 0.5:
                out["image"] = np.flip(out["image"], ax)
                if out.get("mask") is not None:
                    out["mask"] = np.flip(out["mask"], ax)
        out["image"] = np.ascontiguousarray(out["image"])
        if out.get("mask") is not None:
            out["mask"] = np.ascontiguousarray(out["mask"])
        return out


class RandRot90(Transform):
    def __init__(self, axes=(0, 1), p=0.5):
        super().__init__(p)
        self.axes = axes

    def apply(self, s, rng):
        k = int(rng.integers(1, 4))
        out = dict(s)
        out["image"] = np.ascontiguousarray(np.rot90(s["image"], k, self.axes))
        if s.get("mask") is not None:
            out["mask"] = np.ascontiguousarray(np.rot90(s["mask"], k, self.axes))
        return out


class RandAffine(Transform):
    """Small rotation (deg) / isotropic scale about the center. 2D: one angle; 3D: three."""

    def __init__(self, rotate_deg=10, scale=(0.9, 1.1), p=0.5):
        super().__init__(p)
        self.rotate, self.scale = rotate_deg, scale

    def apply(self, s, rng):
        nd = _nd(s)
        ang = np.deg2rad(rng.uniform(-self.rotate, self.rotate, size=1 if nd == 2 else 3))
        sc = rng.uniform(*self.scale)
        if nd == 2:
            c, si = np.cos(ang[0]), np.sin(ang[0])
            R = np.array([[c, -si], [si, c]])
        else:
            def rx(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            def ry(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            def rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            R = rx(ang[0]) @ ry(ang[1]) @ rz(ang[2])
        M = R / sc

        def coords(shape):
            center = (np.array(shape) - 1) / 2.0
            grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")).astype(np.float32)
            g = grid.reshape(nd, -1) - center[:, None]
            return ((M @ g) + center[:, None]).reshape(nd, *shape).astype(np.float32)

        return _warp(s, coords)


class RandElastic(Transform):
    """Smooth random displacement field. `sigma` = smoothness (px), `alpha` = max magnitude (px)."""

    def __init__(self, sigma=10, alpha=8, p=0.3):
        super().__init__(p)
        self.sigma, self.alpha = sigma, alpha

    def apply(self, s, rng):
        def coords(shape):
            grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")).astype(np.float32)
            disp = []
            for _ in shape:
                d = ndi.gaussian_filter(rng.uniform(-1, 1, shape).astype(np.float32), self.sigma, mode="reflect")
                d = d / (np.abs(d).max() + 1e-8) * self.alpha
                disp.append(d)
            return grid + np.stack(disp)

        return _warp(s, coords)


# ---------------- intensity ----------------

class RandGamma(Transform):
    """Gamma on image min-max normalized to [0,1], restored to original range."""

    def __init__(self, gamma=(0.7, 1.5), p=0.3):
        super().__init__(p)
        self.gamma = gamma

    def apply(self, s, rng):
        x = s["image"].astype(np.float32)
        lo, hi = x.min(), x.max()
        g = np.exp(rng.uniform(np.log(self.gamma[0]), np.log(self.gamma[1])))
        out = dict(s)
        out["image"] = (((x - lo) / (hi - lo + 1e-8)) ** g * (hi - lo) + lo).astype(s["image"].dtype)
        return out


class RandGaussianNoise(Transform):
    def __init__(self, std=0.01, p=0.2):
        super().__init__(p)
        self.std = std

    def apply(self, s, rng):
        x = s["image"].astype(np.float32)
        scale = (x.max() - x.min()) or 1.0
        std = rng.uniform(0, self.std) * scale
        out = dict(s)
        out["image"] = (x + rng.normal(0, std, x.shape)).astype(s["image"].dtype)
        return out


class RandRicianNoise(Transform):
    """Magnitude of complex Gaussian noise: realistic for MRI."""

    def __init__(self, std=0.02, p=0.2):
        super().__init__(p)
        self.std = std

    def apply(self, s, rng):
        x = s["image"].astype(np.float32)
        scale = (x.max() - x.min()) or 1.0
        std = rng.uniform(0, self.std) * scale
        out = dict(s)
        out["image"] = np.sqrt((x + rng.normal(0, std, x.shape)) ** 2 + rng.normal(0, std, x.shape) ** 2).astype(s["image"].dtype)
        return out


class RandGaussianBlur(Transform):
    def __init__(self, sigma=(0.3, 1.5), p=0.2):
        super().__init__(p)
        self.sigma = sigma

    def apply(self, s, rng):
        nd = _nd(s)
        sig = rng.uniform(*self.sigma)
        sigmas = [sig] * nd + [0] * (s["image"].ndim - nd)
        out = dict(s)
        out["image"] = ndi.gaussian_filter(s["image"].astype(np.float32), sigmas).astype(s["image"].dtype)
        return out


class RandLowRes(Transform):
    """Simulate lower acquisition resolution: downsample then upsample."""

    def __init__(self, zoom=(0.5, 1.0), p=0.2):
        super().__init__(p)
        self.zoom = zoom

    def apply(self, s, rng):
        nd = _nd(s)
        z = rng.uniform(*self.zoom)
        img = s["image"].astype(np.float32)
        factors = [z] * nd + [1] * (img.ndim - nd)
        small = ndi.zoom(img, factors, order=1)
        back = ndi.zoom(small, [t / sm for t, sm in zip(img.shape, small.shape)], order=1)
        out = dict(s)
        out["image"] = back.astype(s["image"].dtype)
        return out


class RandBiasField(Transform):
    """Smooth multiplicative field (MRI intensity inhomogeneity / uneven illumination)."""

    def __init__(self, strength=0.3, p=0.3):
        super().__init__(p)
        self.strength = strength

    def apply(self, s, rng):
        nd = _nd(s)
        shape = s["image"].shape[:nd]
        low = rng.normal(0, 1, tuple(max(2, n // 32) for n in shape)).astype(np.float32)
        field = ndi.zoom(low, [n / l for n, l in zip(shape, low.shape)], order=3)
        field = np.exp(field / (np.abs(field).max() + 1e-8) * self.strength)
        if s["image"].ndim > nd:
            field = field[..., None]
        out = dict(s)
        out["image"] = (s["image"].astype(np.float32) * field).astype(s["image"].dtype)
        return out


class ClipScale(Transform):
    """Deterministic window + scale (e.g., CT HU window). Not random."""

    def __init__(self, lo=-200, hi=300, out_lo=0.0, out_hi=1.0):
        super().__init__(1.0)
        self.lo, self.hi, self.out_lo, self.out_hi = lo, hi, out_lo, out_hi

    def apply(self, s, rng):
        x = np.clip(s["image"].astype(np.float32), self.lo, self.hi)
        out = dict(s)
        out["image"] = (x - self.lo) / (self.hi - self.lo) * (self.out_hi - self.out_lo) + self.out_lo
        return out
