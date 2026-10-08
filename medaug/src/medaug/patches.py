"""Patch / tile sampling for images too large to augment whole."""
import itertools
import numpy as np


def sample_patch(image, mask=None, size=256, margin=0, min_fg_frac=0.0,
                 rng=None, max_tries=20):
    """Random crop of `size` (+ `margin` on each side, so later spatial transforms
    don't create border artifacts). `image` is channel-less (H,W) or (D,H,W),
    or (H,W,C) when it has one more dim than `mask`.

    If `mask` is given and `min_fg_frac` > 0, retries until the center crop keeps at
    least that fraction of foreground (lesion-retention check); falls back to the
    best candidate seen.
    Returns (image_patch, mask_patch_or_None, origin).
    """
    rng = rng or np.random.default_rng()
    nd = mask.ndim if mask is not None else (image.ndim if image.ndim <= 3 and image.shape[-1] > 4 else image.ndim - 1)
    spatial = image.shape[:nd]
    size = (size,) * nd if np.isscalar(size) else tuple(size)
    full = tuple(min(s + 2 * margin, d) for s, d in zip(size, spatial))

    best, best_frac = None, -1.0
    for _ in range(max_tries):
        origin = tuple(int(rng.integers(0, d - f + 1)) for d, f in zip(spatial, full))
        if mask is None or min_fg_frac <= 0:
            best = origin
            break
        sl = tuple(slice(o + (f - s) // 2, o + (f - s) // 2 + s)
                   for o, f, s in zip(origin, full, size))
        frac = float((mask[sl] > 0).mean())
        if frac > best_frac:
            best, best_frac = origin, frac
        if frac >= min_fg_frac:
            break

    sl = tuple(slice(o, o + f) for o, f in zip(best, full))
    return image[sl].copy(), (mask[sl].copy() if mask is not None else None), best


def center_crop(arr, size, nd=None):
    nd = nd or arr.ndim
    size = (size,) * nd if np.isscalar(size) else tuple(size)
    sl = []
    for ax in range(nd):
        start = (arr.shape[ax] - size[ax]) // 2
        sl.append(slice(start, start + size[ax]))
    return arr[tuple(sl)]


def iter_tiles(shape, tile=512, stride=None):
    """Yield slice tuples tiling a large image (whole-slide / big volumes).
    Edge tiles are clamped, so the last tile may overlap its neighbour; coverage is complete."""
    stride = stride or tile
    ranges = []
    for d in shape:
        if d <= tile:
            ranges.append([0])
        else:
            starts = list(range(0, d - tile + 1, stride))
            if starts[-1] + tile < d:
                starts.append(d - tile)
            ranges.append(starts)
    for starts in itertools.product(*ranges):
        yield tuple(slice(s, min(s + tile, d)) for s, d in zip(starts, shape))
