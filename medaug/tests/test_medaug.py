import numpy as np, yaml, os, pytest
from medaug import build_pipeline, patient_split, sample_patch, iter_tiles
from medaug import transforms as T
from medaug.stain import RandHEDJitter

CFG = os.path.join(os.path.dirname(__file__), "..", "configs")
rng = lambda: np.random.default_rng(0)


def test_split_no_patient_overlap():
    ids = [f"p{i}" for i in range(40)]
    sp = patient_split(ids, seed=1)
    assert not (sp["train"] & sp["val"]) and not (sp["train"] & sp["test"]) and not (sp["val"] & sp["test"])
    assert sp["train"] | sp["val"] | sp["test"] == set(ids)


@pytest.mark.parametrize("shape", [(96, 96), (40, 40, 40)])
def test_spatial_transforms_keep_mask_binary_and_aligned(shape):
    nd = len(shape)
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    mask = (sum((g - n / 2) ** 2 for g, n in zip(grids, shape)) < (min(shape) / 5) ** 2).astype(np.uint8)
    img = mask.astype(np.float32) * 100 + 1.0   # image perfectly tied to mask
    for t in [T.RandFlip(axes=tuple(range(nd)), p=1), T.RandAffine(p=1), T.RandElastic(sigma=5, alpha=3, p=1)]:
        out = t({"image": img, "mask": mask}, rng())
        assert set(np.unique(out["mask"])) <= {0, 1}
        assert out["mask"].shape == shape and out["image"].shape == shape
        # image/mask must stay spatially aligned (image bright where mask==1)
        agree = ((out["image"] > 50) == (out["mask"] == 1)).mean()
        assert agree > 0.97, (type(t).__name__, agree)


def test_patch_lesion_retention():
    mask = np.zeros((512, 512), np.uint8)
    mask[200:260, 300:360] = 1
    img = np.zeros_like(mask, dtype=np.float32)
    kept = 0
    r = np.random.default_rng(3)
    for _ in range(20):
        _, m, _ = sample_patch(img, mask, size=128, margin=0, min_fg_frac=0.05, rng=r, max_tries=200)
        kept += (m > 0).mean() >= 0.05
    assert kept >= 18


def test_pipeline_output_size_ct():
    cfg = yaml.safe_load(open(os.path.join(CFG, "ct3d.yaml")))
    p = build_pipeline(cfg, seed=0)
    vol = np.random.default_rng(0).normal(40, 20, (96, 96, 96)).astype(np.float32)
    seg = (vol > 70).astype(np.uint8)
    s = p(vol, seg)
    assert s["image"].shape == (64, 64, 64) == s["mask"].shape
    assert 0.0 <= s["image"].min() and s["image"].max() <= 1.0 + 1e-3


def test_pipeline_deterministic_with_seed():
    cfg = yaml.safe_load(open(os.path.join(CFG, "histology2d.yaml")))
    img = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    a = build_pipeline(cfg, seed=7)(img)["image"]
    b = build_pipeline(cfg, seed=7)(img)["image"]
    assert np.array_equal(a, b) and a.shape == (256, 256, 3)


def test_hed_jitter_changes_colour_but_keeps_dtype():
    img = np.random.default_rng(0).integers(30, 230, (64, 64, 3), dtype=np.uint8)
    out = RandHEDJitter(sigma=0.1, bias=0.1, p=1)({"image": img}, rng())["image"]
    assert out.dtype == np.uint8 and not np.array_equal(out, img)


def test_clip_scale_window():
    x = np.array([[-1000, -200, 50, 300, 1000]], np.float32)
    out = T.ClipScale(-200, 300)({"image": x}, rng())["image"]
    assert out.min() == 0 and out.max() == 1


def test_iter_tiles_cover():
    cover = np.zeros((300, 500), bool)
    for sl in iter_tiles(cover.shape, tile=128):
        cover[sl] = True
    assert cover.all()
