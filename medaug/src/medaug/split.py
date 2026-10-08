"""Patient-level splitting: split BEFORE augmenting to avoid leakage."""
import numpy as np


def patient_split(patient_ids, val_frac=0.15, test_frac=0.15, seed=0):
    """Return dict of split -> set of patient ids. Every sample of a patient stays together."""
    ids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    return {
        "test": set(ids[:n_test]),
        "val": set(ids[n_test:n_test + n_val]),
        "train": set(ids[n_test + n_val:]),
    }
