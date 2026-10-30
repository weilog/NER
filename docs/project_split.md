# Project split v2

Seed: 718853259, generated once from the system RNG.
Train/Dev/Test: 9539/1343/1209; effective training: 9538.
Original Dev preserved byte-for-byte. One nested sample is reserved in Train and excluded from model training and prompt examples.
No duplicate texts across labeled source splits; all original Train samples are assigned exactly once.
Detailed counts and hashes: data/splits/cluener2020/manifest.json.
Reproduce: python scripts/refresh_split.py (reuses the saved seed).
Old seed42 split is archived; do not run the old split_cluener.py entry point.
