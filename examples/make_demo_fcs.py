"""Write a synthetic 6-channel FCS file to try cykit on.

Three populations with realistic-looking flow data: log-normal fluorescence
with a noise floor that pushes a chunk of the negatives below zero, which is
exactly the regime a biex axis exists to show.

    python examples/make_demo_fcs.py demo.fcs
"""

from __future__ import annotations

import sys
from pathlib import Path

import flowio
import numpy as np

CHANNELS = ["FSC-A", "SSC-A", "FITC-A", "PE-A", "APC-A", "Time"]
MARKERS = ["", "", "CD3", "CD19", "CD8", ""]
FLUOR = [2, 3, 4]
# Per-channel autofluorescence, i.e. where the unstained population sits.
AUTOFLUOR = [30.0, 20.0, 15.0]


def spillover_matrix():
    """The mild crosstalk between the three fluorescence detectors, as truth."""
    spill = np.eye(3)
    spill[0, 1], spill[1, 2], spill[1, 0] = 0.12, 0.08, 0.05
    return spill


def _fluor(rng, n, positive, bright=3.5, noise=60.0):
    """Log-normal signal for positives, pure noise for negatives."""
    out = rng.normal(0.0, noise, n)
    hit = rng.random(n) < positive
    out[hit] += 10 ** rng.normal(bright, 0.25, hit.sum())
    return out


def make_events(n=60_000, seed=0):
    rng = np.random.default_rng(seed)
    fracs = [0.5, 0.3, 0.2]
    sizes = [int(n * f) for f in fracs]
    sizes[-1] = n - sum(sizes[:-1])

    blocks = []
    # T cells, B cells, debris-ish population
    for size, (fsc, ssc), (p_cd3, p_cd19, p_cd8) in zip(
        sizes,
        [(60_000, 25_000), (55_000, 30_000), (20_000, 15_000)],
        [(0.97, 0.02, 0.35), (0.02, 0.96, 0.01), (0.05, 0.05, 0.05)],
    ):
        blocks.append(
            np.column_stack(
                [
                    rng.normal(fsc, fsc * 0.09, size),
                    rng.normal(ssc, ssc * 0.15, size),
                    _fluor(rng, size, p_cd3),
                    _fluor(rng, size, p_cd19),
                    _fluor(rng, size, p_cd8, bright=3.2),
                ]
            )
        )
    events = np.vstack(blocks)
    rng.shuffle(events)
    time = np.sort(rng.uniform(0, 3000, events.shape[0]))[:, None]
    return np.hstack([events, time]).astype(np.float32)


def _write(path, events, meta):
    with open(path, "wb") as fh:
        flowio.create_fcs(fh, events.flatten().tolist(), CHANNELS, MARKERS, meta)
    print(f"wrote {path}: {events.shape[0]} events x {len(CHANNELS)} channels")


def make_control(rng, n, stained=None, bright=3.4):
    """One compensation control: beads that are negative or positive in one detector.

    ``stained=None`` gives the unstained control — every event negative, sitting
    on the same autofluorescence as the negatives in the stained tubes.
    """
    fsc = rng.normal(45_000, 4_000, n)[:, None]
    ssc = rng.normal(20_000, 2_500, n)[:, None]
    fluor = rng.normal(AUTOFLUOR, 60.0, (n, len(FLUOR)))
    if stained is not None:
        hit = rng.random(n) < 0.6
        fluor[hit, stained] += 10 ** rng.normal(bright, 0.2, hit.sum())
    fluor = fluor @ spillover_matrix()
    time = np.sort(rng.uniform(0, 600, n))[:, None]
    return np.hstack([fsc, ssc, fluor, time]).astype(np.float32)


def write_controls(directory, n=20_000, seed=7):
    """Write an unstained control plus one single-stain control per detector.

    No ``$SPILLOVER`` keyword: these are the files you use when the instrument
    did *not* hand you a matrix.

    Parameters
    ----------
    directory
        Directory to write into; created if it does not exist.
    n
        Events per control file.
    seed
        Seed for the random number generator.

    Returns
    -------
    list of Path
        The files written, unstained first.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    paths = [directory / "Unstained.fcs"]
    _write(paths[0], make_control(rng, n), {"$CYT": "cykit demo", "$SRC": "Unstained"})
    for k, j in enumerate(FLUOR):
        path = directory / f"Compensation Controls_{CHANNELS[j]}.fcs"
        _write(path, make_control(rng, n, stained=k), {"$CYT": "cykit demo", "$SRC": CHANNELS[j]})
        paths.append(path)
    return paths


def main(path="demo.fcs", n=60_000, controls=None):
    """Write the demo file, and optionally a directory of compensation controls.

    Parameters
    ----------
    path
        FCS file to write.
    n
        Number of events.
    controls
        Directory to write single-stain controls into, or None to skip them.
    """
    events = make_events(n)
    spill = spillover_matrix()
    events[:, FLUOR] = events[:, FLUOR] @ spill

    names = str(len(FLUOR)) + "," + ",".join(CHANNELS[i] for i in FLUOR)
    spill_kw = names + "," + ",".join(f"{v:g}" for v in spill.ravel())
    meta = {
        "$SPILLOVER": spill_kw,
        "$CYT": "cykit demo",
        "$TIMESTEP": "0.01",
        "$SRC": Path(path).stem,
    }
    _write(path, events, meta)
    if controls is not None:
        write_controls(controls)


if __name__ == "__main__":
    argv = sys.argv[1:] or ["demo.fcs"]
    main(argv[0], controls=argv[1] if len(argv) > 1 else None)
