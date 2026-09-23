"""End-to-end: FCS -> AnnData -> compensate -> transform -> napari.

python examples/make_demo_fcs.py demo.fcs controls
python examples/quickstart.py demo.fcs controls
"""

from __future__ import annotations

import sys

import cytopy


def main(path: str = "demo.fcs", controls: str | None = None) -> None:
    adata = cytopy.read_fcs(path)
    print(adata)
    print("fluorescence channels:", cytopy.fluor_channels(adata))

    # Every transform is explicit: the viewer plots whatever layer you point it
    # at, exactly as stored.
    spillover = None
    if controls is not None:
        # No matrix from the instrument? Derive one from the single-stain
        # controls instead (Bagwell and Adams).
        stained, unstained = cytopy.read_controls(controls)
        spillover = cytopy.compute_spillover_matrix(stained, unstained=unstained)
        print("spillover from controls:\n", spillover.round(4))
    cytopy.compensate(adata, spillover, inplace=True)  # -> layers["comp"]
    cytopy.asinh_transform(adata, 150.0, layer="comp", inplace=True)  # -> layers["asinh"]

    # Pick the channels in the window. Gates drawn there are boolean columns
    # by the time this returns, because it blocks until you close it.
    adata = cytopy.open_napari(adata, "asinh", block=True, verbose=True)
    for name in adata.uns.get("cytopy", {}).get("gates", {}):
        print(cytopy.gate_stats(adata, name))
    return adata


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["demo.fcs"]))
