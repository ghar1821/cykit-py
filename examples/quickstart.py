"""End-to-end: FCS -> AnnData -> compensate -> transform -> napari.

python examples/make_demo_fcs.py demo.fcs controls
python examples/quickstart.py demo.fcs controls
"""

from __future__ import annotations

import sys
from pathlib import Path

import cykit

#: Which control file stains which detector, for the directory
#: ``make_demo_fcs.py`` writes. State this yourself: nothing infers it.
CONTROL_FILES = {
    "CD3 (FITC-A)": "Compensation Controls_FITC-A.fcs",
    "CD19 (PE-A)": "Compensation Controls_PE-A.fcs",
    "CD8 (APC-A)": "Compensation Controls_APC-A.fcs",
}
UNSTAINED_FILE = "Unstained.fcs"


def main(path: str = "demo.fcs", controls: str | None = None) -> None:
    adata = cykit.read_fcs(path)
    print(adata)
    print("fluorescence channels:", cykit.get_fluor_channels(adata))

    # Every transform is explicit: the viewer plots whatever layer you point it
    # at, exactly as stored.
    spillover = None
    if controls is not None:
        # No matrix from the instrument? Derive one from the single-stain
        # controls instead (Bagwell and Adams). The mapping is yours to state.
        directory = Path(controls)
        stained = {
            detector: cykit.read_fcs(directory / name) for detector, name in CONTROL_FILES.items()
        }
        unstained = cykit.read_fcs(directory / UNSTAINED_FILE)
        spillover = cykit.compute_spillover_matrix(stained, unstained=unstained)
        print("spillover from controls:\n", spillover.round(4))
    cykit.compensate(adata, spillover, inplace=True)  # -> layers["comp"]
    cykit.asinh_transform(adata, 150.0, layer="comp", inplace=True)  # -> layers["asinh"]

    # Pick the channels in the window. Gates drawn there are boolean columns
    # by the time this returns, because it blocks until you close it.
    adata = cykit.open_napari(adata, "asinh", block=True, verbose=True)
    for name in adata.uns.get("cykit", {}).get("gates", {}):
        print(cykit.gate_stats(adata, name))
    return adata


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["demo.fcs"]))
