"""cytopy: cytometry analysis on AnnData, gated interactively in napari."""

from __future__ import annotations

from .density import Axes2D, density_curve, density_image
from .gating import (
    GateRecord,
    add_gate,
    ellipse_mask,
    gate_children,
    gate_mask,
    gate_order,
    gate_record,
    gate_stats,
    polygon_mask,
    recompute_gates,
    rectangle_to_polygon,
    shapes_mask,
)
from .io import concat_samples, read_fcs, read_fcs_dir, split_samples
from .plotting import gating_pdf, plot_biaxial, plot_gate
from .scales import (
    AsinhScale,
    LinearScale,
    LogicleScale,
    LogScale,
    PretransformedScale,
    Scale,
    get_scale,
)
from .spillover import (
    compensate,
    compute_spillover_matrix,
    subset_controls,
    write_spillover,
)
from .transforms import (
    asinh_transform,
    find_channel_name,
    get_fluor_channels,
    logicle_transform,
    subsample,
)

__version__ = "0.0.1-alpha"

__all__ = [
    "AsinhScale",
    "Axes2D",
    "CofactorWindow",
    "Cofactors",
    "CytoViewer",
    "GateRecord",
    "LinearScale",
    "LogScale",
    "LogicleScale",
    "Panel",
    "PretransformedScale",
    "Scale",
    "add_gate",
    "as_one_anndata",
    "asinh_transform",
    "compensate",
    "compute_spillover_matrix",
    "concat_samples",
    "current_transform_window",
    "current_viewer",
    "density_curve",
    "density_image",
    "ellipse_mask",
    "faded_colormap",
    "find_channel_name",
    "gate_children",
    "gate_mask",
    "gate_order",
    "gate_record",
    "gate_stats",
    "gating_pdf",
    "get_fluor_channels",
    "get_scale",
    "logicle_transform",
    "open_napari",
    "open_napari_transform",
    "plot_biaxial",
    "plot_gate",
    "polygon_mask",
    "read_fcs",
    "read_fcs_dir",
    "recompute_gates",
    "rectangle_to_polygon",
    "shapes_mask",
    "split_samples",
    "subsample",
    "subset_controls",
    "write_spillover",
]


#: Names served from the modules that pull in napari and Qt, mapped to the
#: module each comes from. Listed here rather than read off their ``__all__``
#: because importing them to find out would defeat the point;
#: ``tests/test_api.py`` checks the lists agree.
_LAZY = {
    "CofactorWindow": "cofactors",
    "Cofactors": "cofactors",
    "CytoViewer": "viewer",
    "Panel": "viewer",
    "as_one_anndata": "viewer",
    "current_transform_window": "cofactors",
    "current_viewer": "viewer",
    "faded_colormap": "viewer",
    "open_napari": "viewer",
    "open_napari_transform": "cofactors",
}


def __getattr__(name: str):
    # napari (and Qt) are heavy and optional: import them only on first use.
    module = _LAZY.get(name)
    if module is not None:
        import importlib

        return getattr(importlib.import_module(f".{module}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
