"""Polygon gating on top of the napari canvas."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd

__all__ = [
    "GateRecord",
    "add_gate",
    "ellipse_mask",
    "gate_children",
    "gate_mask",
    "gate_order",
    "gate_record",
    "gate_stats",
    "polygon_mask",
    "recompute_gates",
    "rectangle_to_polygon",
    "shapes_mask",
]


def rectangle_to_polygon(verts: np.ndarray) -> np.ndarray:
    """Normalise a napari rectangle into four polygon corners.

    napari already stores a rectangle as its four corners, so this is usually
    a pass-through; anything else is reduced to its axis-aligned bounding box.

    Parameters
    ----------
    verts
        ``(n, 2)`` array of shape vertices, in pixel coordinates.

    Returns
    -------
    ndarray
        ``(4, 2)`` array of corners, counter-clockwise from the minimum corner.
    """
    verts = np.asarray(verts, dtype=float)
    if verts.shape[0] == 4:
        return verts
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    return np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]], [hi[0], lo[1]]])


def ellipse_mask(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Select points inside the ellipse inscribed in a bounding box.

    Parameters
    ----------
    points
        ``(n, 2)`` array of point coordinates.
    verts
        ``(4, 2)`` corners of the ellipse's bounding box, as napari stores an
        ellipse shape. A zero-width axis never matches.

    Returns
    -------
    ndarray
        Boolean mask of length ``n``.
    """
    verts = np.asarray(verts, dtype=float)
    centre = verts.mean(axis=0)
    radii = (verts.max(axis=0) - verts.min(axis=0)) / 2.0
    radii = np.where(radii == 0, np.inf, radii)
    d = (points - centre) / radii
    return np.sum(d**2, axis=1) <= 1.0


def polygon_mask(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Select points falling inside a polygon.

    Parameters
    ----------
    points
        ``(n, 2)`` array of point coordinates.
    verts
        ``(m, 2)`` polygon vertices, in the same coordinate space as
        ``points``. Fewer than three vertices enclose nothing.

    Returns
    -------
    ndarray
        Boolean mask of length ``n``.
    """
    from matplotlib.path import Path

    points = np.asarray(points, dtype=float)
    verts = np.asarray(verts, dtype=float)
    if verts.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=bool)
    return Path(verts).contains_points(points)


def shapes_mask(
    points: np.ndarray,
    shapes: Sequence[np.ndarray],
    shape_types: Sequence[str] | None = None,
) -> np.ndarray:
    """Select points inside any of a napari Shapes layer's shapes.

    Drawing several shapes therefore gives their union, which is how a
    population split across two blobs gets gated in one go.

    Parameters
    ----------
    points
        ``(n, 2)`` array of point coordinates, in canvas pixel space.
    shapes
        The Shapes layer's ``.data``: one vertex array per shape.
    shape_types
        The layer's ``.shape_type``, parallel to ``shapes``. ``ellipse`` is
        treated as an inscribed ellipse and everything else as a polygon.
        Defaults to treating every shape as a polygon.

    Returns
    -------
    ndarray
        Boolean mask of length ``n``.
    """
    mask = np.zeros(points.shape[0], dtype=bool)
    types = list(shape_types) if shape_types is not None else ["polygon"] * len(shapes)
    for verts, kind in zip(shapes, types):
        verts = np.asarray(verts, dtype=float)
        if kind == "ellipse":
            mask |= ellipse_mask(points, verts)
        elif kind in ("rectangle", "polygon", "path", "line"):
            mask |= polygon_mask(
                points, rectangle_to_polygon(verts) if kind == "rectangle" else verts
            )
    return mask


#: How "use ``adata.X``" is spelled inside ``uns``. Stored as a string so a
#: gate record survives a round trip through h5ad.
LAYER_X = "X"


def _as_shapes(value) -> list[np.ndarray]:
    """Outlines from a record, as float arrays.

    Written for the fact that ``uns`` comes back from h5ad holding numpy
    arrays where it was given lists, so neither ``if value`` nor ``value or
    []`` is safe here -- both raise on an array with more than one element.
    """
    if value is None:
        return []
    return [np.asarray(v, dtype=float) for v in value]


@dataclass(frozen=True)
class GateRecord:
    """A gate as it was recorded, decoded once so callers stop re-parsing ``uns``.

    Built by :func:`gate_record`. Read-only: :func:`add_gate` remains the only
    thing that writes a gate. ``raw`` keeps the untouched ``uns`` entry, because
    :func:`add_gate` merges arbitrary ``meta`` into a record and a closed type
    would silently drop whatever it did not know about.
    """

    name: str
    parent: str = ""
    x: str = ""
    y: str = ""
    layer: str = LAYER_X
    vertices: list[np.ndarray] = field(default_factory=list)
    shape_types: list[str] = field(default_factory=list)
    n: int = 0
    kind: str = "density"
    raw: dict = field(default_factory=dict)

    def has_outline(self) -> bool:
        """Whether this gate was drawn, and so can be recomputed or redrawn."""
        return bool(self.vertices) and bool(self.x) and bool(self.y)

    def shapes(self) -> Iterator[tuple[np.ndarray, str]]:
        """Each outline with the kind of shape it was drawn as."""
        return zip(self.vertices, self.shape_types)


def gate_record(adata: ad.AnnData, name: str) -> GateRecord:
    """The recorded gate ``name``, decoded.

    Every reader of ``uns['cytopy']['gates']`` goes through here, so the
    sentinel for ``adata.X``, the default shape kind and the h5ad round trip
    are handled once rather than at each call site.

    Parameters
    ----------
    adata
        AnnData carrying the gate in ``uns['cytopy']['gates']``.
    name
        The gate to look up.

    Returns
    -------
    GateRecord
        The decoded record.

    Raises
    ------
    KeyError
        If the gate was never recorded. The message lists the ones that are.
    """
    gates = adata.uns.get("cytopy", {}).get("gates", {})
    if name not in gates:
        raise KeyError(f"no gate {name!r}; recorded gates are {sorted(gates)}")
    raw = dict(gates[name])

    vertices = _as_shapes(raw.get("vertices"))
    kinds = raw.get("shape_types")
    kinds = [] if kinds is None else [str(k) for k in kinds]
    if len(kinds) != len(vertices):
        kinds = ["polygon"] * len(vertices)

    stored = str(raw.get("layer", LAYER_X))
    return GateRecord(
        name=name,
        parent=str(raw.get("parent") or ""),
        x=str(raw.get("x") or ""),
        y=str(raw.get("y") or ""),
        layer=LAYER_X if stored == "" else stored,
        vertices=vertices,
        shape_types=kinds,
        n=int(raw.get("n", 0)),
        kind=str(raw.get("kind") or "density"),
        raw=raw,
    )


def gate_order(adata: ad.AnnData) -> list[str]:
    """Every recorded gate, parents before their children.

    Depth first, so a gate always appears after the one it is nested in.
    Anything whose parent is missing is treated as top level rather than
    dropped, so a partly hand-built hierarchy still lists in full.

    Parameters
    ----------
    adata
        AnnData carrying the gates.

    Returns
    -------
    list of str
        Gate names, parents first.
    """
    gates = adata.uns.get("cytopy", {}).get("gates", {})
    parents = {g: str(record.get("parent") or "") for g, record in gates.items()}

    order: list[str] = []

    def _walk(under: str) -> None:
        """Depth first, appending every gate nested under ``under``."""
        for gate, parent in parents.items():
            if parent == under and gate not in order:
                order.append(gate)
                _walk(gate)

    _walk("")
    for gate in gates:  # orphans: a parent that was never recorded
        if gate not in order:
            order.append(gate)
    return order


def add_gate(
    adata: ad.AnnData,
    name: str,
    mask: np.ndarray,
    *,
    parent: str | None = None,
    within: np.ndarray | None = None,
    meta: dict | None = None,
) -> np.ndarray:
    """Record a gate as a boolean column in ``adata.obs``.

    Parameters
    ----------
    adata
        The AnnData to annotate. Modified in place.
    name
        Column name to write in ``adata.obs``. An existing column of the same
        name is overwritten.
    mask
        Boolean mask over all ``adata.n_obs`` events.
    parent
        Name of an existing gate to nest this one inside. The mask is
        intersected with it, so gates compose into a hierarchy the way they do
        in FlowJo. ``None`` gates the whole file.
    within
        Events this gate was drawn over. Anything outside keeps whatever it
        had, so a gate can be drawn one sample at a time under a single name:
        gating the second does not wipe the first. ``None`` rewrites the whole
        column.
    meta
        Extra provenance (plotted channels, layer, ...) merged into the gate's
        record in ``adata.uns['cytopy']['gates'][name]``.

    Returns
    -------
    ndarray
        The mask actually stored, i.e. after intersection with ``parent``.

    Raises
    ------
    ValueError
        If ``mask`` is not one entry per event.
    KeyError
        If ``parent`` names a gate that does not exist.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != adata.n_obs:
        raise ValueError(f"mask has {mask.shape[0]} entries, expected {adata.n_obs}")
    if parent:
        if parent not in adata.obs:
            raise KeyError(f"parent gate {parent!r} not in adata.obs")
        mask = mask & adata.obs[parent].to_numpy(dtype=bool)
    if within is not None:
        scope = np.asarray(within, dtype=bool)
        if scope.shape[0] != adata.n_obs:
            raise ValueError(f"within has {scope.shape[0]} entries, expected {adata.n_obs}")
        previous = (
            adata.obs[name].to_numpy(dtype=bool)
            if name in adata.obs
            else np.zeros(adata.n_obs, dtype=bool)
        )
        mask = np.where(scope, mask, previous)
    adata.obs[name] = pd.Series(mask, index=adata.obs_names)
    gates = adata.uns.setdefault("cytopy", {}).setdefault("gates", {})
    record = gates.get(name, {}) if within is not None else {}
    gates[name] = {**record, "parent": parent or "", "n": int(mask.sum()), **(meta or {})}
    return mask


def gate_stats(adata: ad.AnnData, name: str, parent: str | None = None) -> dict:
    """Event count and frequency for a gate.

    Parameters
    ----------
    adata
        AnnData carrying the gate as a boolean ``obs`` column.
    name
        The gate to summarise.
    parent
        Gate to express the frequency relative to, in addition to the whole
        file. Omitted from the result when not given.

    Returns
    -------
    dict
        ``gate``, ``n``, ``pct_total`` and, with ``parent``, ``pct_parent``.
    """
    mask = adata.obs[name].to_numpy(dtype=bool)
    n = int(mask.sum())
    out = {"gate": name, "n": n, "pct_total": 100.0 * n / max(adata.n_obs, 1)}
    if parent and parent in adata.obs:
        n_parent = int(adata.obs[parent].to_numpy(dtype=bool).sum())
        out["pct_parent"] = 100.0 * n / max(n_parent, 1)
    return out


def gate_mask(adata: ad.AnnData, name: str) -> np.ndarray:
    """Recompute a recorded gate from the outline it was drawn with.

    The outline is kept in data coordinates, so this does not need the canvas
    it was drawn on. The canvas maps data to pixels with an affine transform,
    which leaves the inside of a shape unchanged, so this reproduces exactly
    what the viewer computed at the time.

    Parameters
    ----------
    adata
        AnnData carrying the gate in ``uns['cytopy']['gates']``.
    name
        The gate to recompute.

    Returns
    -------
    ndarray
        Boolean mask over all events, already intersected with the gate's
        parent.

    Raises
    ------
    KeyError
        If the gate was never recorded, or recorded no outline -- one applied
        from a mask rather than drawn cannot be recomputed.
    """
    record = gate_record(adata, name)
    if not record.has_outline():
        raise KeyError(f"gate {name!r} has no outline to recompute from")

    from ._util import layer_matrix
    from .transforms import find_channel_name

    matrix = layer_matrix(adata, record.layer)
    xi = find_channel_name(adata, record.x)
    yi = find_channel_name(adata, record.y)
    points = np.column_stack(
        [
            np.asarray(matrix[:, xi], dtype=np.float64).ravel(),
            np.asarray(matrix[:, yi], dtype=np.float64).ravel(),
        ]
    )
    mask = shapes_mask(points, record.vertices, record.shape_types)

    if record.parent and record.parent in adata.obs:
        mask = mask & adata.obs[record.parent].to_numpy(dtype=bool)
    return mask


def gate_children(adata: ad.AnnData, name: str) -> list[str]:
    """Gates nested directly inside ``name``.

    Parameters
    ----------
    adata
        AnnData carrying the gates.
    name
        The parent gate.

    Returns
    -------
    list of str
        Child gate names, in the order they were recorded.
    """
    gates = adata.uns.get("cytopy", {}).get("gates", {})
    return [g for g, record in gates.items() if str(record.get("parent") or "") == name]


def recompute_gates(adata: ad.AnnData, name: str) -> list[str]:
    """Bring a gate's descendants back in line after it has been changed.

    A child is stored as the events inside its own outline *and* inside its
    parent, worked out when it was drawn. Move the parent and the child is
    stale until it is recomputed, which is what this does, depth first.

    Parameters
    ----------
    adata
        AnnData carrying the gates. Modified in place.
    name
        The gate that changed. It is not itself recomputed.

    Returns
    -------
    list of str
        The descendants that were updated, parents before children. A child
        with no stored outline is skipped and left as it was.
    """
    updated: list[str] = []
    for child in gate_children(adata, name):
        try:
            mask = gate_mask(adata, child)
        except KeyError:
            continue
        adata.obs[child] = pd.Series(mask, index=adata.obs_names)
        adata.uns["cytopy"]["gates"][child]["n"] = int(mask.sum())
        updated.append(child)
        updated += recompute_gates(adata, child)
    return updated
