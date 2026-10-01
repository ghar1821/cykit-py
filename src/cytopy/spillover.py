"""Spillover matrices: derive one from single-stain controls, or read one in.

There are three ways to get a matrix, in increasing order of effort:

* the acquisition software already computed one and wrote it into the FCS
  file, where :func:`~cytopy.read_fcs` picks it up as ``adata.uns['spillover']``;
* some other program exported one, and you read its CSV into a DataFrame
  yourself -- ``pd.read_csv(path, index_col=0)`` for the usual layout;
* you have the single-stain controls, and :func:`compute_spillover_matrix`
  computes one from them by the Bagwell and Adams method. You say which file
  stains which detector; nothing here guesses that for you.

All three produce the same thing: a square DataFrame indexed by detector, with
``1`` on the diagonal, where ``S[i, j]`` is the fraction of dye *i*'s signal
that lands in detector *j*. :func:`~cytopy.compensate` takes it from there.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ._util import layer_matrix
from .transforms import find_channel_name, get_fluor_channels

__all__ = [
    "compensate",
    "compute_spillover_matrix",
    "subset_controls",
    "write_spillover",
]


# --------------------------------------------------------------------------
# reading and writing
# --------------------------------------------------------------------------
def write_spillover(spillover: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Write a spillover matrix as CSV, with detector names down the first column.

    Parameters
    ----------
    spillover
        Square matrix, as returned by :func:`compute_spillover_matrix`.
    path
        File to write.

    Returns
    -------
    Path
        The path written, so it can be chained.
    """
    path = Path(path)
    spillover.to_csv(path, index_label="detector")
    return path


# --------------------------------------------------------------------------
# single-stain controls
# --------------------------------------------------------------------------
def _otsu_threshold(values: np.ndarray, cofactor: float, nbins: int = 512) -> float:
    """Otsu's two-class threshold, found on an arcsinh scale and returned in raw units.

    Fluorescence is split on a compressed scale for the same reason it is
    plotted on one: on a linear axis the whole negative population sits in a
    single bin, and the threshold that maximises between-class variance lands
    somewhere in the middle of the positives.
    """
    scaled = np.arcsinh(np.asarray(values, dtype=np.float64) / cofactor)
    scaled = scaled[np.isfinite(scaled)]
    if scaled.size < 2:
        return float("inf")
    lo, hi = np.percentile(scaled, [0.1, 99.9])
    if not hi > lo:
        return float("inf")
    counts, edges = np.histogram(scaled, bins=nbins, range=(float(lo), float(hi)))
    total = counts.sum()
    if total == 0:
        return float("inf")
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight = np.cumsum(counts) / total
    mean = np.cumsum(counts * centres) / total
    spread = weight * (1.0 - weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(spread > 0, (mean[-1] * weight - mean) ** 2 / spread, 0.0)
    return float(np.sinh(centres[int(np.argmax(between))]) * cofactor)


def _positive_mask(column: np.ndarray, cofactor: float) -> tuple[np.ndarray, float]:
    """Boolean mask of the stained events in one channel, plus the threshold used."""
    threshold = _otsu_threshold(column, cofactor)
    return column > threshold, threshold


def _separation(column: np.ndarray, positive: np.ndarray, min_events: int = 1) -> float:
    """How far apart a split puts the two populations, in negative-MAD units.

    A genuinely stained control lands in the tens; splitting noise in half
    gives about 2, which is how a mislabelled tube is told apart from a real
    one.
    """
    if positive.sum() < min_events or (~positive).sum() < min_events:
        return -np.inf
    negative = column[~positive]
    centre = np.median(negative)
    mad = 1.4826 * np.median(np.abs(negative - centre))
    return float((np.median(column[positive]) - centre) / max(mad, 1e-9))


def _as_anndata(source: ad.AnnData | str | os.PathLike) -> ad.AnnData:
    if isinstance(source, ad.AnnData):
        return source
    from .io import read_fcs  # local: io imports this module for keyword parsing

    return read_fcs(source)


def _stat(values: np.ndarray, statistic: str) -> np.ndarray:
    if statistic == "median":
        return np.median(values, axis=0)
    if statistic == "mean":
        return np.mean(values, axis=0)
    raise ValueError(f"statistic must be 'median' or 'mean', not {statistic!r}")


def _resolve_detectors(
    loaded: Mapping[str, ad.AnnData], channels: Sequence[str] | None
) -> tuple[dict[str, str], list[str]]:
    """Each control's key as a ``var_name``, and the detectors the matrix covers.

    Every key is resolved against the first control's panel, so that ``"CD3"``,
    ``"FITC-A"`` and ``"CD3 (FITC-A)"`` all name the same row.
    """
    reference = next(iter(loaded.values()))
    primary = {key: str(reference.var_names[find_channel_name(reference, key)]) for key in loaded}
    if len(set(primary.values())) != len(primary):
        raise ValueError(f"two controls resolve to the same detector: {sorted(primary.values())}")

    if channels is None:
        names = [n for n in get_fluor_channels(reference) if n in set(primary.values())]
        names += [n for n in primary.values() if n not in names]
    else:
        names = [str(reference.var_names[find_channel_name(reference, c)]) for c in channels]
        missing = sorted(set(primary.values()) - set(names))
        if missing:
            raise ValueError(f"channels omits detectors that have controls: {missing}")
    return primary, names


def compute_spillover_matrix(
    controls: Mapping[str, ad.AnnData | str | os.PathLike],
    *,
    unstained: ad.AnnData | str | os.PathLike | None = None,
    channels: Sequence[str] | None = None,
    layer: str = "raw",
    statistic: str = "median",
    positive_gate: str | None = None,
    negative_gate: str | None = None,
    use_unstained: Collection[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
    cofactor: float = 150.0,
    min_events: int = 20,
    min_separation: float = 0.0,
) -> pd.DataFrame:
    """Compute a spillover matrix from single-stain controls (Bagwell and Adams).

    For the control stained with dye *i*, the signal it contributes to every
    detector *j* is the difference between its positive and negative
    populations, ``d_j``. Dividing that row through by ``d_i`` — the difference
    in the dye's own detector — gives spillover coefficients that are
    independent of how bright the control happened to be, and puts ``1`` on the
    diagonal by construction. Inverting the assembled matrix is what
    :func:`~cytopy.compensate` then does, which is the N-parameter
    generalisation Bagwell and Adams (1993) described.

    The negative population is taken from the control itself, which is what you
    want: it carries the same autofluorescence and the same beads as the
    positives. Pass ``unstained`` to use a universal negative instead.

    Parameters
    ----------
    controls
        Maps detector to its single-stain control — ``{"FITC-A": adata}``, or
        a path to the FCS file instead of an ``AnnData``. The key is resolved
        like any other channel name, so ``$PnS`` markers (``"CD3"``) work too.
    unstained
        Universal negative, as an ``AnnData`` or a path. When given, its
        per-channel statistic is the negative reference for every control
        rather than each control's own negative events.
    channels
        Detectors the matrix should cover. Defaults to the controls' own
        detectors, which is the only square set the controls constrain. A
        larger set is allowed — the extra detectors get identity rows, which
        asserts they spill into nothing.
    layer
        Layer of each control to measure, by name. Defaults to ``"raw"``:
        a matrix is derived from the values as acquired.
    statistic
        ``"median"`` (default, robust) or ``"mean"`` (as in the original
        paper). Applied to both populations.
    positive_gate
        Name of a boolean ``obs`` column marking the stained events, as
        written by :func:`~cytopy.add_gate`. Falls back to the automatic split
        for any control that lacks the column.
    negative_gate
        Name of a boolean ``obs`` column marking the control's negative
        events, as drawn in :func:`~cytopy.open_napari_compensation`. A
        control that has it uses those events as its negative reference, even
        when ``unstained`` is given; one without it falls back to the whole
        ``unstained``, or else to every event outside the positive population.
        A negative gate that overlaps the positive population is an error.
    use_unstained
        Controls whose negative reference is the whole ``unstained`` tube,
        keyed like ``controls`` (markers and detectors work too). Takes
        precedence over their ``negative_gate``, which is how a control with
        no usable negatives of its own -- beads that are all positive -- is
        handled. Requires ``unstained``.
    thresholds
        Per-control cutoff on the stained detector, in raw units, keyed like
        ``controls``. Overrides the automatic split. Pass ``-inf`` for a tube
        with no negative population of its own — an all-positive bead control
        — which only works alongside ``unstained``.
    cofactor
        Arcsinh cofactor for the automatic positive/negative split. It only
        affects where the split is looked for, never the returned
        coefficients; raise it if a dim control is being cut in the wrong
        place.
    min_events
        Smallest acceptable positive or negative population. A control below
        it cannot give a trustworthy median and is an error rather than a
        quietly bad row of the matrix.
    min_separation
        How far apart the automatic split must put the two populations, in
        negative-MAD units, before the control is believed to be stained at
        all. **Off by default**: the measure does not do the job it looks like
        it does. On real controls a dim but genuine tube scores 3.5 while an
        unstained tube scores 6.7, so any cutoff that passes the real one also
        passes the empty one. It is kept for clean bead controls, where the two
        are orders apart. The check that a control is brighter than its own
        negatives always applies, and is the one that catches a mismatched
        tube. Only consulted when the split is automatic -- an explicit
        ``positive_gate`` or ``thresholds`` entry is taken at face value.

    Returns
    -------
    DataFrame
        Square spillover matrix indexed and columned by the resolved detector
        names, with ``1`` on the diagonal. ``.attrs['cytopy']`` carries the
        per-control event counts, thresholds and where each negative came
        from (``"control gate"``, ``"unstained"`` or ``"complement"``) for
        troubleshooting.

    Raises
    ------
    ValueError
        If ``controls`` is empty, a control does not separate into two
        populations, a control's own detector shows no signal above its
        negatives, its positive and negative gates overlap, or ``channels``
        omits a detector that has a control.
    KeyError
        If a control is missing one of the detectors being solved for.
    """
    if not controls:
        raise ValueError("no controls given")
    _stat(np.zeros((1, 1)), statistic)  # fail on a bad statistic before reading files
    loaded = {key: _as_anndata(value) for key, value in controls.items()}
    primary, names = _resolve_detectors(loaded, channels)
    chosen: set[str] = set()
    if use_unstained:
        if unstained is None:
            raise ValueError("use_unstained needs the unstained tube: pass unstained=")
        reference = next(iter(loaded.values()))
        chosen = {str(reference.var_names[find_channel_name(reference, k)]) for k in use_unstained}
        unknown = sorted(chosen - set(primary.values()))
        if unknown:
            raise ValueError(f"use_unstained names detectors with no control: {unknown}")

    background = None
    if unstained is not None:
        unstained = _as_anndata(unstained)
        columns = [find_channel_name(unstained, n) for n in names]
        matrix = unstained.X if layer is None else unstained.layers[layer]
        background = _stat(np.asarray(matrix, dtype=np.float64)[:, columns], statistic)

    spill = np.eye(len(names))
    diagnostics: dict[str, dict[str, float]] = {}

    for key, control in loaded.items():
        detector = primary[key]
        row = names.index(detector)
        columns = [find_channel_name(control, n) for n in names]
        matrix = np.asarray(
            control.X if layer is None else control.layers[layer], dtype=np.float64
        )[:, columns]
        stained = matrix[:, row]

        automatic = False
        if thresholds is not None and key in thresholds:
            threshold = float(thresholds[key])
            positive = stained > threshold
        elif positive_gate is not None and positive_gate in control.obs:
            positive = np.asarray(control.obs[positive_gate], dtype=bool)
            threshold = float("nan")
        else:
            positive, threshold = _positive_mask(stained, cofactor)
            automatic = True

        negative = ~positive
        baseline = background
        source = "unstained" if background is not None else "complement"
        if detector in chosen:
            source = "unstained (chosen)"
        elif negative_gate is not None and negative_gate in control.obs:
            negative = np.asarray(control.obs[negative_gate], dtype=bool)
            if (negative & positive).any():
                raise ValueError(
                    f"control {key!r}: {int((negative & positive).sum())} events are in both "
                    f"the positive and the negative gate ({negative_gate!r})"
                )
            baseline, source = None, "control gate"

        if positive.sum() < min_events:
            raise ValueError(
                f"control {key!r} has {int(positive.sum())} positive events in {detector} "
                f"(need {min_events}); it may be too dim to split automatically"
            )
        if baseline is None and negative.sum() < min_events:
            raise ValueError(
                f"control {key!r} has {int(negative.sum())} negative events in {detector} "
                f"(need {min_events}); pass an unstained control to use as the negative"
            )

        gap = _separation(stained, positive, min_events)
        if automatic and min_separation and gap < min_separation:
            raise ValueError(
                f"control {key!r} does not separate into two populations in {detector} "
                f"(gap {gap:.1f} < {min_separation}); it may be the wrong tube, or too dim "
                "to split automatically — pass positive_gate or thresholds"
            )

        if baseline is None:
            baseline = _stat(matrix[negative], statistic)
            n_negative = int(negative.sum())
        else:
            n_negative = int(unstained.n_obs)
        difference = _stat(matrix[positive], statistic) - baseline
        if not difference[row] > 0:
            raise ValueError(
                f"control {key!r} is not brighter than its negatives in {detector}; "
                "check that the control is matched to the right detector"
            )
        spill[row] = difference / difference[row]
        diagnostics[detector] = {
            "positive_events": int(positive.sum()),
            "negative_events": n_negative,
            "threshold": threshold,
            "separation": gap,
            "negative": source,
        }

    out = pd.DataFrame(spill, index=pd.Index(names), columns=pd.Index(names))
    out.attrs["cytopy"] = {
        "method": "bagwell-adams",
        "statistic": statistic,
        "negative": "unstained" if background is not None else "internal",
        "controls": diagnostics,
    }
    return out


def subset_controls(
    controls: Mapping[str, ad.AnnData], gate: str, *, required: bool = True
) -> dict[str, ad.AnnData]:
    """Keep only the events inside a gate, in every control.

    The step between gating and computing a matrix: what comes back holds the
    cells and nothing else, so every median the matrix is built from is a
    median of cells rather than of cells and debris together.

    Parameters
    ----------
    controls
        Maps detector to its control.
    gate
        ``obs`` column to subset on, as gated with :func:`~cytopy.gate`.
    required
        Raise if a control has no such column. ``False`` passes it through
        whole, which is rarely what you want and never silent -- a control that
        was missed makes a quietly wrong matrix rather than an error.

    Returns
    -------
    dict
        The same keys, each control cut down to the gate.

    Raises
    ------
    KeyError
        If ``required`` and a control was never gated.
    ValueError
        If a gate holds no events.
    """
    out = {}
    for key, adata in controls.items():
        if gate not in adata.obs:
            if required:
                raise KeyError(f"control {key!r} has no gate {gate!r}; gate it first")
            print(f"control {key!r} has no gate {gate!r}, keeping all {adata.n_obs:,} events")
            out[key] = adata
            continue
        mask = np.asarray(adata.obs[gate], dtype=bool)
        if not mask.any():
            raise ValueError(f"gate {gate!r} holds no events in control {key!r}")
        out[key] = adata[mask].copy()
    return out


def compensate(
    adata: ad.AnnData,
    spillover: pd.DataFrame | np.ndarray | None = None,
    *,
    layer: str = "raw",
    key_added: str = "comp",
    inplace: bool = False,
) -> ad.AnnData:
    """Apply the spillover matrix, undoing fluorescence crosstalk between detectors.

    Compensated values are ``X @ inv(S)``.

    Parameters
    ----------
    adata
        Cytometry AnnData.
    spillover
        Square spillover matrix, in any of the forms cytopy can get one:

        * ``None`` (default) uses ``adata.uns['spillover']``, which
          :func:`~cytopy.read_fcs` parses from the file's ``$SPILLOVER``;
        * a DataFrame, whose index/columns name the detectors and need only
          cover a subset of the channels — as returned by
          :func:`~cytopy.compute_spillover_matrix`, which derives one from
          single-stain controls;
        * a bare array, assumed to be in :func:`get_fluor_channels` order.
    layer
        Input layer to read from, by name. Defaults to ``"raw"``: compensation
        undoes crosstalk in the values as acquired, so it belongs on the
        untouched matrix rather than on whatever has been done since.
    key_added
        Layer to write the compensated matrix to.
    inplace
        Modify ``adata`` and return it. The default is ``False``: the original
        is left alone and a modified copy is returned, so a call that is not
        assigned to anything cannot quietly change the data underneath you.

    Returns
    -------
    AnnData
        The annotated object, with the compensated matrix in
        ``adata.layers[key_added]`` and where the matrix came from in
        ``adata.uns['cytopy']['spillover_source']``. Channels absent from the
        matrix are copied through untouched.

    Raises
    ------
    ValueError
        If no matrix is given and the file recorded none, if the matrix is not
        square, if its rows and columns do not name the same detectors, or if
        it cannot be inverted.
    """
    adata = adata if inplace else adata.copy()
    source = "argument"
    if spillover is None:
        spillover = adata.uns.get("spillover")
        source = "uns"
    if spillover is None:
        raise ValueError("no spillover matrix given and none found in adata.uns['spillover']")
    if not isinstance(spillover, pd.DataFrame):
        names = get_fluor_channels(adata)
        values = np.asarray(spillover, dtype=float)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError(f"spillover must be a square matrix, got shape {values.shape}")
        if values.shape[0] != len(names):
            raise ValueError(
                f"spillover is {values.shape[0]}x{values.shape[1]} but there are "
                f"{len(names)} fluorescence channels; pass a DataFrame to name them"
            )
        spillover = pd.DataFrame(values, index=names, columns=names)
    if spillover.shape[0] != spillover.shape[1]:
        raise ValueError(f"spillover must be square, got shape {spillover.shape}")

    # Rows and columns must describe the same detectors, but need not list them
    # in the same order (or under the same alias), so resolve both and align.
    cols = [find_channel_name(adata, c) for c in spillover.columns]
    rows = [find_channel_name(adata, r) for r in spillover.index]
    if sorted(rows) != sorted(cols):
        raise ValueError(
            f"spillover rows {list(spillover.index)} and columns {list(spillover.columns)} "
            "do not name the same detectors"
        )
    if rows != cols:
        spillover = spillover.iloc[[rows.index(c) for c in cols]]

    X = np.asarray(layer_matrix(adata, layer), dtype=np.float64)
    out = X.copy()
    try:
        inv = np.linalg.inv(spillover.to_numpy(dtype=float))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"spillover matrix is singular and cannot be inverted: {exc}") from exc
    out[:, cols] = X[:, cols] @ inv
    adata.layers[key_added] = out.astype(adata.X.dtype, copy=False)
    cytopy_uns = adata.uns.setdefault("cytopy", {})
    cytopy_uns["compensated_layer"] = key_added
    cytopy_uns["spillover_source"] = source
    return adata
