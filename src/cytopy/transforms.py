"""Preprocessing of cytometry AnnData objects: compensation and transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import anndata as ad
import numpy as np

from ._util import layer_matrix, subsample_indices
from .scales import LogicleScale

__all__ = [
    "asinh_transform",
    "channel_index",
    "fluor_channels",
    "logicle_transform",
    "subsample",
]

DEFAULT_COFACTOR = {"flow": 150.0, "cytof": 5.0, "spectral": 3000.0}


def channel_index(adata: ad.AnnData, name: str) -> int:
    """Resolve a channel name to its column index.

    Parameters
    ----------
    adata
        Cytometry AnnData.
    name
        A ``var_name``, a ``$PnN`` detector (``"FITC-A"``), a ``$PnS`` marker
        (``"CD3"``) or the combined label (``"CD3 (FITC-A)"``). Matched in that
        order, then case-insensitively against the var_names.

    Returns
    -------
    int
        Column index into ``adata.X`` and every layer.

    Raises
    ------
    KeyError
        If nothing matches. The message lists the available channels.
    """
    if name in adata.var_names:
        return int(adata.var_names.get_loc(name))
    for col in ("channel", "marker", "label"):
        if col in adata.var:
            hits = np.flatnonzero(adata.var[col].astype(str).values == name)
            if hits.size:
                return int(hits[0])
    lowered = {str(v).lower(): i for i, v in enumerate(adata.var_names)}
    if name.lower() in lowered:
        return lowered[name.lower()]
    raise KeyError(f"channel {name!r} not found; available: {list(adata.var_names)}")


def fluor_channels(adata: ad.AnnData) -> list[str]:
    """The fluorescence channels, i.e. everything but scatter and time.

    This is the default set that compensation and the transforms act on.

    Parameters
    ----------
    adata
        Cytometry AnnData. Without a ``var['kind']`` column (which
        :func:`~cytopy.read_fcs` adds) every channel is returned.

    Returns
    -------
    list of str
        var_names, in file order.
    """
    if "kind" not in adata.var:
        return list(adata.var_names)
    return list(adata.var_names[adata.var["kind"].values == "fluor"])


def _resolve_channels(adata: ad.AnnData, channels: Sequence[str] | None) -> list[int]:
    if channels is None:
        names = fluor_channels(adata)
    else:
        names = list(channels)
    return [channel_index(adata, n) for n in names]


def _get_matrix(adata: ad.AnnData, layer: str | None) -> np.ndarray:
    return np.asarray(layer_matrix(adata, layer), dtype=np.float64)


# --------------------------------------------------------------------------
# arcsinh
# --------------------------------------------------------------------------
def asinh_transform(
    adata: ad.AnnData,
    cofactor: float | Mapping[str, float] = 150.0,
    *,
    layer: str,
    channels: Sequence[str] | None = None,
    key_added: str | None = "asinh",
    inplace: bool = False,
) -> ad.AnnData:
    """Arcsinh-transform channels: ``asinh(x / cofactor)``.

    Parameters
    ----------
    adata
        Cytometry AnnData (events x channels).
    cofactor
        A single cofactor for every channel, or a mapping from channel name to
        cofactor. Typical values: 5 for mass cytometry, 150 for flow,
        ~3000 for spectral flow. Channels missing from a mapping keep the
        mapping's ``"default"`` entry, or 150.
    channels
        Channels to transform. Defaults to the fluorescence channels; scatter
        and time channels are copied through unchanged.
    layer
        Input layer to read from, by name. ``"raw"`` is what
        :func:`~cytopy.read_fcs` stores untouched; ``"X"`` is the working
        matrix, and ``"comp"`` the compensated one. Required, so a call
        always says which matrix it transformed.
    key_added
        Layer to write. ``None`` overwrites ``adata.X`` instead.
    inplace
        Modify ``adata`` and return it. The default is ``False``: the original
        is left alone and a modified copy is returned, so a call that is not
        assigned to anything cannot quietly change the data underneath you.

    Returns
    -------
    AnnData
        The annotated object, with the transformed matrix in
        ``adata.layers[key_added]`` and the per-channel cofactors in
        ``adata.var['cofactor']`` (``NaN`` for untransformed channels). The
        viewer reads both back to label its axes in raw units.
    """
    adata = adata if inplace else adata.copy()
    idx = _resolve_channels(adata, channels)

    if isinstance(cofactor, Mapping):
        default = float(cofactor.get("default", 150.0))
        cofs = np.full(adata.n_vars, np.nan)
        for j in idx:
            name = str(adata.var_names[j])
            chan = str(adata.var["channel"].iloc[j]) if "channel" in adata.var else name
            mark = str(adata.var["marker"].iloc[j]) if "marker" in adata.var else name
            for key in (name, mark, chan):
                if key in cofactor:
                    cofs[j] = float(cofactor[key])
                    break
            else:
                cofs[j] = default
    else:
        cofs = np.full(adata.n_vars, np.nan)
        cofs[idx] = float(cofactor)

    X = _get_matrix(adata, layer)
    out = X.copy()
    sel = np.asarray(idx, dtype=int)
    out[:, sel] = np.arcsinh(X[:, sel] / cofs[sel][None, :])

    out = out.astype(np.float32, copy=False)
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    adata.var["cofactor"] = cofs
    info = adata.uns.setdefault("cytopy", {})
    info["asinh_layer"] = key_added
    # Per layer as well, so a second transform does not orphan the first.
    info.setdefault("asinh_layers", {})[str(key_added)] = {
        str(n): float(c) for n, c in zip(adata.var_names, cofs) if np.isfinite(c)
    }
    return adata


# --------------------------------------------------------------------------
# logicle
# --------------------------------------------------------------------------
def logicle_transform(
    adata: ad.AnnData,
    *,
    layer: str,
    channels: Sequence[str] | None = None,
    key_added: str | None = "logicle",
    T: float | None = 262144.0,
    M: float = 4.5,
    W: float | None = 0.5,
    A: float = 0.0,
    inplace: bool = False,
) -> ad.AnnData:
    """Logicle (biexponential) transform, per channel, onto a 0..1 display scale.

    The defaults are flowCore's ``logicleTransform`` defaults -- ``T=262144``,
    ``W=0.5``, ``M=4.5``, ``A=0`` -- so an untuned call here and an untuned
    call there put the data on the same scale. ``T=None`` or ``W=None`` swaps
    that parameter for a per-channel fit to the data.

    Parameters
    ----------
    adata
        Cytometry AnnData.
    channels
        Channels to transform. Defaults to :func:`fluor_channels`; scatter and
        time channels are copied through unchanged.
    layer
        Input layer to read from, by name. ``"raw"`` is what
        :func:`~cytopy.read_fcs` stores untouched; ``"X"`` is the working
        matrix, and ``"comp"`` the compensated one. Required, so a call
        always says which matrix it transformed.
    key_added
        Layer to write. ``None`` overwrites ``adata.X`` instead.
    T
        Top of scale: the data value that maps to 1.0. Defaults to 262144, the
        top of an 18-bit range, matching flowCore's ``logicleTransform``. Pass
        ``None`` to use each channel's own maximum instead.
    M
        Total number of decades the scale covers.
    W
        Decades of linearisation around zero. Defaults to 0.5, again matching
        flowCore's ``logicleTransform``. Pass ``None`` to fit it per channel
        against the spread of that channel's negative values -- what
        flowCore's ``estimateLogicle`` does; see
        :meth:`~cytopy.scales.LogicleScale.from_data`.
    A
        Additional decades of negative data shown below zero.
    inplace
        Modify ``adata`` and return it. The default is ``False``: the original
        is left alone and a modified copy is returned, so a call that is not
        assigned to anything cannot quietly change the data underneath you.

    Returns
    -------
    AnnData
        The annotated object, with the transformed matrix in
        ``adata.layers[key_added]`` and the per-channel ``T``/``W``/``M``/``A``
        in ``adata.uns['cytopy']['logicle_params']``, which the viewer reads
        back to label its axes in raw units.
    """
    adata = adata if inplace else adata.copy()
    idx = _resolve_channels(adata, channels)
    X = _get_matrix(adata, layer)
    out = X.copy()
    params: dict[str, dict[str, float]] = {}
    for j in idx:
        col = X[:, j]
        if W is None:
            scale = LogicleScale.from_data(col, T=T, M=M, A=A)
        else:
            top = T if T is not None else float(max(np.max(col), 10.0))
            scale = LogicleScale(T=top, W=W, M=M, A=A)
        out[:, j] = scale.forward(col)
        params[str(adata.var_names[j])] = {"T": scale.T, "W": scale.W, "M": scale.M, "A": scale.A}

    out = out.astype(np.float32, copy=False)
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    info = adata.uns.setdefault("cytopy", {})
    info["logicle_params"] = params
    info["logicle_layer"] = key_added
    # Also by layer: transforming twice -- raw and compensated, say -- used to
    # leave the first layer unlabelled, because there was only ever one slot.
    info.setdefault("logicle_layers", {})[str(key_added)] = params
    return adata


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def subsample(
    adata: ad.AnnData,
    n: int = 200_000,
    *,
    per_sample: bool = False,
    seed: int = 0,
) -> ad.AnnData:
    """Take a random subset of events, for responsive interactive plotting.

    Parameters
    ----------
    adata
        Cytometry AnnData.
    n
        Number of events to keep — in total, or from each sample when
        ``per_sample`` is set.
    per_sample
        Sample ``n`` events from each value of ``adata.obs['sample']``, so a
        small file is not swamped by a large one. Samples with fewer than ``n``
        events are kept whole.
    seed
        Seed for the random draw, so a subset is reproducible.

    Returns
    -------
    AnnData
        A copy holding the selected events in their original order. Returned
        unchanged if it is already small enough and ``per_sample`` is off.
    """
    rng = np.random.default_rng(seed)
    if adata.n_obs <= n and not per_sample:
        return adata
    if per_sample and "sample" in adata.obs:
        keep: list[np.ndarray] = []
        for rows in adata.obs.groupby("sample", observed=True).indices.values():
            rows = np.asarray(rows)
            take = subsample_indices(rows, n, rng=rng)
            keep.append(rows if take is None else take)
        sel = np.sort(np.concatenate(keep))
    else:
        sel = subsample_indices(adata.n_obs, n, rng=rng)
    return adata[sel].copy()
