"""Reading FCS files into :class:`anndata.AnnData`."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

__all__ = ["concat_samples", "read_fcs", "read_fcs_dir", "split_samples"]

# Channels that are not fluorescence measurements and should be excluded from
# transformation / compensation by default.
_SCATTER_RE = re.compile(r"^(fsc|ssc|bsc)[-_ ]?", re.IGNORECASE)
_TIME_RE = re.compile(r"^time$", re.IGNORECASE)


def _sanitise_var_name(name: str) -> str:
    """Strip characters HDF5 reads as path separators out of a channel name.

    A ``/`` in a var name does not break the index itself -- that is written
    as a plain string array -- but cytopy keys several ``uns`` dicts by
    channel name (``logicle_params``, ``asinh_layers``), and those keys become
    HDF5 names on write, where ``/`` means "subgroup". Writing the object then
    either nests the entry under a group that nothing reads back or fails
    outright. Imaging panels hit this: the S8 writes parameters such as
    ``Delta CoM (SSC (Imaging)/FSC)``.

    The original stays in ``var['channel']`` and ``var['label']``, and
    :func:`~cytopy.transforms.find_channel_name` still resolves against both, so
    looking a channel up by the name the file used keeps working.
    """
    return name.replace("/", "_")


def _channel_kind(pnn: str) -> str:
    if _TIME_RE.match(pnn.strip()):
        return "time"
    if _SCATTER_RE.match(pnn.strip()):
        return "scatter"
    return "fluor"


def _linearise(x: np.ndarray, f1: float, f2: float, pnr: float) -> np.ndarray:
    """Undo log amplification stored via ``$PnE`` (mostly FCS2.0 files)."""
    return f2 * np.power(10.0, f1 * x / pnr)


def _build_var(flow_data, channel_count: int) -> pd.DataFrame:
    """Pivot flowio's per-channel metadata into one row per channel.

    flowio parses the flat ``$Pn*`` TEXT keywords (``$P1N``, ``$P1S``, ...)
    into ``flow_data.channels`` for us, already including ``$PnR``, ``$PnE``
    and ``$PnG`` alongside ``$PnN``/``$PnS`` -- this just transposes that into
    a table that can be ``adata.var``.

    The ``label`` and ``kind`` columns are cytopy's, not the file's: ``label``
    is the marker-and-detector name used for the var index and the axis titles,
    ``kind`` is the scatter/fluor/time split that decides which channels
    transforms and compensation touch by default.
    """
    rows = []
    for i in range(1, channel_count + 1):
        chan = flow_data.channels[i]
        pnn = str(chan["pnn"]).strip()
        pns = str(chan["pns"]).strip()
        pne_decades, pne_offset = chan["pne"]
        rows.append(
            {
                "channel": pnn,
                "marker": pns,
                "label": f"{pns} ({pnn})" if pns and pns != pnn else pnn,
                "kind": _channel_kind(pnn),
                "pnr": chan["pnr"],
                "png": chan["png"],
                "pne_decades": pne_decades,
                "pne_offset": pne_offset,
            }
        )
    var = pd.DataFrame(rows)
    # Index on the human-readable label; de-duplicate defensively.
    # Sanitise before de-duplicating, so two labels that differ only in a
    # character that sanitising removes still end up with distinct names.
    labels = [_sanitise_var_name(lab) for lab in var["label"]]
    seen: dict[str, int] = {}
    unique = []
    for lab in labels:
        if lab in seen:
            seen[lab] += 1
            unique.append(f"{lab}.{seen[lab]}")
        else:
            seen[lab] = 0
            unique.append(lab)
    var.index = pd.Index(unique, name="var_names")
    return var


def _spillover_from_text(
    text: dict, var: pd.DataFrame, *, percent: bool = False
) -> pd.DataFrame | None:
    """Parse ``$SPILLOVER`` / ``$SPILL`` into a DataFrame indexed by detector.

    The keyword is a flat comma-separated list: the detector count, then that
    many ``$PnN`` names, then the matrix row by row. Names are kept exactly as
    the file wrote them -- they are checked against the channels the file
    declares, not rewritten to match them.

    A matrix that is malformed is dropped with a warning rather than raised on:
    the events are still worth having. One that parses but does not look like a
    spillover matrix -- anything but ``1`` down the diagonal -- is kept as it
    was found, with a warning. It is not inverted to make it one; deciding what
    to do with it is :func:`~cytopy.compensate`'s job.
    """
    lowered = {k.lower().lstrip("$"): v for k, v in text.items()}
    raw = lowered.get("spillover") or lowered.get("spill")
    if not raw:
        return None

    parts = [p.strip().strip('"') for p in str(raw).split(",")]
    try:
        n = int(parts[0])
    except (ValueError, IndexError):
        warnings.warn(
            f"$SPILLOVER does not start with a detector count, ignoring it: {raw[:60]!r}",
            stacklevel=3,
        )
        return None
    if n <= 0 or len(parts) < 1 + n + n * n:
        warnings.warn(
            f"$SPILLOVER claims {n} detectors but holds {len(parts) - 1} further "
            f"fields, not {n + n * n}; ignoring it",
            stacklevel=3,
        )
        return None

    names = parts[1 : 1 + n]
    try:
        values = np.asarray(parts[1 + n : 1 + n + n * n], dtype=float).reshape(n, n)
    except ValueError:
        warnings.warn("$SPILLOVER holds non-numeric values, ignoring it", stacklevel=3)
        return None

    if percent:
        values = values / 100.0

    known = set(var["channel"]) | set(var.index) | set(var["marker"])
    unknown = [name for name in names if name not in known]
    if unknown:
        warnings.warn(
            f"$SPILLOVER names {unknown}, which are not channels of this file; "
            "the matrix is stored as written, so compensating with it will fail "
            "until the names are corrected",
            stacklevel=3,
        )

    diagonal = np.diag(values)
    if not np.allclose(diagonal, 1.0):
        warnings.warn(
            f"$SPILLOVER has {np.array2string(diagonal, precision=4)} on the diagonal, "
            "not 1, so it is not a spillover matrix as written -- it may be a "
            "compensation matrix, or be stored as percentages (pass "
            "convert_spillover=True). Storing it unchanged.",
            stacklevel=3,
        )

    return pd.DataFrame(values, index=pd.Index(names), columns=pd.Index(names))


def read_fcs(
    path: str | os.PathLike,
    *,
    sample_id: str | None = None,
    linearise: bool = True,
    apply_gain: bool = False,
    store_raw: bool = True,
    dtype: str = "float32",
    ignore_offset_error: bool = False,
    convert_spillover: bool = False,
) -> ad.AnnData:
    """Read a single FCS file into an :class:`~anndata.AnnData`.

    Parameters
    ----------
    path
        Path to the ``.fcs`` file.
    sample_id
        Name recorded in ``adata.obs['sample']``. Defaults to the file stem.
    linearise
        Undo ``$PnE`` log amplification, matching flowCore's default
        ``read.FCS(transformation="linearize")``. Older instruments stored the
        output of an analog log amplifier, where the true intensity is
        ``f2 * 10 ** (f1 * value / $PnR)``; leaving that alone would mean
        transforming already-logged values. Modern digital files write
        ``$PnE = 0,0`` and ``$DATATYPE`` is float/double rather than integer,
        so this is a no-op for them either way: like flowCore, it is only
        ever applied to channels with ``$DATATYPE = I``.
    apply_gain
        Divide by ``$PnG`` gain, matching flowCore's
        ``read.FCS(transformation="linearize-with-PnG-scaling")``. Off by
        default: instruments disagree about whether the gain has already
        been applied to the stored values, so applying it blindly silently
        rescales the data. A channel is never both log-linearised and
        gain-divided -- as in flowCore, the two are mutually exclusive per
        channel, so this only affects channels ``linearise`` left alone.
    store_raw
        Keep an untouched copy of the values in ``adata.layers['raw']``, so
        later transforms always have something to go back to. Costs one extra
        copy of the matrix; turn it off for very large files.
    dtype
        dtype of ``adata.X`` and ``adata.layers['raw']``.
    ignore_offset_error
        Read on despite an inconsistent data-segment offset in the header.
        Some instruments write offsets that disagree with the TEXT segment;
        the file is usually still readable, so this is the escape hatch for
        one that would otherwise raise.
    convert_spillover
        Divide ``$SPILLOVER`` by 100 on the way in, for a file that stores its
        coefficients as percentages. Off by default, and never inferred from
        the values: a matrix that is not read the way it was written is wrong
        in a way nothing downstream can detect. A diagonal of 100 rather than
        1 is what such a file looks like, and is warned about.

    Returns
    -------
    AnnData with one observation per event and one variable per channel.
    ``.X`` and ``.layers['raw']`` both start out as the values read from the
    file; ``.X`` is the working matrix, ``layers['raw']`` stays put.
    ``.var`` holds the ``$Pn*`` channel metadata, ``.uns['fcs']`` the header
    keywords and, when present, ``.uns['spillover']``.
    """
    import flowio

    path = Path(path)
    fd = flowio.FlowData(str(path), ignore_offset_error=ignore_offset_error)

    n_ch = int(fd.channel_count)
    X = fd.as_array(preprocess=False)
    var = _build_var(fd, n_ch)

    is_integer_stored = fd.data_type.lower() == "i"
    for j, (_, row) in enumerate(var.iterrows()):
        if linearise and is_integer_stored and row["pne_decades"] > 0:
            X[:, j] = _linearise(X[:, j], row["pne_decades"], row["pne_offset"], row["pnr"])
        elif apply_gain and row["png"] > 0 and row["png"] != 1:
            X[:, j] = X[:, j] / row["png"]

    name = sample_id or path.stem
    obs = pd.DataFrame(
        {"sample": pd.Categorical([name] * X.shape[0]), "file": [str(path)] * X.shape[0]},
        index=pd.Index([f"{name}_{i}" for i in range(X.shape[0])], name="obs_names"),
    )

    adata = ad.AnnData(X=np.ascontiguousarray(X, dtype=dtype), obs=obs, var=var)
    if store_raw:
        adata.layers["raw"] = adata.X.copy()
    adata.uns["fcs"] = {str(k): str(v) for k, v in fd.text.items()}
    spill = _spillover_from_text(fd.text, var, percent=convert_spillover)
    if spill is not None:
        adata.uns["spillover"] = spill
    timestep = adata.uns["fcs"].get("$TIMESTEP") or adata.uns["fcs"].get("timestep")
    if timestep:
        try:
            adata.uns["timestep"] = float(timestep)
        except ValueError:
            pass
    return adata


def read_fcs_dir(
    directory: str | os.PathLike,
    *,
    pattern: str = "*.fcs",
    recursive: bool = False,
    **kwargs,
) -> ad.AnnData:
    """Read every FCS file in a directory and concatenate them into one object.

    Parameters
    ----------
    directory
        Directory to search.
    pattern
        Glob matched against the file names.
    recursive
        Search sub-directories too.
    **kwargs
        Passed through to :func:`read_fcs` for every file.

    Returns
    -------
    AnnData
        All events stacked, in sorted file-name order, with
        ``adata.obs['sample']`` naming the file each event came from.

    Raises
    ------
    FileNotFoundError
        If nothing matches ``pattern``.
    """
    directory = Path(directory)
    globber = directory.rglob if recursive else directory.glob
    paths = sorted(globber(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")
    return concat_samples([read_fcs(p, **kwargs) for p in paths])


def concat_samples(adatas: Sequence[ad.AnnData] | Iterable[ad.AnnData]) -> ad.AnnData:
    """Stack per-sample AnnData objects on the channels they share.

    Parameters
    ----------
    adatas
        One AnnData per sample, as returned by :func:`read_fcs`. Panels need
        not match: the intersection of the channels is kept, so a channel
        missing from one file is dropped from all. A single object is returned
        untouched.

    Returns
    -------
    AnnData
        The concatenation, with unique ``obs_names`` and ``var``/``uns`` taken
        from the first object that has them.
    """
    adatas = list(adatas)
    if len(adatas) == 1:
        return adatas[0]
    with warnings.catch_warnings():
        # Two files hold the same event names often enough -- the same file
        # twice, or two runs numbered from zero. anndata warns, and the very
        # next line is what it is asking for, so the warning is just noise.
        warnings.filterwarnings("ignore", message=".*Observation names are not unique.*")
        out = ad.concat(adatas, join="inner", index_unique=None, merge="first", uns_merge="first")
    out.obs_names_make_unique()
    return out


def split_samples(adata: ad.AnnData, key: str = "sample") -> dict[str, ad.AnnData]:
    """Split a concatenated object back into one AnnData per sample.

    The other end of :func:`concat_samples`: gate several files together in one
    window, then take them apart again for anything that works file by file --
    deriving a spillover matrix from single-stain controls, say.

    Gates come with them, since they are ``obs`` columns.

    Parameters
    ----------
    adata
        AnnData with a sample column.
    key
        ``obs`` column naming the sample each event came from.

    Returns
    -------
    dict
        Maps sample name to its events, in the order the samples appear.

    Raises
    ------
    KeyError
        If ``key`` is not an ``obs`` column.
    """
    if key not in adata.obs:
        raise KeyError(f"no obs column {key!r}; nothing to split on")
    labels = adata.obs[key].astype(str).to_numpy()
    out: dict[str, ad.AnnData] = {}
    for name in pd.unique(labels):
        out[str(name)] = adata[labels == name].copy()
    return out
