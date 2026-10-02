"""Small internals shared across modules, with nothing of its own to say.

Kept free of cykit imports so any module can use it without a cycle.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

#: How "use ``adata.X``" is spelled. ``X`` is therefore a reserved layer name:
#: ``adata.layers["X"]`` is unreachable through here, which costs nothing since
#: cykit never creates a layer called that.
LAYER_X = "X"


def layer_matrix(adata: ad.AnnData, layer: str | None):
    """The matrix named by ``layer``.

    Parameters
    ----------
    adata
        The object to read from.
    layer
        Name of a layer. ``"X"``, ``""`` and ``None`` all mean ``adata.X``,
        so this accepts both the stored spelling and a plain argument.

    Returns
    -------
    The matrix, not copied.

    Raises
    ------
    KeyError
        If ``layer`` names a layer that is not there.
    """
    if layer in (None, "", LAYER_X):
        return adata.X
    return adata.layers[layer]


def subsample_indices(population, n: int | None, *, rng) -> np.ndarray | None:
    """Sorted row indices to keep, or ``None`` when nothing needs dropping.

    The one place a subsample is actually drawn. Everything that thins events
    for a plot goes through here, so "which events got plotted" has a single
    answer that a seed controls.

    Parameters
    ----------
    population
        How many rows there are, or the candidate row indices themselves.
    n
        How many to keep. ``None`` keeps everything.
    rng
        A ``numpy.random.Generator``. Consumed in place, so a caller drawing
        for several groups in turn gets a different draw for each.

    Returns
    -------
    ndarray or None
        Indices to keep, ascending. ``None`` when the population already fits.
    """
    rows = None if np.isscalar(population) else np.asarray(population)
    size = int(population) if rows is None else rows.size
    if n is None or size <= n:
        return None
    take = rng.choice(size if rows is None else rows, n, replace=False)
    return np.sort(take)
