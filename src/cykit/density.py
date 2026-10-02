"""2-D density maps for cytometry scatter plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Axes2D", "density_curve", "density_image"]


@dataclass
class Axes2D:
    """Mapping between display coordinates and napari image pixel coordinates.

    napari draws images with row 0 at the top and rows increasing downwards, so
    the y axis is flipped relative to a conventional plot. This class owns that
    flip: everything outside it works either in *display* coordinates (the
    values being plotted) or in *pixel* coordinates (napari world coordinates,
    ``(row, col)``).

    Parameters
    ----------
    x_lo, x_hi
        Display-coordinate range spanned by the horizontal axis.
    y_lo, y_hi
        Display-coordinate range spanned by the vertical axis.
    bins
        Number of histogram bins per axis; the image is ``bins`` x ``bins``.
    """

    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    bins: int = 512

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the density image, ``(rows, cols)``."""
        return (self.bins, self.bins)

    def to_pixels(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Convert display coordinates to napari pixel coordinates.

        Parameters
        ----------
        x, y
            Display coordinates, each of length ``n``. Values outside the axis
            range map outside ``[0, bins)``; they are not clipped.

        Returns
        -------
        ndarray
            ``(n, 2)`` array of ``(row, col)`` pixel coordinates.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = (x - self.x_lo) / (self.x_hi - self.x_lo) * self.bins - 0.5
        row = (self.y_hi - y) / (self.y_hi - self.y_lo) * self.bins - 0.5
        return np.column_stack([row, col])

    def to_display(self, rc: np.ndarray) -> np.ndarray:
        """Convert napari pixel coordinates back to display coordinates.

        The inverse of :meth:`to_pixels`; this is how a polygon drawn on the
        canvas is read back in the units of the data being plotted.

        Parameters
        ----------
        rc
            ``(n, 2)`` array of ``(row, col)`` pixel coordinates. A single
            ``(row, col)`` pair is accepted and promoted to ``(1, 2)``.

        Returns
        -------
        ndarray
            ``(n, 2)`` array of display ``(x, y)`` coordinates.
        """
        rc = np.atleast_2d(np.asarray(rc, dtype=float))
        row, col = rc[:, 0], rc[:, 1]
        x = (col + 0.5) / self.bins * (self.x_hi - self.x_lo) + self.x_lo
        y = self.y_hi - (row + 0.5) / self.bins * (self.y_hi - self.y_lo)
        return np.column_stack([x, y])


def density_image(
    x: np.ndarray,
    y: np.ndarray,
    axes: Axes2D,
    *,
    smooth: float = 1.0,
    log: bool = True,
) -> np.ndarray:
    """2-D histogram of display-space points, oriented for a napari image layer.

    Parameters
    ----------
    x, y
        Event coordinates, already in display space.
    axes
        Extent and resolution of the resulting image.
    smooth
        Gaussian smoothing sigma, in bins. ``0`` disables it.
    log
        Return ``log1p(count)``, which is what makes rare populations visible.

    Returns
    -------
    ndarray
        ``(bins, bins)`` float32 image, row 0 at the top of the plot. Events
        outside ``axes`` are dropped, as are non-finite ones.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bins = axes.bins
    span_x = axes.x_hi - axes.x_lo
    span_y = axes.y_hi - axes.y_lo
    if not (
        np.isfinite([axes.x_lo, axes.x_hi, axes.y_lo, axes.y_hi]).all()
        and span_x > 0
        and span_y > 0
    ):
        return np.zeros((bins, bins), dtype=np.float32)

    # Bin by arithmetic rather than with np.histogram2d. The result is
    # identical, but the generic histogram path costs an order of magnitude
    # more, and this runs on every widget change with every event in the file.
    # The range test also drops NaN and +/-inf, since those compare False.
    inside = (x >= axes.x_lo) & (x <= axes.x_hi) & (y >= axes.y_lo) & (y <= axes.y_hi)
    col = np.floor((x[inside] - axes.x_lo) * (bins / span_x)).astype(np.intp)
    row = np.floor((y[inside] - axes.y_lo) * (bins / span_y)).astype(np.intp)
    # An event exactly on the upper edge belongs to the last bin, as it does
    # for np.histogram2d with an explicit range.
    np.clip(col, 0, bins - 1, out=col)
    np.clip(row, 0, bins - 1, out=row)
    H = np.bincount(row * bins + col, minlength=bins * bins).reshape(bins, bins)
    # Row 0 is the top of the plot, so flip: higher y goes to a lower row.
    H = H[::-1].astype(np.float64)
    if smooth and smooth > 0:
        from scipy.ndimage import gaussian_filter

        H = gaussian_filter(H, sigma=float(smooth))
    if log:
        H = np.log1p(H)
    return H.astype(np.float32)


def density_curve(
    values: np.ndarray,
    lo: float,
    hi: float,
    bins: int,
    *,
    smooth: float = 2.0,
    area: bool = True,
) -> np.ndarray:
    """1-D smoothed distribution of ``values`` over ``[lo, hi]``.

    A histogram with a Gaussian applied to it, which is a kernel density
    estimate evaluated on a grid. Doing it that way rather than with
    :class:`scipy.stats.gaussian_kde` keeps it linear in the number of events
    instead of quadratic, so it stays usable on a run of millions.

    Parameters
    ----------
    values
        Event values, already in display space.
    lo, hi
        Range to bin over. Values outside it are dropped.
    bins
        Number of bins, i.e. the resolution of the returned curve.
    smooth
        Gaussian bandwidth, in bins. ``0`` returns the raw histogram.
    area
        Scale the curve to unit area, so distributions from samples of
        different sizes can be compared. ``False`` leaves it as counts.

    Returns
    -------
    ndarray
        ``(bins,)`` float64 curve, all zeros when nothing falls in range.
    """
    values = np.asarray(values, dtype=np.float64)
    span = hi - lo
    if bins <= 0 or not np.isfinite([lo, hi]).all() or span <= 0:
        return np.zeros(max(bins, 0), dtype=np.float64)

    inside = (values >= lo) & (values <= hi)
    index = np.floor((values[inside] - lo) * (bins / span)).astype(np.intp)
    np.clip(index, 0, bins - 1, out=index)
    curve = np.bincount(index, minlength=bins).astype(np.float64)

    if smooth and smooth > 0:
        from scipy.ndimage import gaussian_filter1d

        curve = gaussian_filter1d(curve, sigma=float(smooth), mode="nearest")
    if area:
        total = curve.sum() * (span / bins)
        if total > 0:
            curve = curve / total
    return curve
