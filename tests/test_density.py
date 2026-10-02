"""The 2-D histogram behind every plot.

``density_image`` bins by arithmetic instead of calling ``np.histogram2d``,
because it runs on every widget change with every event in the file and the
generic path costs an order of magnitude more. These tests hold it to giving
exactly what ``np.histogram2d`` would have.
"""

import numpy as np
import pytest

from cykit.density import Axes2D, density_image


def _reference(x, y, axes):
    """What the old np.histogram2d implementation produced."""
    good = np.isfinite(x) & np.isfinite(y)
    counts, _, _ = np.histogram2d(
        x[good],
        y[good],
        bins=axes.bins,
        range=[[axes.x_lo, axes.x_hi], [axes.y_lo, axes.y_hi]],
    )
    return counts.T[::-1]


@pytest.mark.parametrize("bins", [8, 64, 512])
def test_matches_numpy_histogram2d(bins):
    rng = np.random.default_rng(0)
    x = rng.normal(100, 30, 200_000)
    y = rng.normal(-5, 2, 200_000)
    axes = Axes2D(float(x.min()), float(x.max()), float(y.min()), float(y.max()), bins=bins)
    got = density_image(x, y, axes, smooth=0, log=False)
    assert np.array_equal(got, _reference(x, y, axes).astype(np.float32))
    assert got.sum() == len(x)  # the extremes land in the first and last bins


def test_edges_and_non_finite_values_agree_with_numpy():
    axes = Axes2D(0.0, 10.0, 0.0, 10.0, bins=10)
    x = np.array([0.0, 10.0, 5.0, -1.0, 11.0, np.nan, 3.0, np.inf, -np.inf, 2.0])
    y = np.array([0.0, 10.0, 5.0, 5.0, 5.0, 5.0, np.nan, 5.0, 5.0, 2.0])
    got = density_image(x, y, axes, smooth=0, log=False)
    assert np.array_equal(got, _reference(x, y, axes).astype(np.float32))
    assert got.sum() == 4  # (0,0), (10,10), (5,5) and (2,2); the rest are out


def test_events_outside_the_axes_are_dropped_not_clipped():
    axes = Axes2D(0.0, 1.0, 0.0, 1.0, bins=4)
    x = np.array([0.5, 5.0, -5.0])
    y = np.array([0.5, 0.5, 0.5])
    assert density_image(x, y, axes, smooth=0, log=False).sum() == 1


def test_orientation_puts_high_y_at_the_top():
    axes = Axes2D(0.0, 1.0, 0.0, 1.0, bins=4)
    img = density_image(np.array([0.1]), np.array([0.9]), axes, smooth=0, log=False)
    assert img[0, 0] == 1  # top-left: low x, high y
    img = density_image(np.array([0.9]), np.array([0.1]), axes, smooth=0, log=False)
    assert img[-1, -1] == 1


def test_degenerate_axes_give_a_blank_image():
    for axes in (
        Axes2D(1.0, 1.0, 0.0, 1.0, bins=8),
        Axes2D(0.0, 1.0, 5.0, 4.0, bins=8),
        Axes2D(-np.inf, np.inf, 0.0, 1.0, bins=8),
    ):
        img = density_image(np.array([0.5]), np.array([0.5]), axes)
        assert img.shape == (8, 8) and not img.any()


def test_no_events_gives_a_blank_image():
    axes = Axes2D(0.0, 1.0, 0.0, 1.0, bins=8)
    assert not density_image(np.array([]), np.array([]), axes).any()


def test_smoothing_and_log_are_applied():
    axes = Axes2D(0.0, 1.0, 0.0, 1.0, bins=16)
    x = np.full(100, 0.5)
    raw = density_image(x, x, axes, smooth=0, log=False)
    assert raw.max() == 100 and (raw > 0).sum() == 1
    smoothed = density_image(x, x, axes, smooth=2.0, log=False)
    assert smoothed.max() < 100 and (smoothed > 0).sum() > 1
    assert smoothed.sum() == pytest.approx(100, rel=1e-3)  # smoothing conserves counts
    assert density_image(x, x, axes, smooth=0, log=True).max() == pytest.approx(np.log1p(100))
