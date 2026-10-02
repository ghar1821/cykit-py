import numpy as np
import pytest

from cykit.scales import (
    AsinhScale,
    LinearScale,
    LogicleScale,
    LogScale,
    get_scale,
    pad_range,
)


@pytest.mark.parametrize(
    "scale",
    [LinearScale(), LogScale(), AsinhScale(cofactor=150.0), LogicleScale(T=262144, W=0.5, M=4.5)],
)
def test_roundtrip(scale):
    data = np.array([-1000.0, -10.0, 0.0, 1.0, 100.0, 5000.0, 200000.0])
    if isinstance(scale, LogScale):
        data = data[data > 0]
    out = scale.inverse(scale.forward(data))
    assert np.allclose(out, data, rtol=1e-3, atol=1e-2)


def test_logicle_anchor_points():
    s = LogicleScale(T=262144.0, W=0.5, M=4.5, A=0.0)
    assert s.inverse(np.array(s.x1)) == pytest.approx(0.0, abs=1e-9)
    assert s.inverse(np.array(1.0)) == pytest.approx(262144.0, rel=1e-9)
    assert s.forward(np.array(0.0)) == pytest.approx(s.x1, abs=1e-4)


def test_logicle_is_monotonic():
    s = LogicleScale(T=1e5, W=1.0, M=4.5)
    x = np.linspace(-1e4, 1e5, 20001)
    assert np.all(np.diff(s.forward(x)) >= 0)


def test_logicle_from_data_widens_for_negatives():
    # With the top of scale fixed, a broader negative population must buy more
    # decades of linearisation around zero.
    rng = np.random.default_rng(0)
    tight = rng.normal(0, 10, 10000)
    wide = rng.normal(0, 2000, 10000)
    assert LogicleScale.from_data(wide, T=1e5).W > LogicleScale.from_data(tight, T=1e5).W


def test_logicle_from_data_handles_all_positive_data():
    s = LogicleScale.from_data(np.abs(np.random.default_rng(1).normal(1e3, 10, 5000)))
    assert 0 <= s.W <= s.M / 2


def test_ticks_are_inside_limits_and_labelled():
    s = LogicleScale(T=262144.0, W=0.5, M=4.5)
    lo, hi = s.forward(np.array([-1000.0, 262144.0]))
    t = s.ticks(float(lo), float(hi))
    assert len(t.major) == len(t.labels)
    assert np.all((t.major >= lo) & (t.major <= hi))
    assert "0" in t.labels and any("10" in lbl for lbl in t.labels)


def test_get_scale_dispatch():
    assert isinstance(get_scale("biex", np.array([-5.0, 1e4])), LogicleScale)
    assert get_scale("asinh", cofactor=5).cofactor == 5
    with pytest.raises(ValueError):
        get_scale("nope")


@pytest.mark.parametrize(
    "lo,hi",
    [(0.0, 262144.0), (-5234.2, 267000.0), (1200.0, 98000.0), (0.0, 1023.0), (-0.3, 4.2)],
)
def test_linear_ticks_stay_inside_the_range(lo, hi):
    """A tick past the end of the axis is drawn outside the frame."""
    t = LinearScale().ticks(lo, hi)
    assert np.all((t.major >= lo) & (t.major <= hi))
    assert np.all((t.minor >= lo) & (t.minor <= hi))


@pytest.mark.parametrize("lo,hi", [(0.0, 262144.0), (1200.0, 98000.0), (0.0, 1023.0)])
def test_linear_ticks_are_round_and_plentiful(lo, hi):
    """Round steps, and enough of them to read the axis by."""
    t = LinearScale().ticks(lo, hi)
    assert len(t.major) >= 4
    step = np.diff(t.major)
    assert np.allclose(step, step[0])
    mantissa = step[0] / 10.0 ** np.floor(np.log10(step[0]))
    assert round(float(mantissa), 6) in (1.0, 2.0, 5.0)


def test_linear_tick_labels_are_plain_decimals():
    """Not ``-0``, and not ``2.5e+05`` where ``250000`` fits."""
    # An explicit margin, so this is about the labels and not about
    # whatever AXIS_MARGIN happens to be.
    labels = LinearScale().ticks(*pad_range(0.0, 262144.0, 0.05)).labels
    assert "-0" not in labels
    assert "0" in labels
    assert "250000" in labels
    assert not any("e" in lab for lab in labels)
