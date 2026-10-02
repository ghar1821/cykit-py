import contextlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


@pytest.fixture(scope="session")
def demo_path(tmp_path_factory):
    from make_demo_fcs import main

    path = tmp_path_factory.mktemp("data") / "demo.fcs"
    main(str(path))
    return path


@pytest.fixture
def demo(demo_path):
    import cykit

    return cykit.read_fcs(demo_path)


@pytest.fixture(scope="session")
def controls_dir(tmp_path_factory):
    """A directory of single-stain controls plus an unstained one, no $SPILLOVER."""
    from make_demo_fcs import write_controls

    path = tmp_path_factory.mktemp("controls")
    write_controls(path)
    return path


@pytest.fixture(scope="session")
def true_spillover():
    from make_demo_fcs import spillover_matrix

    return spillover_matrix()


#: Which control file stains which detector. Named here rather than guessed
#: from the file names, which is what callers are expected to do too.
CONTROL_FILES = {
    "CD3 (FITC-A)": "Compensation Controls_FITC-A.fcs",
    "CD19 (PE-A)": "Compensation Controls_PE-A.fcs",
    "CD8 (APC-A)": "Compensation Controls_APC-A.fcs",
}


@pytest.fixture
def controls(controls_dir):
    """``(stained, unstained)``, the mapping `compute_spillover_matrix` takes."""
    import cykit

    stained = {
        detector: cykit.read_fcs(controls_dir / name) for detector, name in CONTROL_FILES.items()
    }
    return stained, cykit.read_fcs(controls_dir / "Unstained.fcs")


@pytest.fixture(autouse=True)
def _close_figures():
    """Plot functions hand back open figures; the caller closes them, so do we."""
    yield
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    plt.close("all")


@pytest.fixture(autouse=True)
def _close_open_napari_window():
    """Shut any window ``open_napari`` or ``open_napari_transform`` left behind.

    It deliberately holds a module-level reference so a non-blocking call in a
    notebook does not let the window be collected. Left in place between tests
    that is a leaked Qt object, which napari's own fixture rightly complains
    about.
    """
    yield
    try:
        from cykit import viewer
    except ImportError:  # pragma: no cover - napari not installed
        return
    from cykit import cofactors, compensation

    for module, attr in (
        (viewer, "_CURRENT"),
        (cofactors, "_CURRENT_TRANSFORM"),
        (compensation, "_CURRENT_COMPENSATION"),
    ):
        current = getattr(module, attr, None)
        if current is not None:
            setattr(module, attr, None)
            with contextlib.suppress(Exception):
                current.viewer.close()
