"""The package's public surface, checked against itself.

`__all__` lists drifted apart once already -- names exported by a module but
never re-exported, and names advertised by the package that no module claimed.
These tests make that a failure rather than something you find by grepping.
"""

import importlib
import pkgutil

import pytest

import cytopy

# Modules whose names are served lazily, because importing them pulls in napari
# and Qt. cytopy.__init__ maps each name to its module by hand for exactly that
# reason.
LAZY_MODULES = {"viewer", "cofactors"}


def _module_names():
    for info in pkgutil.iter_modules(cytopy.__path__):
        if not info.name.startswith("_"):
            yield info.name


@pytest.mark.parametrize("name", sorted(_module_names()))
def test_every_module_export_is_reachable_from_the_package(name):
    module = importlib.import_module(f"cytopy.{name}")
    for export in getattr(module, "__all__", []):
        assert hasattr(cytopy, export), f"cytopy.{name}.__all__ has {export!r}, package does not"
        assert export in cytopy.__all__, f"{export!r} is exported but missing from cytopy.__all__"


def test_every_advertised_name_resolves():
    """Also exercises the lazy __getattr__, which a plain import does not."""
    for name in cytopy.__all__:
        getattr(cytopy, name)


def test_all_is_sorted_and_unique():
    assert cytopy.__all__ == sorted(cytopy.__all__)
    assert len(cytopy.__all__) == len(set(cytopy.__all__))


def test_the_lazy_list_matches_what_the_napari_modules_export():
    """The guard that stops the two drifting apart again."""
    pytest.importorskip("napari")

    advertised = sorted(cytopy._LAZY)
    exported = []
    for name in LAZY_MODULES:
        module = importlib.import_module(f"cytopy.{name}")
        exported += module.__all__
        # ... and each name must be served from the module that defines it.
        for export in module.__all__:
            assert cytopy._LAZY[export] == name
    assert advertised == sorted(exported)


def test_the_removed_names_are_really_gone():
    for name in (
        "view",
        "gate",
        "gate_controls",
        "compensate_controls",
        "spillover_from_controls",
        "normalise_beads",
        "gate_beads",
        "debarcode",
        "plot_beads_over_time",
    ):
        assert not hasattr(cytopy, name), f"{name} should have been removed"


def test_importing_cytopy_does_not_import_napari():
    """The lazy names exist so a headless script never pays for Qt."""
    import subprocess
    import sys

    code = "import sys, cytopy; print('napari' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
