"""Every public parameter must be documented.

A one-line docstring is enough for a method that overrides an interface
already documented on its base class (the ``Scale`` transforms); anything else
needs a numpydoc ``Parameters`` section naming each argument.
"""

import ast
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "cytopy"

# Methods whose parameters are documented once, on cytopy.scales.Scale.
INHERITED = {"forward", "inverse", "ticks", "limits", "to_raw", "from_raw", "__call__"}


def _documented(doc: str) -> set[str] | None:
    """Names in the docstring's Parameters section, or None if there is none."""
    m = re.search(
        r"Parameters\n\s*-+\n(.*?)(\n\s*(Returns|Notes|Examples|Raises)\n\s*-+|\Z)", doc, re.DOTALL
    )
    if not m:
        return None
    names = set()
    for line in m.group(1).splitlines():
        if not line.strip() or len(line) - len(line.lstrip()) > 8:
            continue  # a description, not a parameter name
        for part in line.split(":", 1)[0].split(","):
            part = part.strip()
            if re.fullmatch(r"\*{0,2}[A-Za-z_]\w*", part):
                names.add(part.lstrip("*"))
    return names


def _public_defs():
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.ClassDef):
                continue
            if node.name.startswith("_"):
                continue
            yield path.name, node


def _params(node) -> list[str]:
    if isinstance(node, ast.ClassDef):
        init = next(
            (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
        )
        if init is None:
            return []
        args = init.args
    else:
        args = node.args
    names = [a.arg for a in args.args + args.kwonlyargs if a.arg not in ("self", "cls")]
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


@pytest.mark.parametrize(
    ("filename", "node"),
    [(f, n) for f, n in _public_defs()],
    ids=lambda v: v if isinstance(v, str) else v.name,
)
def test_public_parameters_are_documented(filename, node):
    doc = ast.get_docstring(node)
    where = f"{filename}:{node.lineno} {node.name}"
    assert doc, f"{where} has no docstring"

    params = _params(node)
    if not params or node.name in INHERITED:
        return
    documented = _documented(doc)
    assert documented is not None, f"{where} has no Parameters section (needs {params})"
    missing = [p for p in params if p not in documented]
    assert not missing, f"{where} does not document {missing}"
