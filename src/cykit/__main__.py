"""Command line entry point: ``cykit sample.fcs``."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Read the file named on the command line, transform it, and open the viewer.

    Parameters
    ----------
    argv
        Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status; ``0`` once the window has been closed.
    """
    p = argparse.ArgumentParser(prog="cykit", description=__doc__)
    p.add_argument(
        "path",
        type=Path,
        nargs="+",
        help="FCS file, directory of FCS files, or .h5ad; several become samples",
    )
    p.add_argument("-x", "--x-channel", default=None, help="channel to start on")
    p.add_argument("-y", "--y-channel", default=None, help="channel to start on")
    p.add_argument("--cofactor", type=float, default=150.0)
    p.add_argument("--asinh", action="store_true", help="arcsinh transform, then plot that layer")
    p.add_argument("--logicle", action="store_true", help="logicle transform, then plot that layer")
    p.add_argument(
        "--compensate", action="store_true", help="apply the file's own $SPILLOVER matrix"
    )
    p.add_argument("--subsample", type=int, default=0, help="plot at most N events")
    args = p.parse_args(argv)

    import cykit
    from cykit.viewer import as_one_anndata

    adata = as_one_anndata(args.path if len(args.path) > 1 else args.path[0])
    print(adata)

    layer = "raw"
    if args.compensate:
        cykit.compensate(adata, inplace=True)
        layer = "comp"
    if args.asinh and args.logicle:
        p.error("pick one of --asinh / --logicle")
    if args.asinh:
        cykit.asinh_transform(adata, args.cofactor, layer=layer, inplace=True)
        layer = "asinh"
    if args.logicle:
        cykit.logicle_transform(adata, layer=layer, inplace=True)
        layer = "logicle"
    if args.subsample:
        adata = cykit.subsample(adata, args.subsample)

    cykit.open_napari(adata, layer, x=args.x_channel, y=args.y_channel, block=True, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
