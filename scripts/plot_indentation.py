"""Plot a saved GenTac fixed-bench run."""

import argparse
from pathlib import Path

from gentac.visualization import plot_run


def main() -> None:
    """Parse arguments and render the saved run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing state.npz")
    parser.add_argument("--frame", type=int, default=-1, help="Sample index (default: last)")
    parser.add_argument("--texture", type=Path, help="Image mapped onto the rest contact face")
    args = parser.parse_args()
    try:
        output = plot_run(args.run_dir, frame=args.frame, texture=args.texture)
    except (OSError, ValueError, IndexError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(output.resolve())


if __name__ == "__main__":
    main()
