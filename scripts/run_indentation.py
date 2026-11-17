"""Run sphere indentation on the fixed GenTac bench and save sensor-frame outputs."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from gentac.indentation import (
    DEFAULT_HYDROELASTIC_MODULUS_PA,
    DEFAULT_RADIUS_M,
    run,
)


def main() -> None:
    """Parse arguments and run one indentation sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-mm", type=float, default=0.5)
    parser.add_argument("--radius-mm", type=float, default=DEFAULT_RADIUS_M * 1_000)
    parser.add_argument(
        "--hydroelastic-modulus-pa",
        type=float,
        default=DEFAULT_HYDROELASTIC_MODULUS_PA,
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("indent-%Y%m%dT%H%M%S%fZ")
    output = (
        args.out if args.out else Path(__file__).resolve().parents[1] / "output" / run_id
    )
    try:
        report = run(
            output,
            depth_m=args.depth_mm / 1_000,
            steps=args.steps,
            save_every=args.save_every,
            radius_m=args.radius_mm / 1_000,
            hydroelastic_modulus_pa=args.hydroelastic_modulus_pa,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError, NotADirectoryError) as exc:
        parser.error(str(exc))
    print(
        f"{output.resolve()}\n"
        f"Surface down: {1e3 * report['max_surface_down_m']:.3f} mm, "
        f"marker in-plane: {1e3 * report['max_marker_in_plane_m']:.3f} mm, "
        f"surface overlap: {1e3 * report['max_surface_overlap_m']:.3f} mm"
    )


if __name__ == "__main__":
    main()
