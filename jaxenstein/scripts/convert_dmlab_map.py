"""Export DMLab geometry and object placements as JSON or geometry-only ASCII."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from jaxenstein.maps.dmlab import dmlab_map_to_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_path", type=Path)
    parser.add_argument(
        "--format",
        choices=("json", "ascii"),
        default="json",
        help="JSON includes object placements; ASCII contains geometry only",
    )
    args = parser.parse_args()

    spec = dmlab_map_to_spec(args.map_path.read_text())
    print(
        json.dumps(asdict(spec), indent=2) if args.format == "json" else spec.ascii_map
    )


if __name__ == "__main__":
    main()
