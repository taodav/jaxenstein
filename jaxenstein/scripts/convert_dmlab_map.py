"""Print the JAXenstein ASCII conversion of a DMLab nav_maze .map file."""

from __future__ import annotations

import argparse
from pathlib import Path

from jaxenstein.maps.dmlab import dmlab_map_to_ascii


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_path", type=Path)
    args = parser.parse_args()

    print(dmlab_map_to_ascii(args.map_path.read_text()))


if __name__ == "__main__":
    main()
