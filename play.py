"""Run the packaged JAXenstein interactive player."""

from jaxenstein.play import (
    _fit_display_scale,
    _parse_resolution,
    _scaled_size,
    main,
)


__all__ = ["_fit_display_scale", "_parse_resolution", "_scaled_size", "main"]


if __name__ == "__main__":
    main()
