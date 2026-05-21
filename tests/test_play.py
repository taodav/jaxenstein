from play import _fit_display_scale, _parse_resolution, _scaled_size


def test_parse_resolution_accepts_square_and_rectangular_values():
    assert _parse_resolution("256") == (256, 256)
    assert _parse_resolution("160X120") == (160, 120)


def test_fit_display_scale_keeps_requested_scale_when_it_fits():
    assert _fit_display_scale((160, 120), 4, (1440, 900)) == 4.0


def test_fit_display_scale_caps_to_integer_scale_when_window_is_too_large():
    scale = _fit_display_scale((160, 120), 8, (1440, 900))

    assert scale == 6.0
    assert _scaled_size((160, 120), scale) == (960, 720)


def test_fit_display_scale_downscales_when_native_render_is_too_large():
    scale = _fit_display_scale((2000, 1000), 1, (1440, 900))
    width, height = _scaled_size((2000, 1000), scale)

    assert scale < 1.0
    assert width <= 1440 - 96
    assert height <= 900 - 96
