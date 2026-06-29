# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from nefusi.membership import (
    N_BREAKPOINTS,
    partition_from_breakpoints,
    scalar_trapmf,
    trapmf,
    trimf,
)


def test_trapmf_corners_and_plateau():
    assert trapmf(0.0, 0.2, 0.4, 0.6, 0.8) == 0.0
    assert trapmf(0.2, 0.2, 0.4, 0.6, 0.8) == 0.0
    assert trapmf(0.3, 0.2, 0.4, 0.6, 0.8) == pytest.approx(0.5)
    assert trapmf(0.5, 0.2, 0.4, 0.6, 0.8) == 1.0  # plateau
    assert trapmf(0.7, 0.2, 0.4, 0.6, 0.8) == pytest.approx(0.5)
    assert trapmf(0.8, 0.2, 0.4, 0.6, 0.8) == 0.0


def test_trapmf_left_and_right_shoulder():
    # left shoulder: full membership at 0
    assert trapmf(0.0, 0.0, 0.0, 0.3, 0.5) == 1.0
    assert trapmf(0.4, 0.0, 0.0, 0.3, 0.5) == pytest.approx(0.5)
    # right shoulder: full membership at 1
    assert trapmf(1.0, 0.5, 0.7, 1.0, 1.0) == 1.0
    assert trapmf(0.6, 0.5, 0.7, 1.0, 1.0) == pytest.approx(0.5)


def test_trapmf_vectorised_matches_scalar():
    xs = np.linspace(0, 1, 11)
    corners = (0.1, 0.3, 0.6, 0.9)
    vec = trapmf(xs, *corners)
    scal = np.array([scalar_trapmf(x, *corners) for x in xs])
    assert np.allclose(vec, scal)


def test_trapmf_in_unit_interval():
    xs = np.linspace(0, 1, 50)
    y = trapmf(xs, 0.0, 0.25, 0.5, 0.75)
    assert np.all((y >= 0) & (y <= 1))


def test_trimf_is_special_trapezoid():
    assert trimf(0.5, 0.0, 0.5, 1.0) == 1.0
    assert trimf(0.25, 0.0, 0.5, 1.0) == pytest.approx(0.5)


def test_partition_from_breakpoints_structure():
    part = partition_from_breakpoints([0.2, 0.4, 0.3, 0.45, 0.55, 0.7, 0.6, 0.8])
    assert set(part) == {"low", "medium", "high"}
    # low is a left shoulder, high is a right shoulder
    assert part["low"][0] == 0.0 and part["low"][1] == 0.0
    assert part["high"][2] == 1.0 and part["high"][3] == 1.0


def test_partition_sorts_within_term():
    # unsorted breakpoints must still give monotone corners
    part = partition_from_breakpoints([0.4, 0.2, 0.7, 0.45, 0.3, 0.55, 0.8, 0.6])
    for corners in part.values():
        assert list(corners) == sorted(corners)


def test_partition_wrong_length_raises():
    with pytest.raises(ValueError):
        partition_from_breakpoints([0.1, 0.2, 0.3])
    assert N_BREAKPOINTS == 8
