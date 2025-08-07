from duration import parse_duration


def test_duration_seconds_suffix():
    assert parse_duration("60s") == 60.0


def test_duration_minutes_suffix():
    assert parse_duration("1m") == 60.0


def test_duration_no_suffix():
    assert parse_duration("15") == 15.0
