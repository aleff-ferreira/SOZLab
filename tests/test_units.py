from engine.units import nm_to_unit, to_nm


def test_units_roundtrip():
    assert to_nm(10.0, "A") == 1.0
    assert nm_to_unit(1.0, "A") == 10.0
    assert to_nm(1.5, "nm") == 1.5
