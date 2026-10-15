"""``input.da``: fixed-width fields, validated codes, and a byte-exact round trip."""

import pytest

from mekong_da.inputda import InputDA


def test_published_configuration_renders_to_22_lines():
    text = InputDA().to_text()
    assert len(text.strip().splitlines()) == 22


def test_round_trip(tmp_path):
    original = InputDA(da=0, n_en_member=50, da_stopday=4000)
    path = original.write(tmp_path / "input.da")
    assert InputDA.read(path) == original


def test_f52_fields_are_five_characters_wide():
    """The Fortran reads these under f5.2; a wider field is misread, not rejected."""
    lines = InputDA().to_text().splitlines()
    for line in lines[1:11]:
        assert len(line) == 5, repr(line)


def test_out_of_range_float_is_refused():
    with pytest.raises(ValueError, match="f5.2"):
        InputDA(sm_std1=123.456)


def test_bad_codes_are_refused():
    with pytest.raises(ValueError, match="da must be"):
        InputDA(da=2)
    with pytest.raises(ValueError, match="n_sampling_method"):
        InputDA(n_sampling_method=9)
    with pytest.raises(ValueError, match="da_alg"):
        InputDA(da_alg=3)


def test_short_file_is_refused(tmp_path):
    path = tmp_path / "input.da"
    path.write_text("1\n 0.50\n")
    with pytest.raises(ValueError, match="consumes"):
        InputDA.read(path)


def test_describe_names_the_arm():
    assert "OPEN LOOP" in InputDA(da=0).describe()
    assert "assimilating" in InputDA(da=1).describe()
