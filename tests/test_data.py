"""Checks on the private data: skipped on GitHub, run on your laptop."""
import pytest

from config import PROJECT_DIR, all_features, load_data, load_params

P = load_params()
pytestmark = pytest.mark.skipif(not (PROJECT_DIR / P["data_file"]).exists(), reason="private data not available")


def test_all_inputs_present():
    assert set(all_features(P)) <= set(load_data(P).columns)


def test_soil_moisture_is_physically_possible():
    df = load_data(P)
    for d in P["depths"]:
        assert df[d].dropna().between(0, 0.6).all(), d


def test_inputs_have_no_missing_values():
    assert load_data(P)[all_features(P)].isna().sum().sum() == 0
