"""
Tests for arviz.utils
"""
import pytest
from arviz.utils import _var_names


@pytest.mark.parametrize(
    "var_names_expected", [("mu", ["mu"]), (None, None), (["mu", "tau"], ["mu", "tau"])]
)
def test_var_names(var_names_expected):
    """Test var_name handling"""
    var_names, expected = var_names_expected
    assert _var_names(var_names) == expected
