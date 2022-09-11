import pytest

from ...labels import (
    BaseLabeller,
    DimCoordLabeller,
    DimIdxLabeller,
    IdxLabeller,
    MapLabeller,
    NoModelLabeller,
    NoVarLabeller,
    mix_labellers,
)


class Data:
    def __init__(self):
        self.sel = {
            "instrument": "a",
            "experiment": 3,
        }
        self.isel = {
            "instrument": 0,
            "experiment": 4,
        }


@pytest.fixture
def multidim_sels():
    return Data()


class Labellers:
    def __init__(self):
        self.labellers = {
            "BaseLabeller": BaseLabeller(),
            "DimCoordLabeller": DimCoordLabeller(),
            "IdxLabeller": IdxLabeller(),
            "DimIdxLabeller": DimIdxLabeller(),
            "MapLabeller": MapLabeller(),
            "NoVarLabeller": NoVarLabeller(),
            "NoModelLabeller": NoModelLabeller(),
        }


@pytest.fixture
def labellers():
    return Labellers()


@pytest.mark.parametrize(
    "args",
    [
        ("BaseLabeller", "theta\na, 3"),
    ],
)
class TestLabellers:
    def test_make_label_vert(self, args, multidim_sels, labellers):
        labeller_arg = labellers.labellers[args[0]]
        label = labeller_arg.make_label_vert("theta", multidim_sels.sel, multidim_sels.isel)
        assert label == args[1]
