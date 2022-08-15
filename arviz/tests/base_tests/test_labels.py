from pytest_cases import fixture_ref
from ..labels import (BaseLabeller,
                    DimCoordLabeller,
                    IdxLabeller,
                    DimIdxLabeller,
                    MapLabeller,
                    NoVarLabeller,
                    NoModelLabeller,
                    mix_labellers)

@pytest.fixture(scope="module")
class Data():
    def __init__(self):
            self.sel= {
                "instrument": "a",
                "experiment": 3,
            }
            self.isel= {
                "instrument": 0,
                "experiment": 4,
            }

class TestLabellers():
    @pytest.fixture(scope="class")
    def __init__(self):
        # the MapLabeller should be initialized with some mappings on all levels
        # if we decide to add NoRepeatLabeller as part of ArviZ it should not be
        # tested here but have dedicated test functions due to its special functionality,
        # for now simply skip it
    
        self.labellers = {"BaseLabeller": BaseLabeller(), 
                          "DimCoordLabeller": DimCoordLabeller(),
                          "IdxLabeller": IdxLabeller(),
                          "DimIdxLabeller": DimIdxLabeller(),
                          "MapLabeller": MapLabeller(),
                          "NoVarLabeller": NoVarLabeller(),
                          "NoModelLabeller": NoModelLabeller(),
                          "mix_labellers": mix_labellers()}

    @pytest.mark.parametrize("args", [("BaseLabeller", "theta\na, 3", Data()), 
                                      ("IDxLabeller", "theta\n0, 4", Data()), ...])
    def test_make_label_vert(self, labeller, args, multidim_sels):
        labeller = self.labellers[labeller]
        label = labeller.make_label_vert("theta", 
                                        multidim_sels.sel, 
                                        multidim_sels.isel)
        assert label == args[1]

    @pytest.mark.parametrize(...)
    def test_make_label_flat...
        pass

    @...
    def test_make_pp_label...
        pass


    @...
    def test_make_model_label...
        pass

    @...
    def test_mix_labellers(multidim_sels):
        pass

# possible extra mix_labellers tests


# then maybe also test behaviour when both sel and ìsel are empty dicts, 
# and/or var_name is an empty string.