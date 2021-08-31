# pylint: disable=redefined-outer-name, no-member
import importlib
import pytest
import dask
import arviz as az
from arviz.utils import Dask


pytestmark = pytest.mark.skipif(  # pylint: disable=invalid-name
    importlib.util.find_spec("dask") is None,
    reason="test requires dask which is not installed",
)

class TestDataDask:

    def test_dask_chunk_group_kwds(self):
        Dask.enable_dask(dask_kwargs={"output_dtypes": [float]})
        with dask.config.set(scheduler="synchronous"):
            group_kwargs = {
                'posterior': {'chunks': {'w_dim_0': 2, 'true_w_dim_0' : 2}},
                'posterior_predictive': {'chunks': {'w_dim_0': 2, 'true_w_dim_0': 2}}
            }
            centered_data = az.load_arviz_data("regression10d", **group_kwargs)
            exp = [('chain', (4,)),
                   ('draw', (500,)),
                   ('true_w_dim_0', (2, 2, 2, 2, 2,)),
                   ('w_dim_0', (2, 2, 2, 2, 2,))]
            res = list(centered_data.posterior.chunks.items())
            res.sort()
            exp.sort()
            assert res ==  exp


    def test_dask_chunk_group_regex(self):
        with dask.config.set(scheduler="synchronous"):
            Dask.enable_dask(dask_kwargs={"output_dtypes": [float]})
            group_kwargs = {
                "posterior.*": {'chunks': {'w_dim_0': 10, 'true_w_dim_0' : 10}}
            }
            centered_data = az.load_arviz_data("regression10d", regex=True, **group_kwargs)
            exp = [('chain', (4,)),
                   ('draw', (500,)),
                   ('true_w_dim_0', (10,)),
                   ('w_dim_0', (10,))]
            res = list(centered_data.posterior.chunks.items())
            res.sort()
            exp.sort()
            assert res == exp
