# pylint: disable=redefined-outer-name, no-member
import importlib
import arviz as az
from arviz.utils import Dask
import pytest


pytestmark = pytest.mark.skipif(  # pylint: disable=invalid-name
    (importlib.util.find_spec("dask") is None) and not running_on_ci(),
    reason="test requires dask which is not installed",
)

class TestDataDask:

    def test_dask_chunk_group_kwds(self):
        from dask.distributed import Client
        Dask.enable_dask(dask_kwargs={"dask": "parallelized", "output_dtypes": [float]})
        client = Client(threads_per_worker=4, n_workers=2, memory_limit="2GB")
        group_kwargs = {
            'posterior': {'chunks': {'w_dim_0': 2, 'true_w_dim_0' : 2}},
            'posterior_predictive': {'chunks': {'w_dim_0': 2, 'true_w_dim_0': 2}}
        }
        centered_data = az.load_arviz_data("regression10d", **group_kwargs)
        exp = [('chain', (4,)),
               ('draw', (500,)),
               ('true_w_dim_0', (2, 2, 2, 2, 2)),
               ('w_dim_0', (10,))]
        self.assertListEqual(list(centered_data.chunks.items()), exp)
        client.close()

    def test_dask_chunk_group_regex(self):
        from dask.distributed import Client
        Dask.enable_dask(dask_kwargs={"dask": "parallelized", "output_dtypes": [float]})
        client = Client(threads_per_worker=4, n_workers=2, memory_limit="2GB")
        group_kwargs = {
            "posterior.*": {'chunks': {'w_dim_0': 2, 'true_w_dim_0' : 2}}
        }
        centered_data = az.load_arviz_data("regression10d", regex=True, **group_kwargs)
        exp = [('chain', (4,)),
               ('draw', (500,)),
               ('true_w_dim_0', (2, 2, 2, 2, 2)),
               ('w_dim_0', (10,))]
        self.assertListEqual(list(centered_data.chunks.items()), exp)
        client.close()
