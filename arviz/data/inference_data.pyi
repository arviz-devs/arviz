from typing import Optional, List
import xarray as xr

class InferenceData:
    posterior: Optional[xr.Dataset]
    observations: Optional[xr.Dataset]
    constant_data: Optional[xr.Dataset]
    prior: Optional[xr.Dataset]
    prior_predictive: Optional[xr.Dataset]
    posterior_predictive: Optional[xr.Dataset]
    predictions: Optional[xr.Dataset]
    predictions_constant_data: Optional[xr.Dataset]
    def __init__(self, **kwargs): ...
    def __repr__(self) -> str: ...
    def __delattr__(self, group: str) -> None: ...
    def __add__(self, other: "InferenceData"): ...
    @staticmethod
    def from_netcdf(filename: str) -> "InferenceData": ...
    def to_netcdf(
        self, filename: str, compress: bool = True, groups: Optional[List[str]] = None
    ) -> str: ...
    def sel(
        self, inplace: bool = False, chain_prior: bool = False, **kwargs
    ) -> "InferenceData": ...

# Note, should put an overload here, based on the value of `inplace`
def concat(
    *args,
    dim: Optional[str] = None,
    copy: bool = True,
    inplace: bool = False,
    reset_dim: bool = True,
) -> Optional[InferenceData]: ...
