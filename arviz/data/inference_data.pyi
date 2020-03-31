from typing import Optional, List, overload, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Literal
import xarray as xr

# pylint has some problems with stub files...
# pylint: disable=unused-argument, multiple-statements

class InferenceData:
    posterior: Optional[xr.Dataset]
    posterior_predictive: Optional[xr.Dataset]
    predictions: Optional[xr.Dataset]
    log_likelihood: Optional[xr.Dataset]
    sample_stats: Optional[xr.Dataset]
    observed_data: Optional[xr.Dataset]
    constant_data: Optional[xr.Dataset]
    predictions_constant_data: Optional[xr.Dataset]
    prior: Optional[xr.Dataset]
    prior_predictive: Optional[xr.Dataset]
    sample_stats_prior: Optional[xr.Dataset]
    _warmup_posterior: Optional[xr.Dataset]
    _warmup_posterior_predictive: Optional[xr.Dataset]
    _warmup_predictions: Optional[xr.Dataset]
    _warmup_log_likelihood: Optional[xr.Dataset]
    _warmup_sample_stats: Optional[xr.Dataset]
    def __init__(self, **kwargs): ...
    def __repr__(self) -> str: ...
    def __delattr__(self, group: str) -> None: ...
    def __add__(self, other: "InferenceData"): ...
    @property
    def _groups_all(self) -> List[str]: ...
    @staticmethod
    def from_netcdf(filename: str) -> "InferenceData": ...
    def to_netcdf(
        self,
        filename: str,
        compress: bool = True,
        groups: Optional[List[str]] = None,  # pylint: disable=line-too-long
    ) -> str: ...
    def sel(
        self, inplace: bool = False, chain_prior: bool = False, **kwargs
    ) -> "InferenceData": ...

@overload
def concat(
    *args,
    dim: Optional[str] = None,
    copy: bool = True,
    inplace: "Literal[True]",
    reset_dim: bool = True,
) -> None: ...
@overload
def concat(
    *args,
    dim: Optional[str] = None,
    copy: bool = True,
    inplace: "Literal[False]",
    reset_dim: bool = True,
) -> InferenceData: ...
@overload
def concat(
    ids: Iterable[InferenceData],
    dim: Optional[str] = None,
    *,
    copy: bool = True,
    inplace: "Literal[False]",
    reset_dim: bool = True,
) -> InferenceData: ...
@overload
def concat(
    ids: Iterable[InferenceData],
    dim: Optional[str] = None,
    *,
    copy: bool = True,
    inplace: "Literal[True]",
    reset_dim: bool = True,
) -> None: ...
@overload
def concat(
    ids: Iterable[InferenceData],
    dim: Optional[str] = None,
    *,
    copy: bool = True,
    inplace: bool = False,
    reset_dim: bool = True,
) -> Optional[InferenceData]: ...
