"""Data structure for using netcdf groups with xarray."""
import netCDF4 as nc
import xarray as xr


class InferenceData():
    """Container for accessing netCDF files using xarray."""

    def __init__(self, *_, **kwargs):
        """Initialize InferenceData object from keyword xarray datasets.

        Examples
        --------
        InferenceData(posterior=posterior, prior=prior)

        Parameters
        ----------
        kwargs :
            Keyword arguments of xarray datasets
        """
        self._groups = []
        for key, dataset in kwargs.items():
            if dataset is None:
                continue
            elif not isinstance(dataset, xr.Dataset):
                raise ValueError('Arguments to InferenceData must be xarray Datasets '
                                 '(argument "{}" was type "{}")'.format(key, type(dataset)))
            setattr(self, key, dataset)
            self._groups.append(key)

    def __repr__(self):
        """Make string representation of object."""
        return 'Inference data with groups:\n\t> {options}'.format(
            options='\n\t> '.join(self._groups)
        )

    @staticmethod
    def from_netcdf(filename):
        """Initialize object from a netcdf file.

        Expects that the file will have groups, each of which can be loaded by xarray.

        Parameters
        ----------
        filename : str
            location of netcdf file

        Returns
        -------
        InferenceData object
        """
        groups = {}
        with nc.Dataset(filename, mode='r') as data:
            data_groups = list(data.groups)

        for group in data_groups:
            with xr.open_dataset(filename, group=group) as data:
                groups[group] = data
        return InferenceData(**groups)

    def to_netcdf(self, filename, compress=True):
        """Write InferenceData to file using netcdf4.

        Parameters
        ----------
        filename : str
            Location to write to
        compress : bool
            Whether to compress result. Note this saves disk space, but may make
            saving and loading somewhat slower (default: True).

        Returns
        -------
        str
            Location of netcdf file
        """
        mode = 'w' # overwrite first, then append
        for group in self._groups:
            data = getattr(self, group)
            kwargs = {}
            if compress:
                kwargs['encoding'] = {var_name: {'zlib': True} for var_name in data.variables}
            data.to_netcdf(filename, mode=mode, group=group, **kwargs)
            data.close()
            mode = 'a'
        return filename
