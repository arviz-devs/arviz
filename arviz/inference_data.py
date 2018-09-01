import netCDF4 as nc
import xarray as xr


class InferenceData():
    """Container for accessing netCDF files using xarray."""

    def __init__(self, *_, **kwargs):
        """Attach to a netcdf file.

        This will inspect the netcdf for the available groups, so that they can be
        later loaded into memory.

        Parameters:
        -----------
        filename : str
            netcdf4 file that contains groups for accessing with xarray.
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
        return 'Inference data with groups:\n\t> {options}'.format(
            options='\n\t> '.join(self._groups)
        )

    @staticmethod
    def from_netcdf(filename):
        groups = {}
        for group in nc.Dataset(filename, mode='r').groups:
            groups[group] = xr.open_dataset(filename, group=group)
        return InferenceData(**groups)

    def to_netcdf(self, filename):
        mode = 'w' # overwrite first, then append
        for group in self._groups:
            data = getattr(self, group)
            data.to_netcdf(filename, mode=mode, group=group)
            data.close()
            mode = 'a'
        return filename
