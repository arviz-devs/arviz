import netCDF4 as nc
import xarray as xr


class InferenceData():
    """Container for accessing netCDF files using xarray."""

    def __init__(self, filename):
        """Attach to a netcdf file.

        This will inspect the netcdf for the available groups, so that they can be
        later loaded into memory.

        Parameters:
        -----------
        filename : str
            netcdf4 file that contains groups for accessing with xarray.
        """
        if filename == '':  # netcdf freezes in this case
            raise FileNotFoundError("No such file b''")
        self._filename = filename
        self._nc_dataset = nc.Dataset(filename, mode='r')
        self._groups = self._nc_dataset.groups

    def __repr__(self):
        return 'Inference data from "{filename}" with groups:\n\t> {options}'.format(
            filename=self._filename,
            options='\n\t> '.join(self._groups)
        )

    def __getattr__(self, name):
        """Lazy load xarray DataSets when they are requested"""
        if name in self._groups:
            setattr(self, name, xr.open_dataset(self._filename, group=name))
            return getattr(self, name)
        return self.__getattribute__(name)

    def __dir__(self):
        """Enable tab-completion in iPython and Jupyter environments"""
        return super(InferenceData, self).__dir__() + list(self._groups.keys())
