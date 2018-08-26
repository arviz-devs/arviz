# pylint: disable=wildcard-import,invalid-name,wrong-import-position
__version__ = '0.1.0'
from matplotlib.pyplot import style

config = {'default_data_directory': '.arviz_data'}

from .inference_data import InferenceData
from .plots import *
from .stats import *
from .utils import trace_to_dataframe, load_data, convert_to_netcdf, load_arviz_data
