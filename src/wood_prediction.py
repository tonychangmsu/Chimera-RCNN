import numpy as np
from src.analysis_tools import reshape_rwm
from src.wood_constants import DataConstants as dc

def dem_norm(dem_in):
    dem_out = np.zeros(np.shape(dem_in))
    dem_out[0] = (np.cos(np.deg2rad(dem_in[2]))+1)/2
    dem_out[1] = (dem_in[0]-dc._elevation[0])/(dc._elevation[1]-dc._elevation[0])
    dem_out[2] = dem_in[1]/dc._slope    
    return(dem_out)

def climate_norm(climate_in):
    climate_out = np.zeros(np.shape(climate_in))
    n_list = [dc._ppt,dc._tmean,dc._tmin,dc._tmax, dc._tdmean, dc._vpdmin, dc._vpdmax]
    for i in range(len(n_list)):
        climate_out[i] = (climate_in[i]-n_list[i][0])/(n_list[i][1]-n_list[i][0])
    return(climate_out)

# create a sample list
def construct_data(naip, dem, climate, landsat):
    naip_samples = reshape_rwm(naip)
    dem_sample = reshape_rwm(dem)
    dem_samples = np.expand_dims(np.expand_dims(dem_sample, axis=1),axis=1)
    climate_sample = reshape_rwm(climate)
    climate_samples = np.expand_dims(np.expand_dims(climate_sample, \
                                                         axis=2), axis=2)
    landsat_samples = reshape_rwm(landsat)
    X_pred = [naip_samples, dem_samples, climate_samples, landsat_samples]
    return(X_pred)

def construct_data_test(naip, dem, landsat):
    naip_samples = reshape_rwm(naip)
    dem_sample = reshape_rwm(dem)
    dem_samples = np.expand_dims(np.expand_dims(dem_sample, axis=1),axis=1)
    landsat_samples = reshape_rwm(landsat)
    X_pred = [naip_samples, dem_samples, landsat_samples]
    return(X_pred)
