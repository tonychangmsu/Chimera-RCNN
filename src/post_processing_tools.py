import re
from glob import glob
import os
import numpy as np
import gdal
from osgeo import osr
from tqdm import tqdm
from keras.models import model_from_json
from keras import optimizers
from src.tsNet_model import rmse, det_coeff
from src.wood_constants import DataConstants as dc
import src.analysis_tools as at
from skimage import filters
import subprocess
import datetime
import matplotlib.pyplot as plt

def model_loader(model_date=datetime.datetime.now().strftime("%m%d%Y")):
    # explore prediction
    # first load the model from keras
    model_json = './output/model/tsnet/tsnet_%s.json'%(model_date)
    model_weight = './output/model/tsnet/tsnet_%s.h5'%(model_date) 
    model = at.import_model(model_json, model_weight)
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=['categorical_crossentropy', rmse],\
                  optimizer=opt, metrics=['accuracy',det_coeff])
    print('Model compiled!')
    return(model)

def import_model(model_json, model_weights):
    '''
    Imports a keras model architecture and 
    associated weights.

    Parameters:
    -----------
    model_json : <str> of keras model in json
    format

    model_weights : <str> of keras model parameters weights
    '''

    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights)
    return(loaded_model)

def write_raster(test_img, y_, offset, stride, n_bands=None,outfilename=None):
    if outfilename is None:
        outfilename = os.path.basename(test_img)
    #y_ = np.expand_dims(y_, axis=-1)
    if n_bands is None:
        n_bands = np.shape(y_)[-1]
    n_rows, n_cols = np.shape(y_)[:2]
    ref = gdal.Open(test_img)
    ref_gt = ref.GetGeoTransform()
    out_proj = osr.SpatialReference()
    out_proj.ImportFromWkt(ref.GetProjectionRef())
    out_gt = [ref_gt[0]+offset[0], stride[0], 0, ref_gt[3]-offset[1], 0, -stride[1]]
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(outfilename, n_cols, n_rows, n_bands, gdal.GDT_Float32)
    ds.SetGeoTransform(out_gt)
    ds.SetProjection(out_proj.ExportToWkt())
    if n_bands==1:
        ds.GetRasterBand(1).WriteArray(y_[:,:])
    else:
        for i in range(n_bands):
            ds.GetRasterBand(i+1).WriteArray(y_[:,:,i])
    ds = None
    print('Wrote %s'%outfilename)

def convert_im(in_im, offset=0, scale=1, bands=4, movebands=True):
    im = gdal.Open(in_im)
    out = im.ReadAsArray(offset, offset, \
    im.RasterXSize-offset, im.RasterYSize-offset)
    if movebands==True:
        out = np.moveaxis(out, 0, -1)
    return(scale*out[:,:,:])

def format_landsat(ds,divs=4, bands=6):
    out = np.zeros((divs,bands))
    for q in range(divs):
        for b in range(bands):
            out[q,b] = ds[q*(bands-1)+b]
    return(out)

def reshape_rwm(array):
    new_shape = [np.shape(array)[0] * np.shape(array)[1]]
    for i in range(2,array.ndim):
        new_shape.append(np.shape(array)[i])
    new_shape = tuple(new_shape)
    return(np.reshape(array, new_shape))

def dem_norm(dem_in):
    #rearranges the order of the DEM so that is it aspect, elevation, slope
    dem_out = np.zeros(np.shape(dem_in))
    dem_out[0] = (np.cos(np.deg2rad(dem_in[2]))+1)/2
    dem_out[1] = (dem_in[0]-dc._elevation[0])/(dc._elevation[1]-dc._elevation[0])
    dem_out[2] = dem_in[1]/dc._slope    
    return(dem_out)

def climate_norm(climate_in):
    climate_out = np.zeros(np.shape(climate_in))
    n_list = [dc._ppt,dc._tmean,dc._tmin,dc._tmax,dc._tdmean,dc._vpdmin,dc._vpdmax]
    for i in range(len(n_list)):
        climate_out[i] = (climate_in[i]-n_list[i][0])/(n_list[i][1]-n_list[i][0])
    return(climate_out)

def climate_shrink(climate_files):
    '''
    Reads in a list of climate files
    Returns a summarized climate into a mean and standard deviation for
    the full time series.
    '''
    climate_df = [climate_norm(gdal.Open(c).ReadAsArray()) for c in tqdm(climate_files)]
    N = len(climate_df)
    #calculate the mean 
    m = np.zeros(np.shape(climate_df[0]))
    for c in climate_df:
        m+=c
    m = m/N
    #calculate the std
    s = np.zeros(np.shape(climate_df[0]))
    for c in climate_df:
        s+= np.sqrt(((1/N*(c-m)**2)))
    agg = np.array([m, s])
    return(np.moveaxis(agg.reshape((m.shape[0]*2, m.shape[1],m.shape[2])),0,-1))

# create a sample list

def construct_data_old(naip, dem, climate, landsat):
    naip_samples = reshape_rwm(naip)
    dem_sample = reshape_rwm(dem)
    dem_samples = np.expand_dims(dem_sample, axis=1)
    climate_sample = reshape_rwm(climate)
    c = np.moveaxis(np.array([np.mean(climate_sample,axis=1),\
        np.std(climate_sample, axis=1)]),0,-1)
    climate_samples = c.reshape((c.shape[0], c.shape[1]*c.shape[2]))
    climate_samples = np.expand_dims(climate_samples,axis=1)
    landsat_samples = reshape_rwm(landsat)
    X_pred = [naip_samples, dem_samples, climate_samples, landsat_samples]
    return(X_pred)

def construct_data(naip, dem, climate):
    naip_samples = reshape_rwm(naip)
    dem_sample = reshape_rwm(dem)
    dem_samples = np.expand_dims(dem_sample, axis=1)
    climate_sample = reshape_rwm(climate)
    climate_samples = np.expand_dims(climate_sample,axis=1)
    X_pred = [naip_samples, dem_samples, climate_samples]
    return(X_pred)


def construct_data_test(naip, dem, landsat):
    naip_samples = reshape_rwm(naip)
    dem_sample = reshape_rwm(dem)
    dem_samples = np.expand_dims(np.expand_dims(dem_sample, axis=1),axis=1)
    landsat_samples = reshape_rwm(landsat)
    X_pred = [naip_samples, dem_samples, landsat_samples]
    return(X_pred)

# unpack data
def unpack_prediction(i, wd='./data/test/sagehen_pred_v1.0.0'):
    naip_list = sorted(glob('%s/*[0-9].tif'%test_dir))
    aux_list = sorted(glob('%s/*[A-Za-z].tif'%test_dir))

def unpack_prediction_old(i,wd='./data/processed/test', landsat=False):
    naip_pdirc = '%s/NAIP'%wd
    dem_pdirc = '%s/DEM'%wd
    climate_pdirc = '%s/CLIMATE'%wd
    naip_list = sorted(glob('%s/*.tif'%naip_pdirc))
    index_list = [int(re.findall(r"\D(\d{8})\D",n)[0]) for n in naip_list]
    ix = index_list[i]
    # get naip 
    naipfile = naip_list[i]
    arr = convert_im(naipfile,offset=0,scale=1/255.,bands=4)
    rolled_naip = at.rolling_window_multichannel(arr, window=(120,120), \
                                             stride=(30,30), channel_last=True, \
                                                 agg=False)
    # get dem
    dem_files = sorted(glob('%s/*.tif'%dem_pdirc))
    dem_list = [d for d in dem_files if '%08d'%ix in d]
    dem = gdal.Open(dem_list[0]).ReadAsArray()
    #rolled_dem = at.rolling_window_multichannel(dem_norm(dem), window=(4,4), stride=(1,1),\
    #                                        channel_last=False, agg=True)
    rolled_dem = at.rolling_window_multichannel(dem, window=(4,4), stride=(1,1),\
                                            channel_last=False, agg=True)
    
    # get climate
    climate_files = sorted(glob('%s/*.tif'%climate_pdirc))
    climate_list = [c for c in climate_files if '%08d'%ix in os.path.basename(c)[:8]]
    #climate_list = [c for c in climate_files if '%08d'%ix in c]
    #this method using os.path.basename(c) gets the climate summary file....
    climate = gdal.Open(climate_list[0]).ReadAsArray()
    #rolled_climate = at.timeseries_rwm(climate_norm(climate), window=(4,4), stride=(1,1),\
    #                               channel_last=False, agg=True)
    rolled_climate = at.rolling_window_multichannel(climate, window=(4,4), stride=(1,1),\
                                            channel_last=False, agg=True)
    
    if landsat:
        landsat_pdirc = '%s/LANDSAT'%wd
        # get landsat
        landsat_files = sorted(glob('%s/*mean*.vrt'%landsat_pdirc))
        landsat_list = [l for l in landsat_files if '%08d'%ix in l]
        landsat = np.array([gdal.Open(l).ReadAsArray() for l in landsat_list])
        rolled_landsat = at.timeseries_rwm(landsat, window=(4,4), stride=(1,1),\
                                       channel_last=False, agg=False)
        X_pred = construct_data_old(rolled_naip, rolled_dem, rolled_climate, rolled_landsat)
    else:
        X_pred = construct_data(rolled_naip, rolled_dem, rolled_climate)
    outshape = (rolled_naip.shape[:2])
    return(X_pred, outshape)

def create_seam_mask(wd, outwd, width):
    im_list = sorted(glob('%s/y_reg*'%wd))
    for i in range(len(im_list)):
        im = gdal.Open(im_list[i])
        img = im.ReadAsArray()
        # generate a mask
        mask = np.zeros(np.shape(img)).astype('bool')
        mask[:width] = True
        mask[-width:] = True
        mask[:,:width] = True 
        mask[:,-width:] = True
        maskfilename = '%s/seammask_%s'%(outwd,os.path.basename(im_list[i]))
        at.export_tif(mask, im, maskfilename, bands=1, dtype=gdal.GDT_Byte)

        # then reproject to WGS84
        fileout = os.path.splitext(os.path.basename(maskfilename))[0]
        reproj_rastername = '%s/%s_WGS84.tif'%(outwd, fileout)
        reproj_cmd = "gdalwarp -t_srs EPSG:4326 %s %s"%(maskfilename, reproj_rastername)
        subprocess.call(reproj_cmd, shell=True)

    maskfilename = '%s/seammask_WGS84.vrt'%outwd
    mosaic_cmd = "gdalbuildvrt -srcnodata 0 %s %s/*WGS84.tif"%(maskfilename, outwd)
    subprocess.check_output(mosaic_cmd, shell=True)
    return(gdal.Open(maskfilename).ReadAsArray().astype('bool'))
   
def seam_fill(im, seammask):
    smoothed = filters.gaussian(im/np.max(im), sigma=3, mode='reflect')*np.max(im)
    im[seammask] = np.max((im[seammask],smoothed[seammask]),axis=0)
    return(im)
