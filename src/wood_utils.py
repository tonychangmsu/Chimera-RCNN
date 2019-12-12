import numpy as np
import datetime as datetime
import gdal
import pandas as pd
from glob import glob
import os
import re
from sklearn.utils.class_weight import compute_class_weight
from src.base_classes import FieldData, TSData
from src.wood_constants import FileConstants as fc
from src.wood_constants import DataConstants as dc

'''
All GeospatialData will be stored in a <dict> with 
keys ['id': <int>, 'centroid': <tuple>, 'data': <GeospatialData>]
'''
def get_class_weight(y_train):
    class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
    return(class_weight)

def one_hot_it(arr, n_classes):
    '''
    val : <np.array>
    '''
    n = arr.shape[0]
    if n == 1:
        z = np.zeros(n_classes).astype('uint8')
        z[arr] = 1
    else:
        z = np.zeros((n, n_classes)).astype('uint8')
        z[np.arange(n),arr] = 1
    return(z)

def get_fia_data(datafile=None, zeros=False):
    '''
    FIA classes are:
    {
    0 : No tree,
    1 : Conifer - no dead,
    2 : Decidous - no dead,
    3 : Mixed - no dead,
    4 : Conifer - >80% dead,
    5 : Decidous - >80% dead,
    6 : Mixed - >80% dead
    }
    '''
    if datafile is None:
        datafile = fc._master
    fia = pd.read_csv(datafile)
    fia = fia[['UNIQID', 'INVYR', 'PLOT', 'LAT', 'LON', 'Class', 'LS_Biomass','DS_Biomass']]
    return(fia)

def crop_img(img, outsize=(60,60)):
    ycentroid = int(img.shape[0]//2)
    xcentroid = int(img.shape[1]//2)
    cropped_img = img[ycentroid-(outsize[0]//2):ycentroid+(outsize[0]//2), \
    xcentroid-(outsize[1]//2):xcentroid+(outsize[1]//2), :]
    return(cropped_img)

def get_image_data(filename, imsize=None):
    im = np.moveaxis(gdal.Open(filename).ReadAsArray(),\
            0, -1)
    if imsize is not None:
        im = crop_img(im, imsize)
    return(im)

def get_files_for_id(id_index, data_format=None):
    #returns a list of files for a corresponding index number
    if data_format == 'landsat':
        #go into the LANDSAT folder
        data_dirc = fc._landsat_wd 
        full_list = sorted(glob('%s/*.tif'%data_dirc))
        data_list = [f for f in full_list if \
                (int(re.findall(r"\D(\d{8})\D",f)[0]) == id_index)]

    elif data_format == 'climate':
        #go into the DEM_CLIMATE folder
        data_dirc = fc._climate_wd
        full_list = sorted(glob('%s/*CLIMATE*.csv'%data_dirc))
        data_list = [f for f in full_list if \
                (int(re.findall(r"\D(\d{8})\D",f)[0]) == id_index)]
        
    elif data_format == 'terrain':
        data_dirc = fc._dem_wd
        full_list = sorted(glob('%s/*DEM*.csv'%data_dirc))
        data_list = [f for f in full_list if \
                (int(re.findall(r"\D(\d{8})\D",f)[0]) == id_index)]

    elif data_format == 'naip':
        #go to the NAIP folder
        data_dirc = fc._naip_wd
        full_list = sorted(glob('%s/*NAIP*.tif'%data_dirc))
        data_list = [f for f in full_list if \
                (int(re.findall(r"\D(\d{8})\D",f)[0]) == id_index)]
    else:
        return(print('Please specify the data format!'))
    return(data_list)

def terrain_acquire(imageid):
    dem_ls = get_files_for_id(imageid, 'terrain')
    tmp = pd.read_csv(dem_ls[0])
    bands = ['aspect', 'elevation', 'slope']
    terrain = np.zeros(3)
    for b in range(len(bands)):
        terrain[b] = tmp['%s_0_0'%(bands[b])]
    adate = tmp['_IMG_DATE'][0]
    acqdates = [datetime.datetime.strptime(adate, '%Y, %j').date()]
    return(terrain, dem_ls, acqdates)

def climate_acquire(imageid):
    climate_ls = get_files_for_id(imageid, 'climate')
    c = pd.read_csv(climate_ls[0])
    acqdates = [datetime.datetime.strptime(c['_IMG_DATE'][i], '%Y, %j').date() \
            for i in range(len(c))]
    bands = ['ppt', 'tmean', 'tmin', 'tmax', 'tdmean', 'vpdmin', 'vpdmax']
    climate = np.zeros((len(c), len(bands)))
    for i in range(len(bands)):
        if '%s_0_0'%bands[i] in c.columns:
            climate[:,i] = c['%s_0_0'%bands[i]] 
        else:
            climate[:,i] = -9999 #fill with constant 
            break
    return(climate, climate_ls, bands, acqdates)

def parse_date(filename):
    # exclusive for landsat data
    n = os.path.splitext(os.path.basename(filename))[0].split('_')
    year = int(n[2])
    jd = int(n[-1])
    date = (datetime.datetime.strptime('%s-%s'%(year, jd), '%Y-%j')).date()
    return(date)

def landsat_aggregate(images, acqdates, aggregate='monthly'):
    output = []
    if aggregate=='monthly':
        for i in range(1,13):
            subset = images[acqdates.month==i][...,:-1]
            qa =  images[acqdates.month==i][...,-1] #just the last channel
            m = ((qa==dc._shadow) | (qa==dc._cloud) | (qa==dc._fill))
            pixel_mask = np.broadcast_to(m[...,np.newaxis], subset.shape)
            masked = np.ma.array(subset, mask=pixel_mask)
            aggr = np.ma.median(masked, axis=0)
            output.append(aggr)
    output = np.array(output)
    return(output)

def landsat_acquire(data_list, aggregation=True):
    #for now assume this is just landsat
    images = []
    acqdates = []
    for i in range(len(data_list)):
        filename = data_list[i]
        images.append(get_image_data(filename))
        acqdates.append(parse_date(filename))
    images = np.array(images)
    acqdates = np.array(acqdates)
    s_idx = np.argsort(acqdates)
    images = images[s_idx]
    acqdates = pd.DatetimeIndex(acqdates[s_idx])
    #here we can aggregate the data appropriately
    #let's state the month method of aggregation.
    if aggregation:
        images = landsat_aggregate(images, acqdates, \
                aggregate='monthly')
    return(images, acqdates)

def time_independent_acquire(data_list):
    # only one item in the data_list
    filename = data_list[0]
    image = get_image_data(filename, imsize=(120,120))[:,:,:3]
    return(image)

def populate_field_data(imageid, datafile=None):
    fia = get_fia_data()
    fielddata = FieldData(imageid, fia.loc[imageid,'INVYR'], \
            fia.loc[imageid,'PLOT'], fia.loc[imageid,'LAT'],
            fia.loc[imageid,'LON'],int(fia.loc[imageid, 'Class']), \
            fia.loc[imageid,'LS_Biomass']/dc._bm_scale, \
            fia.loc[imageid,'DS_Biomass']/dc._bm_scale)
    return(fielddata)

def populate_naip_data(imageid):
    naip_ls = get_files_for_id(imageid, 'naip')
    naip_im = time_independent_acquire(naip_ls)
    naipdata = TSData(filenames=naip_ls, acqdates=None, \
            bands=['R','G', 'B'], \
            images=naip_im, units=['brightness', 'brightness',\
            'brightness'])
    return(naipdata)

def populate_landsat_data(imageid):
    landsat_ls = get_files_for_id(imageid, 'landsat')
    landsat_im, landsat_ts = landsat_acquire(landsat_ls)
    landsatdata = TSData(filenames=landsat_ls, acqdates=landsat_ts,\
            bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B8', 'QA'],\
            images=landsat_im, units=['TOA', 'TOA','TOA','TOA','TOA',\
            'TOA','brightness','DN'])
    return(landsatdata)

def populate_terrain_data(imageid):
    terrain, dem_ls, acqdates = terrain_acquire(imageid)
    terraindata = TSData(filenames=dem_ls, acqdates=acqdates,\
            bands=['aspect', 'elevation', 'slope'], \
            images=terrain, units=['degrees', 'meters', 'degrees'])
    return(terraindata)

def populate_climate_data(imageid):
    climate, climate_ls, bands, acqdates = climate_acquire(imageid)
    climatedata = TSData(filenames=climate_ls, acqdates=acqdates,\
            bands=bands, images=climate, units=['mm', 'deg C', 'deg C', 'deg C',\
            'deg C', 'hPa', 'hPa'])
    return(climatedata)

