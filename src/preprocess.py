import re
import os
import datetime
import gdal
import pickle
import bz2
import time 
import pandas as pd
from glob import glob
import numpy as np
from src.wood_constants import FileConstants as fc
from src.wood_constants import DataConstants as dc
from tqdm import tqdm
from src.analysis_tools import export_tif

def get_data(f):
    return(pickle.load(bz2.open(f)))
    
def generate_uniqidlist(filelist):
    '''
    Takes in a list of files and finds the uniqids and writes to a 
    master csv

    Parameters:
    -----------
    filelist : <list> of strings
             
    '''
    uniqid = np.array([os.path.splitext(os.path.basename(n))[0].split('_')[1] for n in filelist])
    out = pd.DataFrame(np.array([filelist,uniqid]).T, columns=['filename','UNIQID'])
    return(out)

def generate_y_table(wd='data', state_subset=None, cull=True):
    '''
    Generates the FIA table for a set number of predictor variables,
    limited by the NAIP image uniqids

    Parameters:
    ----------
    wd : <str> data directory where FIA files are stored
    state_subset : <list> of states by two letter <str> abbreviation i.e. "CA"
    '''
    #ds = pd.read_csv('./%s/csv/ALL_NONZERO_METADATA.csv'%wd)
    ds = pd.read_csv('./%s/csv/FINAL_METADATA_MgperHa_v1.0.0.csv'%wd)
    ds.rename(columns={'uniqid': 'CN', 'class':'CLASS'}, inplace=True)
    plots = pd.read_csv('%s/csv/ALL_PLOTS_ACTUAL.csv'%wd)
    if cull:
        misclass = pd.read_csv('%s/csv/MISCLASS_QA.csv'%wd)
        misclimate = pd.read_csv('%s/csv/MISCLIMATE.csv'%wd)
        #plots = pd.read_csv('./%s/csv/ALL_PLOTS_ACTUAL_MgperHa.csv'%wd)
        plots = plots[~plots.CN.isin(misclass.CN)]
        plots = plots[~plots.CN.isin(misclimate.CN)]
    fia = pd.read_csv('%s/csv/PLOT.csv'%wd, low_memory=False)
    plots = plots.merge(fia[['CN','STATECD']],how='left',on='CN')
    #add state code so we can parse by state
    out_ds = plots.merge(ds, how='left', on='CN').fillna(0)
    statelist = '%s/csv/STATECD.csv'%wd
    states = pd.read_csv(statelist)
    out_ds = pd.merge(out_ds, states, how='left', on='STATECD')
    if state_subset is not None:
        out_ds = subset_by_state(out_ds, state_subset)
    out_ds.CLASS = out_ds.CLASS.astype('uint8')
    return(out_ds)

def generate_y_table_testdata(wd='data', varlist=['Total_Live_Bio']):
    '''
    Generates the Sagehen table for a set number of predictor variables,
    limited by the NAIP image uniqids

    Parameters:
    ----------
    wd : <str> data directory where Sagehen files are stored
    '''
    ds = pd.read_csv('./%s/test/sagehen_metadata_withfia.csv'%wd)
    uniqid = [ds['Unnamed: 0'][i].astype(str).zfill(8) for i in range(len(ds))]
    ds['UNIQID'] = uniqid
    varlist.append('UNIQID')
    varlist.append('LAT')
    varlist.append('LON')
    out_ds = ds[varlist]
    return(out_ds)


def subset_by_state(data, state):
    '''
    subsets data by the state code

    Parameters:
    -----------
    data : <pandas DataFrame>
    state : <str list>
    '''
    s = pd.DataFrame(state, columns=['STATEAB'])
    return(data.merge(s, on='STATEAB'))


def subset_by_naip(data, naip_dir='/data/wood-supply/NAIP/images'):
    #generate table based on naip
    naip_files = glob('%s/*.tif'%naip_dir)
    naip = generate_uniqidlist(naip_files) 
    #link naip with plot uniqids
    naip_plots = pd.merge(naip, data, how='right', on='UNIQID').dropna()
    return(naip_plots)

def subset_by_landsat(data):
    #generate table based on naip
    ls_files = glob('./data/training/LANDSAT/*.tif')
    ls = generate_uniqidlist(ls_files) 
    uls = pd.DataFrame(ls.UNIQID.unique(),columns=['UNIQID'])
    ls_plots = pd.merge(uls, data, how='right', on='UNIQID').dropna()
    return(ls_plots)

def parse_date(filename):
    '''
    parses LANDSAT file name for its date

    Parameters:
    -----------
    filename : <str> filename
    '''
    # exclusive for landsat data
    n = os.path.splitext(os.path.basename(filename))[0].split('_')
    year = int(n[2])
    jd = int(n[-1])
    date = (datetime.datetime.strptime('%s-%s'%(year, jd), '%Y-%j')).date()
    return(date)

def ndvi(im):
    off = 1e-6
    return((im[:,:,3]-im[:,:,2])/(im[:,:,3]+im[:,:,2]+off))

def landsat_month_aggregate(df, m):
    '''
    From a single landsat plot sample, aggeregates all the data for a single month
    and calculates the mean by each pixel, considering for clouds, shadows, and snow.
    If no monthly data exists, returns a zero array.

    Parameters:
    -----------
    df : <pandas DataFrame>
    m : <int> month
    '''
    month_ls = df[df.index.month == m]
    #put an exception here if there are no records of the month
    lt_shape = (4,4,7)
    if len(month_ls)==0:
        return(np.zeros(lt_shape))
    lt_df = np.moveaxis(np.array([gdal.Open(f).ReadAsArray() for f in month_ls['filename_ls']]),1,-1)
    qa = lt_df[...,-1]
    mask = ((qa==dc._shadow) | (qa==dc._cloud) | (qa==dc._fill))
    pixel_mask = np.broadcast_to(mask[...,np.newaxis], lt_df[...,:-1].shape)
    masked_lt = np.ma.array(lt_df[...,:-1], mask=pixel_mask, fill_value=0)
    aggr = np.ma.mean(masked_lt/255., axis=0).filled(fill_value=0)
    return(aggr)

def process_landsat_sample(subdf, sampleid):
    '''
    finds all the corresponding landsat filenames of a single fia plot sample
    
    Parameters:
    -----------
    subdf : <pandas DataFrame> Dataframe that has been prefiltered to match for existing NAIp
    sampleid : <str>
    '''
    lt_by_uid = subdf[subdf['UNIQID'] == sampleid]
    date_list = pd.Series([pd.Timestamp(parse_date(f)) for f in lt_by_uid['filename_ls'].values])
    lt_by_uid.index = pd.Series(date_list) 
    return(np.array([landsat_month_aggregate(lt_by_uid, m) for m in range(1,13)]))

def preprocess_landsat(df, landsat_list=None, landsat_dir='/data/wood-supply/LANDSAT/images'):
    '''
    opens a list of landsat files that refer to the same sample location
    and performs quality control and normalization
    
    Parameters:
    df : <pandas DataFrame> to link to 
    landsat_list : <str list>
    
    '''
    #open generate a numpy tensor for each month?
    if landsat_list is None:
        landsat_list = sorted(glob('%s/*.tif'%landsat_dir))
    #create a pandas DataFrame of the landsat list
    ls = generate_uniqidlist(landsat_list)
    subls = pd.merge(ls, df, how='left', on='UNIQID').dropna()
    subls.rename(columns={'filename':'filename_ls'}, inplace=True)
    ulist = subls['UNIQID'].unique()
    n_samples = len(ulist)
    lt_ls = [process_landsat_sample(subls, sample) for sample in tqdm(ulist)]
    return(np.array(lt_ls), ulist)

def write_landsat(df, im_list, landsat_list=None, \
        landsat_dir='/data/wood-supply/LANDSAT/images', out_dir='/data/processed/LANDSAT'):
    '''
    Writes the processed landsat to tif format

    Parameters:
    df : <pandas DataFrame> data that has been filtered 
    im_list : <list> 2D list where the first index are the landsat 
                images in nd.array format. The second index refers to 
                the uniqid of each landsat image stack.
    '''
    if landsat_list is None:
        landsat_list = glob('%s/*.tif'%(landsat_dir))
    ls = generate_uniqidlist(landsat_list)
    ref_list = ls.drop_duplicates(subset='UNIQID',keep='first')
    n_samples = len(im_list[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(n_samples):
        ref = gdal.Open(ref_list['filename'][ref_list['UNIQID']==im_list[1][i]].values[0])
        im = np.reshape(np.moveaxis(im_list[0][i],0,-1), (4,4,84))
        outname = '%s/LT_%s.tif'%(out_dir,im_list[1][i])
        export_tif(im, ref, outname, bands=84)
    return(print('wrote LANDSAT to %s'%out_dir))
    
def unpack_landsat(filename):
    '''
    Transform 84x4x4 band landsat to 12x4x4x7
    
    Parameters:
    -----------
    filename : <str>
    '''
    landsat = gdal.Open(filename).ReadAsArray()
    out = np.moveaxis(np.reshape(np.moveaxis(landsat,0,-1), (4,4,7,12)),-1,0)
    return(out)

def unpack_naip(filename, nbands=None):
    '''
    Unpack NAIP tif file to np.array of dim (120,120,3) 
    and normalizes the data (divides by 255)

    Parameters:
    -----------
    filename : <str>
    '''
    if nbands is not None:
        naip = gdal.Open(filename).ReadAsArray()[:nbands,...]
    else:
        naip = gdal.Open(filename).ReadAsArray()
    out = np.moveaxis(naip, 0, -1)/255.
    return(out)

def mm_norm(data, minimum, maximum):
    '''
    Normalizes data between min and max to be between 0 and 1
    Parameters:
    -----------
    data : <float>
    minimum : <float> 
    maximum : <float> 
    '''
    return((data - minimum)/(maximum - minimum))

def un_norm(data, minimum, maximum):
    return((data*(maximum - minimum)+minimum))

def ne_transform(deg):
    return((np.cos(np.deg2rad(deg))+1)/2)

def ne_to_deg_old(ne):
    # deprecated
    return(np.rad2deg(np.arccos(ne)*2)-1)

def ne_to_deg(ne):
    return(np.rad2deg(np.arccos((2*ne)-1)*2))

def terrain_normalize(terrain):
    '''
    normalizes the terrain between 0 and 1
    Parameter:
    ----------
    terrain : <np.array> of dim (3)
              where band 1:aspect, band 2: elevation, band 3: slope
    '''
    terrain[0] = ne_transform(terrain[0])
    terrain[1] = mm_norm(terrain[1], dc._elevation[0], dc._elevation[1])
    terrain[2] = terrain[2] / dc._slope
    return(terrain)

def unpack_dem(filename):
    '''
    Preprocess the DEM file to an np.array of dim (3)
    and normalizes the data between 0 and 1

    Parameters:
    -----------
    filename : <str>
    '''
    tmp = pd.read_csv(filename)
    bands = ['aspect', 'elevation', 'slope']
    terrain = np.array([tmp['%s_0_0'%(bands[b])] for b in range(len(bands))])
    return(terrain_normalize(terrain).T)

def climate_normalize(climate, bands):
    '''
    normalizes the climate between 0 and 1
    Parameter:
    ----------
    climate : <np.array> of dim (360,7)
    '''
    for i in range(len(bands)):
        climate[:,i]= mm_norm(climate[:,i],\
            dc._climate_cnts[bands[i]][0], dc._climate_cnts[bands[i]][1])
    return(climate)

def aggregate_climate(climate):
    '''
    Aggregates the climate data into the mean and std for the entire 
    time series. 

    Parameters:
    -----------
    climate : <nd.array> of shape (360,7)
    '''
    agg = np.array([np.mean(climate, axis=0), np.std(climate,axis=0)])
    channels = np.shape(climate)[-1]
    return(agg.reshape(2*channels))
    #agg = np.mean(climate, axis=0)
    #channels = np.shape(climate)[-1]
    #return(agg)

def unpack_climate(filename, aggregate=True):
    '''
    Preprocess the CLIMATE filelist to an np.array of dim (360,1,1,7)
    and normalizes the data between 0 and 1

    Parameters:
    -----------
    filelist : <list> with <str>
    '''
    c = pd.read_csv(filename)
    bands = ['ppt', 'tdmean', 'tmax', 'tmean', 'tmin', 'vpdmax', 'vpdmin']
    #bands = ['ppt', 'tmean', 'tmin', 'tmax', 'tdmean', 'vpdmin', 'vpdmax']
    # order above is for v0.9
    climate = np.zeros((len(c), len(bands)))
    for i in range(len(bands)):
        if '%s_0_0'%bands[i] in c.columns:
            climate[:,i] = c['%s_0_0'%bands[i]] 
    out = climate_normalize(climate, bands)
    if aggregate:
        out = aggregate_climate(out)
    return(out)

def climate_qa(filename):
    '''
    QA the CLIMATE filelist to an np.array of dim (360,1,1,7)
    and normalizes the data between 0 and 1

    Parameters:
    -----------
    filelist : <list> with <str>
    '''
    c = pd.read_csv(filename)
    bands = ['ppt', 'tdmean', 'tmax', 'tmean', 'tmin', 'vpdmax', 'vpdmin']
    #bands = ['ppt', 'tmean', 'tmin', 'tmax', 'tdmean', 'vpdmin', 'vpdmax']
    # order above is for v0.9
    for i in range(len(bands)):
        if '%s_0_0'%bands[i] not in c.columns:
            return(filename)

def one_hot_it(arr, n_classes):
    '''
    Performs the one hot array transform on classified data list

    Parameters:
    -----------
    arr : <np.array> n dim array of classified data
    n_classes : <int> specified number of classes to one hot 
    '''
    n = arr.shape[0]
    if n == 1:
        z = np.zeros(n_classes).astype('uint8')
        z[arr] = 1
    else:
        z = np.zeros((n, n_classes)).astype('uint8')
        z[np.arange(n),arr] = 1
    return(z)

def generate_data(xlist, ylist, state_subset=None, naip_dirc='./data/training/NAIP', nbands=3,test=False, cull=True):
    '''
    Generates predictor data for landsat based on data list 

    Parameters:
    -----------
    df : <pandas DataFrame>
    ylist : <list str> of variables from FIA to be assembled
    xlist : <list str> of predictor variables to be assembled
    '''
    df = generate_y_table('data', state_subset=state_subset, cull=cull)
    subdf = subset_by_naip(df, naip_dirc)
    subdf.rename(columns={'filename':'filename_naip'}, inplace=True)
    #need a multiple if statements to determine which 
    if 'landsat' in xlist:
        landsat_list = pd.DataFrame(glob('./data/processed/LANDSAT/*.tif'), columns=['filename'])
        landsat_list['UNIQID'] = [os.path.splitext(i)[0].split('_')[1] for i in landsat_list['filename']]
        subdf = pd.merge(landsat_list, subdf, on='UNIQID', how='inner')
        subdf.rename(columns={'filename':'filename_ls'}, inplace=True)
    # now prep the DEM and CLIMATE data
    demfiles = glob('./data/training/DEM/*.csv')
    # make a dataframe of dem files and their unique ID and link to CN
    dem_df = generate_uniqidlist(demfiles)
    dem_df.rename(columns={'filename':'filename_dem'}, inplace=True)
    subdf = pd.merge(subdf, dem_df,how='left', on='UNIQID')
    climatefiles = glob('./data/training/CLIMATE/*.csv')
    climate_df = generate_uniqidlist(climatefiles)
    climate_df.rename(columns={'filename':'filename_climate'}, inplace=True) 
    subdf = pd.merge(subdf, climate_df,how='left', on='UNIQID')
    X = []
    for v in xlist:
        if 'naip' in v:
            print('unpacking NAIP')
            time.sleep(1)
            X.append(np.array([unpack_naip(f, nbands=nbands) for f in tqdm(subdf['filename_naip'])]))
        if 'landsat' in v:
            print('unpacking LANDSAT')
            time.sleep(1)
            X.append(np.array([unpack_landsat(f) for f in tqdm(subdf['filename_ls'])]))
        if 'dem' in v:
            print('unpacking DEM')
            time.sleep(1)
            X.append(np.array([unpack_dem(f) for f in tqdm(subdf['filename_dem'])]))
        if 'climate' in v:
            print('unpacking CLIMATE')
            time.sleep(1)
            X.append(np.expand_dims(np.array([unpack_climate(f) for f in tqdm(subdf['filename_climate'])]), axis=1))
            #X.append(np.expand_dims(np.expand_dims(np.array([unpack_climate(f) for f in tqdm(subdf['filename_climate'])]),axis=2),axis=2))
    y_reg = np.array([subdf[i] for i in ylist]).T
    y_class = one_hot_it(subdf['CLASS'],5)
    y = [y_class, y_reg] 
    points = np.array([(subdf.iloc[i].LON, subdf.iloc[i].LAT, subdf.iloc[i].UNIQID, subdf.iloc[i].CN) for i in range(len(subdf))])
    return(X, y, points)   

def subset_testdata(X, y, pnts, n=500, seed=1234):
    '''
    returns X, y, and pnts, but also subsets out a number of test points for 
    model diagnostics.
    '''
    np.random.seed(seed)
    idx = np.arange(len(X[0]))
    np.random.shuffle(idx)
    Xtest = [X[i][idx[:n]] for i in range(len(X))]
    ytest = [y[i][idx[:n]] for i in range(len(y))]
    pntstest = pnts[idx[:n]]
    Xout = [X[i][idx[n:]] for i in range(len(X))]
    yout = [y[i][idx[n:]] for i in range(len(y))]
    pntsout = pnts[idx[n:]]
    return(Xtest, ytest, pntstest, Xout, yout, pntsout)

def generate_testdata(xlist, ylist, \
        naip_dirc='./data/test/sagehen_plots_training_v0.9.0/NAIP/images', nbands=None):
    '''
    Generates predictor data for landsat based on data list 

    Parameters:
    -----------
    df : <pandas DataFrame>
    ylist : <list str> of variables from Sagehen to be assembled
    xlist : <list str> of predictor variables to be assembled
    '''
    wd = './data/test/sagehen_plots_training_v0.9.0'
    metadatafile = 'sagehen_metadata_withfia.csv' 
    df = pd.read_csv('%s/%s'%(wd,metadatafile))
    uniqid = [df['Unnamed: 0'][i].astype(str).zfill(8) for i in range(len(df))]
    df['UNIQID'] = uniqid
    subdf = subset_by_naip(generate_y_table_testdata('data', ylist), naip_dirc)
    subdf = subset_by_naip(df, naip_dirc)
    subdf.rename(columns={'filename':'filename_naip'}, inplace=True)
    #need a multiple if statements to determine which 
    if 'landsat' in xlist:
        landsat_list = pd.DataFrame(glob('%s/LS/processed/*.tif'%wd), columns=['filename'])
        landsat_list['UNIQID'] = [os.path.splitext(os.path.basename(i))[0].split('_')[1] for i in landsat_list['filename']]
        subdf = pd.merge(landsat_list, subdf, on='UNIQID', how='inner')
        subdf.rename(columns={'filename':'filename_ls'}, inplace=True)
    # now prep the DEM and CLIMATE data
    demfiles = glob('%s/DEM/images/*.csv'%wd)
    # make a dataframe of dem files and their unique ID and link to CN
    dem_df = generate_uniqidlist(demfiles)
    dem_df.rename(columns={'filename':'filename_dem'}, inplace=True)
    subdf = pd.merge(subdf, dem_df,how='left', on='UNIQID')
    climatefiles = glob('%s/CLIMATE/images/*.csv'%wd)
    climate_df = generate_uniqidlist(climatefiles)
    climate_df.rename(columns={'filename':'filename_climate'}, inplace=True) 
    subdf = pd.merge(subdf, climate_df,how='left', on='UNIQID')
    X = []
    for v in xlist:
        if 'naip' in v:
            print('unpacking NAIP')
            time.sleep(1)
            X.append(np.array([unpack_naip(f, nbands=nbands) for f in tqdm(subdf['filename_naip'])]))
        if 'landsat' in v:
            print('unpacking LANDSAT')
            time.sleep(1)
            X.append(np.array([unpack_landsat(f) for f in tqdm(subdf['filename_ls'])]))
        if 'dem' in v:
            print('unpacking DEM')
            time.sleep(1)
            X.append(np.array([unpack_dem(f) for f in tqdm(subdf['filename_dem'])]))
        if 'climate' in v:
            print('unpacking CLIMATE')
            time.sleep(1)
            X.append(np.expand_dims(np.array([unpack_climate(f) for f in tqdm(subdf['filename_climate'])]), axis=1))
    points = np.array([(subdf.iloc[i].LON, subdf.iloc[i].LAT, subdf.iloc[i].UNIQID) for i in range(len(subdf))])
    return(X, subdf, points)   
