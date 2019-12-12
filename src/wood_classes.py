import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from glob import glob
import src.wood_utils as ws
from collections import defaultdict
from functools import reduce
from src.wood_constants import DataConstants as dc
from src.wood_constants import FileConstants as fc

class GeospatialData:
    def __init__(self, sampleid: int):
        self.sampleid = sampleid
        self.fia = ws.populate_field_data(sampleid)
        self.landsat = ws.populate_landsat_data(sampleid)
        self.naip = ws.populate_naip_data(sampleid) 
        self.terrain = ws.populate_terrain_data(sampleid)
        self.climate = ws.populate_climate_data(sampleid)
        self.centroid = (self.fia.lon, self.fia.lat)

def search_for_indices():
    #get the naip, landsat, dem, and climate indices
    naip_i = np.array([int(x.split('_')[-10]) for x in \
                       sorted(glob('%s/*.tif'%fc._naip_wd))])
    dem_i = np.array([int(x.split('_')[1][:-4]) for x in \
                      sorted(glob('%s/*.csv'%fc._dem_wd))])
    climate_i = np.array([int(x.split('_')[1][:-4]) for x in \
                          sorted(glob('%s/*.csv'%fc._climate_wd))])
    landsat_i = np.unique(np.array([int(x.split('_')[1]) for x in \
                          sorted(glob('%s/*.tif'%fc._landsat_wd))]))

    indices = reduce(np.intersect1d,\
            (naip_i, dem_i, climate_i, landsat_i))
    return(indices, [naip_i, dem_i, climate_i, landsat_i])


def construct_training_data(samplesize=None):
    '''
    1. goes into the data directory
    2. gets all the file indices

    3. builds a GeospatialData object
    4. checks that the object has 4 band NAIP
    5. checks that there are actually landsat data
    6. modifies the landsat data such that it is aggregated by month for each year
    7. normalizes data
        a. divide NAIP data by 255
        b. QA the Landsat
        c. terrain.slope divide by 360, terrain.elevation 

    8. creates a 4 element list for X [naip, landsat, terrain, climate]
        Dimensions should be as follows
        a. naip [n, 120, 120, 4]
        b. landsat [n, t, 3, 3, 6]
        c. terrain [n, 1, 1, 1]
        d. climate [n, ct, 1, 1, 7]

    9. creates a 3 element list for y [treeclass, livebiomass, deadbiomass]
    '''
    def qa_naip(gsd):
        #check that there are the minimum number of bands
        return(gsd.naip.images.shape[2] == dc._min_bands)

    def qa_landsat(gsd):
        #check if there is first a minimum number of images
        return(gsd.landsat.shape >= dc._min_t)

    def qa_climate(gsd):
        return((not np.any(gsd.climate.images==dc._error)))

    def norm(data, minimum, maximum):
        '''input data, and normalize between min and max
        '''
        return((data - minimum)/(maximum - minimum))

    def normalize_data(gsd):
        #normalize the naip
        gsd.naip.images = gsd.naip.images / dc._rbg
        #normalize the terrain
        #get the "northness" of aspect
        gsd.terrain.images[0] = (np.cos(np.deg2rad(gsd.terrain.images[0]))+1)/2
        gsd.terrain.images[1] = norm(gsd.terrain.images[1],\
                dc._elevation[0], dc._elevation[1])
        gsd.terrain.images[2] = gsd.terrain.images[2] / dc._slope
        #normalize the climate
        for i in range(7):
            gsd.climate.images[:,i]= norm(gsd.climate.images[:,i],\
                    dc._climate_cnts[i][0], dc._climate_cnts[i][1])
        gsd.landsat.images = gsd.landsat.images / dc._band_reg
        return(gsd)

    def build_geospatialdata(indices, normalize=True):
        gdict = defaultdict(list) 
        print(len(indices))
        for idx in tqdm(indices):
            raw_gsd = GeospatialData(idx) 
            if not qa_naip(raw_gsd):
                print('%s, naip fail'%(idx))
                continue
            if not qa_landsat(raw_gsd):
                print('%s,landsat fail'%(idx))
                continue
            if not qa_climate(raw_gsd):
                print('%s,climate fail'%(idx))
                continue
            if normalize:
                raw_gsd = normalize_data(raw_gsd)
            #print('%s passed qa'%idx)
            gdict['uids'].append(idx)
            gdict['centroid'].append((raw_gsd.fia.lon, raw_gsd.fia.lat))
            gdict['data'].append(raw_gsd)
        return(gdict)
            
    indices, _ = search_for_indices()

    if samplesize is None:
    #just use all the indices, otherwise subset the data
    #build GeospatialData object
        gsd = build_geospatialdata(indices)
    return(gsd)

def create_subxdata_batch(data, name, batchindex):
    '''
    send in batch of indices (could be randomly generated)
    gathers a nd-array of size (len(batchindex), dims)
    and returns it.
    '''
    batch = []
    for i in batchindex:
        if (name == 'naip'):
            _d = getattr(data['data'][i],name).images
        elif (name == 'terrain'):
            _d = np.expand_dims(getattr(data['data'][i],name).images, axis=0)
            _d = np.expand_dims(_d, axis=1)
        elif (name == 'climate'):
            _d = np.expand_dims(getattr(data['data'][i],name).images, axis=1)
            _d = np.expand_dims(_d, axis=1)
        if (name == 'landsat'):
            _d = getattr(data['data'][i],name).images
        batch.append(_d) 
    return(np.array(batch))

def create_xdata_batch(data, batchindex,\
        batchlist = ['naip', 'terrain', 'climate', 'landsat']):
    data_batch = []
    for b in range(len(batchlist)):
        data_batch.append(create_subxdata_batch(data, batchlist[b], batchindex))
    return(data_batch)

def create_ydata_batch(data, batchindex):
    '''
    send in batch of indices, gathers a nd-array of
    size (len(batchindex), dims)
    '''
    lbiomass_batch = []
    dbiomass_batch = []
    n_classes = 7
    class_y = np.zeros((len(batchindex),n_classes)).astype('uint8')
    reg_y = np.zeros((len(batchindex),2))
    for i in range(len(batchindex)):
        class_y[i,int(data['data'][batchindex[i]].fia.classtype)] = 1
        reg_y[i] = np.array([data['data'][batchindex[i]].fia.livebiomass,\
                data['data'][batchindex[i]].fia.deadbiomass])
    return([class_y, reg_y]) 

def gen_data(data, batchsize, validation=True, seed=None):
    np.random.seed(seed) 
    batchindex = np.random.choice(len(data['data']), batchsize, replace=False)
    if not validation:
        X = create_xdata_batch(data, batchindex)
        y = create_ydata_batch(data, batchindex)
        return(X, y, batchindex)
    else:
        #use a 9 to 1 ratio training to validation
        n = int(0.9*len(batchindex))
        train_batch = batchindex[:n]
        val_batch = batchindex[n:]
        X_train = create_xdata_batch(data, train_batch)
        y_train = create_ydata_batch(data, train_batch)
        X_val = create_xdata_batch(data, val_batch)
        y_val = create_ydata_batch(data, val_batch)
        batchindex = [train_batch, val_batch]
        return(X_train, y_train, X_val, y_val, batchindex)

    return(X, y, batchindex)
