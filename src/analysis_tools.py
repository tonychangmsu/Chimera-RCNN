import matplotlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
import gdal
import osr
import os
from shutil import copyfile
from tqdm import tqdm
from osgeo import gdalconst 
from keras.models import model_from_json
from keras import optimizers
from sklearn.metrics import confusion_matrix
from src.wood_constants import DataConstants as dc
import pandas as pd

'''
def data_to_geojson(df, fileout):
    features = []
    insert_features = lambda x: features.append(geojson.Feature(geometry=\
                    geojson.Point((x['LON'], x['LAT'])),\
                    properties=dict(name=x['UNIQID'])))
    df.apply(insert_features, axis=1)
    with open('%s'%fileout, 'w', encoding='utf8') as fp:
        geojson.dump(geojson.FeatureCollection(features), fp, sort_keys=True,\
                    ensure_ascii=False)
    return(print('wrote %s'%fileout))
'''
def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.
    
    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.       
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.
    
    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int) # maybe crude to cast to int...
    
    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w
    
    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.") 

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps
        
        if np.any(asteps < 1):
             raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps
    
    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
             raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape # just renaming...
    
    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window==0] = 1
    
    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape
    
    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps
    
    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _
        
        new_shape = np.zeros(len(shape)*2, dtype=int)
        new_strides = np.zeros(len(shape)*2, dtype=int)
        
        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides
    
    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]
    
    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)

def rolling_window_multichannel(array, window=(0,), stride=None, channel_last=True, agg=True):
    '''
    performs the rolling window function for many channels
    '''
    if not channel_last:
        array = np.moveaxis(array, 0, -1)
    w, h, c = array.shape
    out_shape = (np.floor((w-window[0])/stride[0]+1).astype(int),\
                 np.floor((h-window[1])/stride[1]+1).astype(int))
    if not agg:
        mc_array = np.empty((out_shape[0],out_shape[1],window[0],window[1],c)).astype('float32')
        for i in range(c): 
            mc_array[...,i]=rolling_window(array[...,i],\
                                        window=(window[0],window[1]),\
                                        asteps=stride)
    else:
        mc_array = np.empty((out_shape[0],out_shape[1],c)).astype('float32')
        for i in range(c): 
            mc_array[...,i]=np.mean(rolling_window(array[...,i], \
                                         window=(window[0],window[1]), \
                                         asteps=stride),axis=(2,3))

    return(mc_array)

def timeseries_rwm(array, window=(0,), stride=None, channel_last=True, agg=True):
    '''
    performs the rolling window function for many channels
    '''
    t, w, h, c = array.shape
    t_array = []
    for i in range(t): 
        t_array.append(rolling_window_multichannel(array[i], \
                                         window=window, \
                                         stride=stride, \
                                         channel_last=channel_last, \
                                         agg=agg))
    return(np.moveaxis(np.array(t_array),0,2))

def reshape_rwm(array):
    new_shape = [np.shape(array)[0] * np.shape(array)[1]]
    for i in range(2,array.ndim):
        new_shape.append(np.shape(array)[i])
    new_shape = tuple(new_shape)
    return(np.reshape(array, new_shape))

def subsample(ds, sample_size=(120,120,4)):
    #if we use floor, we note that we cut off some of the image
    im = np.moveaxis(ds.ReadAsArray(),0,-1)[:,:,:sample_size[-1]]
    ncols = int(np.floor(ds.RasterXSize/sample_size[1]))
    nrows = int(np.floor(ds.RasterYSize/sample_size[0]))
    pr_data = np.zeros((nrows, ncols, sample_size[0], \
            sample_size[1],sample_size[2]))
    #so now assign values to this thing
    #go with nieve solution first
    for i in range(nrows):
        for j in range(ncols):
            lb = i*sample_size[0]
            rb = ((i+1)*sample_size[0])
            ub = j*sample_size[1]
            bb = ((j+1)*sample_size[1])
            pr_data[i,j,:,:,:] = im[lb:rb,ub:bb,:]
            #need to get rid of the nans
    pr_data[np.isnan(pr_data)] = 0
    #reshape this thing to be just a list of samples?
    #n_samples = nrows*ncols
    #pr_data = np.reshape(pr_data, (n_samples, sample_size[0],\
    #        sample_size[1],sample_size[2]))
    return(pr_data)

def aggregate_subsample(ds, sample_size=(120,120,4), agg=np.mean):
    #if we use floor, we note that we cut off some of the image
    im = np.moveaxis(ds.ReadAsArray(),0,-1)[:,:,:sample_size[-1]]
    ncols = int(np.floor(ds.RasterXSize/sample_size[1]))
    nrows = int(np.floor(ds.RasterYSize/sample_size[0]))
    pr_data = np.zeros((nrows, ncols, sample_size[2]))
    for i in range(nrows):
        for j in range(ncols):
            lb = i*sample_size[0]
            rb = ((i+1)*sample_size[0])
            ub = j*sample_size[1]
            bb = ((j+1)*sample_size[1])
            pr_data[i,j,:] = agg(im[lb:rb,ub:bb,:],axis=(0,1))
            #need to get rid of the nans
    pr_data[np.isnan(pr_data)] = 0
    #reshape this thing to be just a list of samples?
    n_samples = nrows*ncols
    pr_data = np.reshape(pr_data, (n_samples, sample_size[2]))
    return(pr_data)

def timeseries_agg_sub(ds_list, ref_file, sample_size=(120,120,7), agg=np.mean):
    #runs the aggregate_subsample effort, but for a time series
    #also utilizes the reference naip as the resampler.
    ts = []
    tmp_filename = './data/ex/temp.tif'
    for dst_f in tqdm(ds_list):
        ds = reproject_resample(dst_f, ref_file, \
                tmp_filename, sample_size[-1])
        ts.append(aggregate_subsample(ds, sample_size))
    return(np.array(ts))

def transform_coordinate(x,y,ref,target=None):
    ref_proj = osr.SpatialReference()
    ref_proj.ImportFromWkt(ref.GetProjection())
    target_proj = osr.SpatialReference()
    if target is None:
        target_proj.ImportFromEPSG(4326)
        #target_proj.ImportFromEPSG(3857) #use web mercator
    else:
        target_proj.ImportFromWkt(target.GetProjection())
    coord_transform = osr.CoordinateTransformation(ref_proj, target_proj)
    target_x, target_y = coord_transform.TransformPoint(x, y, 0)[:-1]
    return(target_x, target_y)

def get_bbox(ref):
    #returns the bounding box in [ulx, uly, lrx, lry]
    gt = ref.GetGeoTransform()
    xdim = ref.RasterXSize
    ydim = ref.RasterYSize
    return([gt[0], gt[3], gt[0]+(xdim*gt[1]), gt[3]+(ydim*gt[-1])])

def transform_bounds(ref, target):
    ref_bbox = get_bbox(ref)
    target_ulx, target_uly = transform_coordinate(ref_bbox[0], ref_bbox[1], ref, target)
    target_lrx, target_lry = transform_coordinate(ref_bbox[2], ref_bbox[3], ref, target)
    return([target_ulx, target_uly, target_lrx, target_lry])

def clip_to_numpixels_gt(ref_file, pixelsize, outwd):
    outname = '%s/%s'%(outwd, os.path.basename(ref_file))
    ref = gdal.Open(ref_file)
    gt = ref.GetGeoTransform()
    outname = '%s/%s'%(outwd, os.path.basename(ref_file))
    ref = gdal.Open(ref_file)
    gt = ref.GetGeoTransform()
    ydim = ref.RasterYSize
    return([gt[0], gt[3], gt[0]+(xdim*gt[1]), gt[3]+(ydim*gt[-1])])

def transform_bounds(ref, target):
    ref_bbox = get_bbox(ref)
    target_ulx, target_uly = transform_coordinate(ref_bbox[0], ref_bbox[1], ref, target)
    target_lrx, target_lry = transform_coordinate(ref_bbox[2], ref_bbox[3], ref, target)
    return([target_ulx, target_uly, target_lrx, target_lry])

def naip_norm(image):
    image[np.isnan(image)] = 0
    image = image/255.
    return(np.moveaxis(image, 0, -1))

def clip_to_numpixels(ref_file, pixelsize, outwd=None):
    outname = '%s/%s'%(outwd, os.path.basename(ref_file))
    ref = gdal.Open(ref_file)
    gt = ref.GetGeoTransform()
    proj = ref.GetProjection()
    xsize = int(np.floor(ref.RasterXSize/pixelsize))*pixelsize
    ysize = int(np.floor(ref.RasterYSize/pixelsize))*pixelsize
    image = ref.ReadAsArray()[:,:ysize,:xsize]
    if outwd is not None:
        bands = ref.RasterCount
        driver = gdal.GetDriverByName('GTiff')
        out = driver.Create(outname, xsize, ysize, bands, gdal.GDT_Byte)
        out.SetGeoTransform(gt)
        out.SetProjection(proj)
        for i in range(bands):
            im = image[i] 
            #fix the nan issue
            #im[np.isnan(im)] = 0
            #normalizing now....
            out.GetRasterBand(i+1).WriteArray(im)
        return(print('created %s'%outname))
    else: 
        return(image)

def reproject_resample(src_filename, match_filename, dst_filename, bands):
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    # We want a section of source that matches this:
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, bands, \
            gdalconst.GDT_Float32)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)
    #dst.GetRasterBand(1).WriteArray()
    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour)
    del dst
    return(gdal.Open(dst_filename))

def reproject_resample_scaled(src_filename, match_filename, scale, dst_dirc=None):
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    # We want a section of source that matches this:
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    # match_ds is the naip at 1 m resolution. 
    wide = int(np.floor(match_ds.RasterXSize/scale))
    high = int(np.floor(match_ds.RasterYSize/scale))
    bands = src.RasterCount
    # Output / destination
    # include the match file index
    index = match_filename.split('_')[4]
    dst_filename = '%s/%s_%s'%(dst_dirc, index, os.path.basename(src_filename))
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, bands, \
            gdalconst.GDT_Float32)
    dst.SetGeoTransform([match_geotrans[0], scale, 0, match_geotrans[3], 0, -scale])
    dst.SetProjection(match_proj)
    #dst.GetRasterBand(1).WriteArray()
    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour)
    del dst
    return(print('wrote %s'%dst_filename))

def export_tif(image, ref_tif, outname, bands=None, dtype=gdal.GDT_Float32, metadata=None, bandmeta=None, verbose=True):
    '''
    Input a numpy array image and a reference geotif 
    to convert image to geotiff of same geotransform
    and projection. Note, if alpha_mask is not None,
    creates a 4 channel geotiff (alpha as last channel)

    Parameters:
    -----------
    image - 3D <numpy array>
    ref_tif - Geotiff reference <gdal object> of same dimensions
    outname - <str> file name to output to (use .tif extension)
    dtype - <str> denoting data type for GeoTiff. Defaults to 8 bit image,
    but can use gdal.GDT_Float32
    '''
    gt = ref_tif.GetGeoTransform()
    proj = ref_tif.GetProjection()
    xsize = np.shape(image)[1] 
    ysize = np.shape(image)[0] 
    if bands is None:
        bands = ref_tif.RasterCount 
    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(outname, xsize, ysize, bands, dtype)
    out.SetGeoTransform(gt)
    out.SetProjection(proj)
    if metadata is not None:
        out.SetMetadata(metadata)
    if bands == 1:
        band = out.GetRasterBand(1)
        band.WriteArray(image)
        if bandmeta is not None:
            band.SetMetadata(bandmeta) 
    else:
        for i in range(bands):
            band = out.GetRasterBand(i+1)
            band.WriteArray(image[:,:,i]) #if we want a red image for a 4 channel
            if bandmeta is not None:
                band.SetMetadata(bandmeta[i]) 
    out = None
    if verbose:
        return(print('created %s'%(outname)))

def landsat_simplify(src, outname):
    '''
    Takes in a 10 band landsat file and converts it to a uint8 file
    with only 7 bands. 
    Bands: 1,2,3,4,5,7,QA 
    '''
    df = gdal.Open(src)
    gt = df.GetGeoTransform()
    proj = df.GetProjection()
    xsize = df.RasterXSize()
    ysize = df.RasterYSize()
    bands = 7
    im = df.ReadAsArray() 
    #convert everything to 0-255?
    out_im = im[:5]
    
def landsat7_destripe(src, outname=None):
    '''
    performs a gdal_fillnodata.py on src file
    '''
    if outname is not None:
        copyfile(src, outname)
        to_destripe = outname
    else:
        to_destripe = src
    bandcount = gdal.Open(to_destripe).RasterCount
    bands = [0,1,2,3,4,6,9]
    mask = ~np.isnan(gdal.Open(to_destripe).ReadAsArray())
    maskfile = './data/ex/tempmask.tif'
    for i in range(1, bandcount):
        export_tif(mask[i-1], gdal.Open(to_destripe), outname=maskfile,\
                bands=1, dtype=gdal.GDT_Byte)
        #make a copy of the file that will be destriped
        src_ds = gdal.Open(to_destripe, gdalconst.GA_Update)
        srcband = src_ds.GetRasterBand(i)
        mask_ds = gdal.Open(maskfile)
        maskband = mask_ds.GetRasterBand(1)
        gdal.FillNodata(srcband, maskband, maxSearchDist=5, smoothingIterations=0)
        srcband = None
        maskband = None
        del src_ds
        del mask_ds
        return(print('%s made!'%outname))

def aggregate_mosaic_landsat(datalist):
    ims, masks = [],[]
    for ds in datalist:
        try:
            i, m = qa_landsat(ds)
        except:
            continue
        ims.append(i)
        masks.append(m)
    ims = np.array(ims)
    masks = np.array(masks)
    out_array = np.ma.mean(np.ma.array(ims, mask=masks), axis=0)
    return(out_array)

def qa_landsat(datafile):
    qa_clear = np.array([672, 676, 680, 684]) #map to 0
    qa_cloud = np.array([752,756,760,764]) #map to 1
    qa_drop = np.array([1, 2, 674])
    qa_snow = np.array([1696,1700,1704,1708,1728,1732,1736,1740]) #map to 2
    qa_shadow = np.array([928,932,936,940,960,964,968,972]) #map to 3
    bad_pixels = np.array([928,932,936,940,960,964,968,972,752,756,760,764,1])
    im = gdal.Open(datafile).ReadAsArray()
    if im is None:
        print('No image in %s'%datafile)
    else:
        #first get rid of the thermal bands
        qa = im[-1]
        qa[np.isnan(qa)] = 1
        im = np.delete(im, [5,6,9], axis=0) #remove thermal bands
        #then make a qa mask
        qa_mask = np.zeros((im.shape[1], im.shape[2])).astype('bool')
        qa_mask[np.isin(qa, bad_pixels)] = True
        mask = np.broadcast_to(qa_mask[...,np.newaxis], (qa_mask.shape[0], qa_mask.shape[1], 7))
        im = np.moveaxis(im, 0, -1)
        #mask is made, now use this mask 
        return(im, mask)

def ndvi(im):
    off = 1e-6
    return((im[:,:,3]-im[:,:,2])/(im[:,:,3]+im[:,:,2]+off))

def import_model(model_json, model_weights):
    '''
    Imports a keras model architecture and 
    associated weights.abs
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

def summarize_model_output(model_history, n_epochs=60, i_epochs=20, \
    dataset='', acc_type='class'):
    m1 = np.zeros(n_epochs) 
    m2 = np.zeros(n_epochs)   
    if acc_type=='class':
        for i in range(int(n_epochs/i_epochs)): 
            m1[i*i_epochs:((i+1)*i_epochs)] = \
            model_history[i].history['%sfc_%s_layer_acc'%(dataset, acc_type)]
            m2[i*i_epochs:((i+1)*i_epochs)] = \
            model_history[i].history['%sfc_%s_layer_loss'%(dataset, acc_type)]
    if acc_type=='regress':
        for i in range(int(n_epochs/i_epochs)): 
            m1[i*i_epochs:((i+1)*i_epochs)] = \
            model_history[i].history['%sfc_%s_layer_det_coeff'%(dataset, acc_type)]
            m2[i*i_epochs:((i+1)*i_epochs)] = \
            model_history[i].history['%sfc_%s_layer_loss'%(dataset, acc_type)]
    return(m1, m2)

def save_accuracies(model_history, n_epochs, i_epochs, outname=None):
    train_class_acc, train_class_loss = summarize_model_output(model_history, \
            n_epochs, i_epochs, dataset='',acc_type='class')
    val_class_acc, val_class_loss = summarize_model_output(model_history, \
            n_epochs, i_epochs, dataset='val_',acc_type='class')
    train_regress_acc, train_regress_loss = summarize_model_output(model_history, \
            n_epochs, i_epochs, dataset='',acc_type='regress')
    val_regress_acc, val_regress_loss = summarize_model_output(model_history, \
            n_epochs, i_epochs, dataset='val_',acc_type='regress')
    out = np.array([[train_class_loss, train_regress_loss, val_class_loss, \
val_regress_loss],[train_class_acc, train_regress_acc, val_class_acc, val_regress_acc]])
    if outname is not None:
        np.save(outname, out)
    return(out)

def plot_fit_history(model_history_file, outname):
    mh = np.load(model_history_file)
    #mh[:,:,3] = mh[:,:,4]
    loss = mh[0]
    acc = mh[1]
    n_epochs = len(acc[0])
    f, ax = plt.subplots(2,2, figsize=(18,9))
    ax[0,0].plot(np.arange(n_epochs), acc[0], label='train')
    ax[0,0].plot(np.arange(n_epochs), acc[2], label='val')
    ax[0,0].set_title('Classification Accuracy', loc='left')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Accuracy')
    ax[0,0].legend(loc='lower right')
    ax[0,0].grid(alpha=0.3)
    ax[0,1].plot(np.arange(n_epochs), loss[0], label='train')
    ax[0,1].plot(np.arange(n_epochs), loss[2], label='val')
    ax[0,1].set_title('Classification Loss', loc='left')
    ax[0,1].grid(alpha=0.3)
    ax[0,1].set_xlabel('Epoch')
    ax[0,1].set_ylabel('Loss')
    ax[0,1].legend(loc='upper right')
    ax[1,0].plot(np.arange(n_epochs), acc[1], label='train')
    ax[1,0].plot(np.arange(n_epochs), acc[3], label='val')
    ax[1,0].set_title('Regression $R^2$', loc='left')
    ax[1,0].grid(alpha=0.3)
    ax[1,0].set_xlabel('Epoch')
    ax[1,0].set_ylabel('$R^2$')
    ax[1,0].legend(loc='lower right')
    ax[1,1].plot(np.arange(n_epochs), loss[1], label='train')
    ax[1,1].plot(np.arange(n_epochs), loss[3], label='val')
    ax[1,1].set_title('Regression RMSE', loc='left')
    ax[1,1].grid(alpha=0.3)
    ax[1,1].set_xlabel('Epoch')
    f.savefig(outname, dpi=300, bbox_inches='tight', padding=None)

def plot_confusion_matrix(cm, classes, normalize=False, outname='conf_mat.png', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
        print(title)
    else:
        title = "Confusion matrix"
        print(title)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    cb=plt.colorbar()

    if normalize:
        cb.set_label('Percent')
    else:
        cb.set_label('$n$')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",\
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Measured label')
    plt.xlabel('Predicted label')

    plt.savefig(outname, dpi=300, bbox_inches='tight', padding=False)
