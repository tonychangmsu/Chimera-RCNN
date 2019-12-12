import numpy as np
import pandas as pd
import gdal
from osgeo import gdalconst 
import os
from glob import glob 

def export_tif(image, ref_tif, outname, bands=None, dtype=gdal.GDT_Float64, q=True,\
 metadata=None, bandmeta=None):
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
    if not q:
        return(print('created %s'%(outname)))


