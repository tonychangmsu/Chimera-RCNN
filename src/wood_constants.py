class FileConstants(object):
    _naip_wd = './data/wood-supply/NAIP'
    _dem_wd = './data/wood-supply/DEM'
    _climate_wd = './data/wood-supply/CLIMATE'
    _landsat_wd = './data/wood-supply/LANDSAT'
    _master = './data/master_Metadata_Source.csv'
    
    #zeros
    _naip_wdz = './data/wood-supply/ZEROS/NAIP'
    _dem_wdz = './data/wood-supply/ZEROS/DEM'
    _climate_wdz = './data/wood-supply/ZEROS/CLIMATE'
    _landsat_wdz = './data/wood-supply/ZEROS/LANDSAT'
    _masterz = './data/wood-supply/ZEROS/WS_NAIP2_ZEROS_USDA_NAIP_DOQQ_metadata_source.txt'

class DataConstants(object):
    #biomass constants
    _bm_scale = 10000/(2204.62*672)
    #naip constants
    _rbg = 255
    _min_bands = 3
    #landsat constants
    _min_t = 1
    _shadow = 1
    _cloud = 2
    _fill = 5
    _band_reg = 255
    #terrain constants
    #_elevation_old = (-85, 4000) #deprecated
    _elevation = (-86, 4414) #meters
    _slope = 360 #0-1 degree/360
    _aspect = 360 #0-1 degree/360
    #climate constants
    #from full range of PRISM for entire time period
    _error = -9999
    _ppt = (0, 2639.82) #mm/month
    _tdmean = (-29.98, 26.76)
    _tmax = (-29.8, 49.11)
    _tmean = (-30.8, 41.49) #C
    _tmin = (-32.33, 34.72)
    _vpdmax = (0.009, 110.06)
    _vpdmin = (0, 44.79)
    _climate_cnts = {'ppt':_ppt, 'tdmean':_tdmean, 'tmax':_tmax,\
                    'tmean':_tmean, 'tmin':_tmin,\
                    'vpdmax':_vpdmax, 'vpdmin':_vpdmin}
    #deprecated
    _climate_cnts_old2 = [_ppt, _tmean, _tmin, _tmax,\
            _tdmean, _vpdmin, _vpdmax]
    _climate_cnts_old = [_ppt, _tdmean, _tmax, _tmean,\
            _tmin, _vpdmax, _vpdmin]

