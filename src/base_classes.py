import pandas as pd
import numpy as np
import datetime

class FieldData:
    ''' FieldData contains the 
    plotid, inventory year, class, and biomass value
    '''
    def __init__(self, uniqid: int, invyear: int, plotid: int, lat:int, lon:int,\
            classtype:int, livebiomass: float, deadbiomass: float):
        self.uniqid = uniqid
        self.invyear = invyear
        self.plotid = plotid
        self.lon = lon
        self.lat = lat
        self.classtype = classtype
        self.livebiomass = livebiomass
        self.deadbiomass = deadbiomass

class TSData:
    def __init__(self, filenames: list, acqdates: list,\
            bands: list, images: np.ndarray, \
            units: list):
        self.filenames = filenames
        self.acqdates = acqdates
        self.bands = bands # <list> of bands in image ['B1', 'B2', ...]
        self.images = images
        self.units = units
        if acqdates is not None:
            self.shape = len(acqdates)
    def plot(band=None):
        if band is None:
            plt.plot(self.acqdates,self.images.mean(axis=(1,2)))
        else:
            plt.plot(self.acqdates,self.images[:,:,:,band].mean(axis=(1,2)))
        plt.show()


