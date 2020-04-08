#This python function reads NetCDF files downloaded from CHRS Data Portal (http://chrsdata.eng.uci.edu/)
#netCDF4 and collections libraries are required


from netCDF4 import Dataset
from collections import OrderedDict

from pathlib import Path
import numpy as np
import pandas as pd

import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.wkt import loads, dumps
from shapely.geometry import Point
import shapely.geometry as shg

#from mpl_toolkits.basemap import Basemap
# by Phu Nguyen, Hoang Tran, 09-13-2016, contact: ndphu@uci.edu
# Reading satellite precipitation data in NetCDF format downloaded from 
# UCI CHRS's DataPortal(chrsdata.eng.uci.edu)
# Data domain: see info.txt file in the downloaded package for detailed information



def read_netcdf(netcdf_file):
    contents = OrderedDict()
    data = Dataset(netcdf_file, 'r')
    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]
    date = data.variables['datetime'][:]
    crs = data.variables['crs'][:]
    precip = data.variables['precip'][:]
#    for var in data.variables:
#        attrs = data.variables[var].ncattrs()
#        if attrs:
#            for attr in attrs:
#                print('\t\t%s:' % attr, repr(data.variables[var].getncattr(attr)))
#                contents[var] = data.variables[var][:]
#                data = contents['precip']
#                if len(data.shape) == 3:
#                    data = data.swapaxes(0,2)
#                    data = data.swapaxes(0,1)
#            
#            return data,lons,lats,precip,date,crs
#        else:
    return data,lons,lats,precip,date,crs
        
#%%
            
def plot_data_irain(lons,lats,precip):
    lon_0 = lons.mean()
    lat_0 = lats.mean()
    
    m = Basemap(width=5000000,height=3500000,
                resolution='l',projection='stere',\
                lat_ts=40,lat_0=lat_0,lon_0=lon_0)
    
    lon, lat = np.meshgrid(lons, lats)
    xi, yi = m(lon, lat)
    cs = m.pcolor(xi,yi,np.squeeze(precip))
    
    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10)
    
    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    
    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    #cbar.set_label(tmax_units)
    
    # Add Title
    plt.title('prec')
    return None

#%%

