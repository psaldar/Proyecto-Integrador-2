
import re
import os
import sqlite3
import requests
import numpy as np
import pandas as pd
from pathlib import Path
import shapely.geometry as shg
from shapely.wkt import loads, dumps
from shapely.geometry import Point,Polygon
#%%
import zipfile
from dateutil import tz
from netCDF4 import Dataset
from datetime import datetime
from collections import OrderedDict
from scripts.iRain.read_netcdf import plot_data_irain
import logging
#%%
logger = logging.getLogger(__name__) 
#ulx: Start_lon
#uly: End_lat
#lry: Start_lat
#lrx: End_lon

def get_irain_file(path,
                   startDate,
                   endDate,
                   correo = 'psaldar2@eafit.edu.co',
                   ulx = -75.733,
                   lrx = -75.413,
                   lry=6.084,
                   uly = 6.404,
                   timestepText = "1hrly",
                   dataType = "CCS",
                   formattext = "NetCDF",
                   compression = "zip",
                   timestepAlt = "1h",
                   domain= "rectangle"):
    
    
    domainLatLng = str(ulx)+" "+str(lry)+" "+str(lrx)+" "+str(+uly)
    rq = requests.Session()
    
    logger.info(startDate)
    download_rect_url = f'http://chrsdata.eng.uci.edu/php/downloadRectangleData.php?ulx={ulx}&uly={uly}&lrx={lrx}&lry={lry}&startDate={startDate}&endDate={endDate}&dataType={dataType}&format={formattext}&timestep={timestepText}&compression={compression}&timestepAlt={timestepAlt}'

    response = rq.get(download_rect_url).json()
    domainLatLng = str(ulx)+" "+str(lry)+" "+str(lrx)+" "+str(+uly)
    if 'error' in response:
        logger.info(f"{response['error']} + 'Para las fechas: {startDate} y {endDate}'")
        print(response['error'] + f'Para las fechas: {startDate} y {endDate}')

    email_download = f"http://chrsdata.eng.uci.edu/php/emailDownload.php?email={correo}&downloadLink=http://chrsdata.eng.uci.edu/userFile/{response['userIP']}/tempShape/{response['_dataType']}/{response['dataType']}_{response['zipFile']}.{compression}&fileExtension={compression}&dataType={response['dataType']}&timestep={timestepText}&startDate={startDate}&endDate={endDate}&domain={domain}&domain_parameter={domainLatLng}"
    response2 = rq.get(email_download)
    dlik = f"http://chrsdata.eng.uci.edu/userFile/{response['userIP']}/tempRectangle/{response['_dataType']}/{response['dataType']}_{response['zipFile']}.{compression}"
    response3 = rq.get(dlik)
    logger.debug(response3.status_code)   
    
    zip_file_path = path / f"{response['dataType']}_{response['zipFile']}.{compression}"
    file_name = f"{response['dataType']}_{response['zipFile']}"

    with open(zip_file_path.expanduser().resolve(), 'wb') as zip_file:
        zip_file.write(response3.content)
        
        zip_file.close()
    return file_name,zip_file_path

#%%

def unzip_iradin_file(path,file_name,zip_file_path):
    zfile = zipfile.ZipFile(zip_file_path.expanduser().resolve())
    file = file_name+'.nc'    
    file_path = zfile.extract(file, path.expanduser().resolve())
    info ={}
    with zfile.open('info.txt') as f:
        for i , line in enumerate(f):
            if i > 1:
                aux = re.search('([\S]*)([\s]*)([\S]*)',line.decode())
                info[aux.group(1)]=aux.group(3)

    return Path(file_path),info


def read_netcdf(netcdf_file):
    data = Dataset(netcdf_file, 'r')
    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]
    date = data.variables['datetime'].units
    crs = data.variables['crs'][:]
    precip = data.variables['precip'][:]
    
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Bogota')
    utc = datetime.strptime(date[-16:], '%Y-%m-%d %H:%M')
    utc = utc.replace(tzinfo=from_zone)
    central = utc.astimezone(to_zone)
    
    return data,lons,lats,precip,central,utc,crs

def read_lat_lon(netcdf_file,utc_col):
    data = Dataset(netcdf_file, 'r')
    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]
    precip = data.variables['precip'][:]
    lon, lat = np.meshgrid(lons, lats)
    lon_1 = lon.ravel()
    lat_1 = lat.ravel()
    precip_1 = precip.ravel()
    
    data_frame = pd.DataFrame({'LON':lon_1,'LAT':lat_1,'PRECIP':precip_1})
    data_frame['FECHAHORA'] =  utc_col
    return data_frame
    #return data_frame[data_frame['PRECIP']>0]


def process_data(lons,lats,precip,date,utc,info):
    lon, lat = np.meshgrid(lons, lats)
    tot = []
    cellsize = float(info['cellsize'])
    for la, lo, prec in zip(lat,lon,precip):
        for x,y,p in zip(lo,la,prec):
            if (p >= 0 ):
#                tot.append([Point(x, y),p,date,utc])
                pol = Polygon([[x,y],[x+cellsize,y],[x+cellsize,y-cellsize],[x,y-cellsize], [x, y]])
                tot.append([pol,p,date,utc,cellsize])
    df = pd.DataFrame(tot,columns=['POLYGON','PRECIP_VALUE','DATE','DATE_UTC','CELLSIZE'])            
#    df['DATE'] = pd.to_datetime(df['DATE'])
    return df
   

def get_data(path ,
             startDate ,
             endDate, 
             ulx = -75.733,
             lrx = -75.413,
             lry = 6.084,
             uly = 6.404):    
    
    file_name,zip_file_path = get_irain_file(path,startDate,endDate,ulx = ulx,uly =uly,lry=lry,lrx = lrx)
    netcdf_file,info_data = unzip_iradin_file(path,file_name,zip_file_path)
    data,lons,lats,precip,central,utc,crs = read_netcdf(netcdf_file)
    results = process_data(lons,lats,precip,central,utc,info_data)
#    os.remove(zip_file_path.expanduser().resolve())
#    os.remove(netcdf_file.expanduser().resolve())
    
    return results

def get_lat_lon_file(path ,startDate ,endDate,aux, ulx = -77.656,uly = 5.056,lry=2.930,lrx = -75.546):    
    file_name,zip_file_path = get_irain_file(path,startDate,endDate,ulx = ulx,uly =uly,lry=lry,lrx = lrx)
    netcdf_file,info_data = unzip_iradin_file(path,file_name,zip_file_path)
    data = read_lat_lon(netcdf_file,aux)
    os.remove(zip_file_path.expanduser().resolve())
    #os.remove(netcdf_file.expanduser().resolve())
    return data
#%%

