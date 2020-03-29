# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:47:16 2020

@author: Pablo Saldarriaga
"""
#%%
import requests
import logging
logger = logging.getLogger(__name__)

from datetime import datetime
#%%
class darksky:
    
    def __init__(self,apiKey):
        logger.info("Inicialización del cliente")           
        self.apiKey = apiKey
        self.url = 'https://api.darksky.net'
        
    def get_hourly(self, lat, lon, r_date=None):
        
        """Request the darksky API for hourly data at (latitude, longitude).
    
            https://darksky.net/dev/docs
        """
        params = {
            "exclude": "currently,daily,alerts,flags",
            "lang": "en",
            "units": "si",
        }
    
        if r_date:
            logger.info("Realiza consulta pasando una fecha en particular")
            end_point ='/forecast/{key}/{lat},{lon},{time}'        
            q_date = r_date.isoformat()
            r = requests.get(self.url + end_point.format(key=self.apiKey, time=q_date, lat=lat, lon=lon), params=params)
        else:
            logger.info("Realiza consulta sin fecha")
            end_point ='/forecast/{key}/{lat},{lon}'
            r = requests.get(self.url + end_point.format(key=self.apiKey, lat=lat, lon=lon), params=params)
        
        if r.status_code >= 200 and r.status_code < 300:
            try:
                res = r.json()
            except Exception as e:
                logger.info(f"Error al convertir los resultados a formato JSON: {e}")
                raise        
            
            logger.info("Organiza la informacion de tiempo como estampa de tiempo")
            data = res['hourly']['data']
            for i in range(len(data)):
                point_date = datetime.fromtimestamp(data[i]['time'])
                data[i]['time'] = point_date
                
        else:
            print("ERROR: ", r.status_code)
    
        return data, r.status_code

