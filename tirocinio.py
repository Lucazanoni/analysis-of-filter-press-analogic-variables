# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 08:51:14 2021

@author: lucaz
"""

import bson
import pandas as pd
import pylab as plt
import datetime
#%%
#read Bson file
with open("MindsphereFleetManager/AQS_cycle0.bson", "rb") as rf:
    data = bson.decode(rf.read())
    
#se il file è phase aìha variabili normali e phase, cioè due etichette nel dict, altrimenti solo 1    
datafr1=pd.DataFrame(data[list(data)[0]])
if len(list(data))==2:
    datafr_phase=pd.DataFrame(data[list(data)[1]])
#%%
#read excel file
datafr=pd.read_excel("MindsphereFleetManager/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogFast.xlsx")

#%%
# read all the bson filename in the path
#funziona
import glob, os
os.chdir("D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/MindsphereFleetManager")
for file in glob.glob("*.bson"):
    print(file)
    
#%%
#take time from iso 8601 data

def timefromiso(dataiso):
    return datetime.datetime.fromisoformat(dataiso[:-1])

#trasforma orario formato hh,mm,ss,micros in secondi 
def time_to_num(t):
    return(t.microsecond*10**(-6)+t.second+t.minute*60+t.hour*60*60)

#%%
timenum=[]
for time in datafr['_time']:
    t=timefromiso(time)
    timenum.append(time_to_num(t))