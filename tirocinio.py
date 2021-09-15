# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 08:51:14 2021

@author: lucaz
"""
import numpy as np
import bson
import json
import pandas as pd
import pylab as plt
import datetime
import scipy as sp
from scipy.signal import argrelextrema
#%%
path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/MindsphereFleetManager"
#%%
#read Bson file
with open(path+"/AQS_cycle0.bson", "rb") as rf:
    data = bson.decode(rf.read())
    
#se il file è phase ha variabili normali e phase, cioè due etichette nel dict, altrimenti solo 1    
datafr1=pd.DataFrame(data[list(data)[0]])
if len(list(data))==2:
    datafr_phase=pd.DataFrame(data[list(data)[1]])
#%%
#read excel file
datafr=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogFast.xlsx")

#%%
# read all the bson filename in the path
#funziona
import glob, os
files=[]
os.chdir(path)
for file in glob.glob("*.bson"):
    files.append(file)
    
#%%
#take time from iso 8601 data

def timefromiso(dataiso):
    return datetime.datetime.fromisoformat(dataiso[:-1])

#trasforma orario formato hh,mm,ss,micros in secondi 
def time_to_num(t):
    return(t.microsecond*10**(-6)+t.second+t.minute*60+t.hour*60*60+t.day*60*60*24)

#%%
def numbers_from_time(df,timename):
    timenum=[]
    #time0=df[timename][:1]
    
    #t0=time_to_num(timefromiso(time0.iloc[time0['Index']==0]))
    for time in df[timename]:
        t=timefromiso(time)
        timenum.append(time_to_num(t))
    return (np.array(timenum)-timenum[0])

    #%%
with open('mapping_phasevariables.json') as json_file:
    data_phase=json.load(json_file)
phasevars=pd.DataFrame(data_phase['phaseIdVars'])



#%%

#mappo le due variabili, se una è il tempo solo variabile tempo, altrimenti faccio x-y, x-t e y-t
def mapping_var_bigthan(df,namevar,namevar2,limit=None,timename='_time'):
   if limit==None:
       df1=df 
   else:
       df1=df.loc[df[namevar]>limit]
   y=df1[namevar]
   if (namevar2=='_time'):
       x= numbers_from_time(df1,namevar2)
       plt.show()
       plt.scatter(x, df1[namevar],marker='.')
   else:
       time=numbers_from_time(df1,timename)
       x=df1[namevar2]
       plt.figure()
       plt.scatter(x, y,marker='.')
       plt.xlabel(namevar2)
       plt.ylabel(namevar)
       plt.figure()
       plt.scatter(time, y,marker='.')
       plt.xlabel('time')
       plt.ylabel(namevar)
       plt.figure()
       plt.scatter(time, x,marker='.')
       plt.xlabel('time')
       plt.ylabel(namevar2)
       
#%%

minima=[]   
minima=argrelextrema(np.array(datafr3['analogFast3']), np.less)

   
   
   
   
   
   
   
   
   
   
   
   
   
   