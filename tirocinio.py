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
import warnings
import scipy as sp
import statistics as st
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/MindsphereFleetManager"
#%%
#read Bson file
with open(path+"/AQS_cycle10002_phase1.bson", "rb") as rf:
    data = bson.decode(rf.read())
    
#se il file è phase ha variabili normali e phase, cioè due etichette nel dict, altrimenti solo 1    
datafr1=pd.DataFrame(data[list(data)[0]])
if len(list(data))==2:
    datafr_phase=pd.DataFrame(data[list(data)[1]])
    
    #%%
with open('mapping_phasevariables.json') as json_file:
    data_phase=json.load(json_file)
phasevars=pd.DataFrame(data_phase['phaseIdVars'])

#%%
# read all the bson filename in the path
#funziona
import glob, os
files=[]
os.chdir(path)
for file in glob.glob("*.bson"):
    files.append(file)
    
#%%
#read excel file
df_fast=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogFast.xlsx")
df_slow=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogSlow.xlsx")

#%%
#take time from iso 8601 data

def timefromiso(dataiso):
    return datetime.datetime.fromisoformat(dataiso[:-1])

#trasforma orario formato hh,mm,ss,micros in secondi 
def time_to_num(t):
    return(t.second+t.minute*60+t.hour*60*60+(t.day-1)*60*60*24)

#%%
def numbers_from_time(df,timename='_time'):
    timenum=[]
    if not(timename in df.columns):
        raise TypeError(timename+' do not exist in the dataframe')
    #time0=df[timename][:1]
    
    #t0=time_to_num(timefromiso(time0.iloc[time0['Index']==0]))
    for time in df[timename]:
        t=timefromiso(time)
        timenum.append(time_to_num(t))
    return (np.array(timenum))#-timenum[0])



#%%

#mappo le due variabili, se una è il tempo solo variabile tempo, altrimenti faccio x-y, x-t e y-t
def mapping_var_big_than(df,namevar,namevar2,limit=None,timename='_time',title='correlation time-variables'):
   if limit==None:
       df1=df 
   else:
       df1=df.loc[df[namevar]>limit]
   y=df1[namevar]
   if (namevar2=='_time'):
       x= numbers_from_time(df1,namevar2)
       plt.figure()
       plt.scatter(x, df1[namevar],marker='.')
   else:
       time=numbers_from_time(df1,timename)
       x=df1[namevar2]
       fig, axs = plt.subplots(2, 2)
       fig.suptitle(title)
       axs[0,0].scatter(x, y,marker='.')
       axs[0,0].set_title(namevar+' - '+namevar2)
#       plt.xlabel(namevar2)
#       plt.ylabel(namevar)
#       plt.figure()
       axs[0,1].scatter(time, y,marker='.')
       axs[0,1].set_title(namevar+' - time')
#       plt.xlabel('time')
#       plt.ylabel(namevar)
#       plt.figure()
       axs[1,0].scatter(time, x,marker='.')
       axs[1,0].set_title(namevar2+' - time')
#       plt.xlabel('time')
#       plt.ylabel(namevar2)
##       

#%%
#add to df a column of time in second
def add_time_as_number(df,timename='_time'):
    if not(timename in df.columns):
        raise TypeError(timename+' do not exist in the dataframe')
    timenumberdf=pd.DataFrame({"Time number":numbers_from_time(df,timename)})
    return(pd.concat([df,timenumberdf],axis=1))
#%%

#calc volume as itegral of flow

def volume_from_flow(flows,times):
    volume=[]
    volume.append(0)
    for i in range(0,len(flows)-1):
        volume.append(volume[i]+(flows[i+1]+flows[i])*(times[i+1]-times[i])/(2*3600))
    return volume

#%%
    """voglio selezionare una zona di punti, idealmente quelli di un ciclo, che sanno sopra un certo valore eludendo le fluttuazioni"""
def selecting_cycle(arr, limit, n_point_under,n_point_over):
    #arr=array of values
    #limit=lower limit in which points are not good
    #n_point_under=number of point under which no start cycle index is taken
    #n_point_over=number of point under which no end cycle index is taken
    #return=starting and ending cycle indices 
        if(n_point_under<1):
            raise TypeError("n_point_under should be at least 1")
        if(n_point_over<1):
            raise TypeError("n_point_over should be at least 1")
        if(n_point_under>len(arr)):
            raise TypeError("n_point_under could not be more than the length of arr")
        if(n_point_over>len(arr)):
            raise TypeError("n_point_over could not be more than the length of arr")
        if len(arr)<1:
            raise TypeError("arr is empty")
        indices=[]
        index_start=0
        index_end=0
        
        #        h=False
        k=False
        count1=0 #count1 count the numbers over limit
        count2=0
        for i in range(0,len(arr)):
            if arr[i]>limit:
        #                h=True
                count1=count1+1
                count2=0
            else:
        #                h=False
                count1=0
                count2=count2+1
            if (count1==n_point_under  and not(k)):
                index_start=i-n_point_under
                k=True
            if (count2==n_point_over and k):
                index_end=i-n_point_over
                indices.append([index_start,index_end])
                index_start=0
                index_end=0
                k=False
        return indices

"""con  n_point_under=3, e  n_point_over=5 funziona abbastanza bene"""


#%%
"""scattr plot of all division of the df"""
def scatter_plot_cycles(df,indices,y_var,x_var='Time number'):
    for index in indices:
        plt.figure()
        plt.scatter(df[x_var][index[0]:index[1]],df[y_var][index[0]:index[1]],marker='.')
        
        
#%%
"""analisi densità-volume"""
def density_volume(density,flows,time1,time2):
    index_flow=selecting_cycle(flows,1.,3,5) #divide the points of flow in cycles in which flow is not null avoiding fluctuations
    density_interp=np.interp(time1,time2,density) #linear interpolation of density (less sample) with the time of flow (more sample)
    volumes=[]
    densities=[]
    for index in index_flow:
        volume=volume_from_flow(flows[index[0]:index[1]],time1[index[0]:index[1]]) #calulate volumes as integral of flow in time
        volumes.append(volume)
        densities.append(density_interp[index[0]:index[1]])
        plt.figure()
        plt.scatter(volume,density_interp[index[0]:index[1]] , marker='.')
#        plt.figure()
#        plt.scatter(time1[index[0]:index[1]],volume, marker='.')
    return(densities, volumes)
    
    
#%%
"""divide in cycle using total feeding time=analog 21"""
#
def select_cycle2_time(df,var,timename='Time number'):
    j=0
    start_i=[]
    end_i=[]
    times_start=[]
    times_end=[]
    for i in range(0,len(df[var])):
        if (df[var][i]!=0 and j==0):
            start_i.append(i)
            j=1
        if (df[var][i]==0 and j==1):
            end_i.append(i)
            j=0
    for i in range (0,len(end_i)):
        times_start.append(df[timename][start_i[i]])
        times_end.append(df[timename][end_i[i]])
    return (times_start,times_end)



#if we have time start and time end, this takes the indices of start and end of the cycle

def select_cycles_indices_by_time(df, times_start,times_end, timename='Time number'):
    indices=[]
    for i in range(0,len(times_start)):
        df1=df.loc[df[timename]<=times_end[i]]
        df1=df1.loc[df1[timename]>=times_start[i]]
        indices.append([df1.index[0]+1,df1.index[len(df1)-1]+1])
    return indices




#%%

"""we want to compare some measure of different cycle
this function, giving the array of values, theindices of the cycles and the measure we want, give a statistic about the cycles
"""

def cycle_stat_measure(arr, indices,statistic,statistic_libr):
    """indices should be a matrix Nx2 or a array of array 2x1"""
    vals=[]
    module = __import__(statistic_libr)
    method_to_call = getattr(module, statistic)
    for index in indices:
        vals.append(method_to_call(arr[index[0]:index[1]]))
    return vals

def boxplot(arr, indices):
    plt.figure()
    data=[]
    for index in indices:
        data.append(arr[index[0]:index[1]])
    plt.boxplot(data)


#%%
    
"""time series analysis"""











    


#%%
"""polynomial fitting"""
def poly_fit(x,y,deg=10):
    
    
    z=np.polyfit(x, y,deg)
    p=np.poly1d(z)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        p30 = np.poly1d(np.polyfit(x, y, 30))
    xp = np.linspace(0,x[len(x)] , len(y))
    _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')



#%%
"""
PROVE FUNZIONI E PER CAPIRE I DATI

#%%
#division of df analogFast about 1-9-2021 in cycles
df_1_9=add_time_as_number(datafr[:5343])
#timenumberdf=pd.DataFrame({"Time number":numbers_from_time(df_1_9,'_time')})
#df_1_9=pd.concat([df_1_9,timenumberdf],axis=1)

#timefs_1_9=numbers_from_time(df_1_9,'_time')
df_1_9_cut1=df_1_9[116:880]
df_1_9_cut2=df_1_9[1220:1574]
   
df_1_9_cut3=df_1_9[1585:2601]  
   
df_1_9_cut4=df_1_9[2680:5155]
df_1_9_cut5=df_1_9[5167:5247]
df_sl_1_9=add_time_as_number(datafrsl[:858])
#%%
   
   
df_1_9=datafr[:5343]
timefs_1_9=numbers_from_time(df_1_9,'_time')   
timesl_1_9=numbers_from_time(df_sl_1_9,'_time')
plt.figure()
plt.scatter(timesl_1_9,df_sl_1_9['analogSlow8'],marker='.')

plt.scatter(timefs_1_9,df_1_9['analogFast1'],marker='.')
plt.figure()
plt.scatter(timesl_1_9,df_sl_1_9['analogSlow8'],marker='.')

plt.scatter(timefs_1_9,df_1_9['analogFast3'],marker='.')

#%%
plt.figure()
plt.scatter(df_sl_1_9['Time number'],df_sl_1_9['analogSlow8'],marker='.')
plt.scatter(df_1_9['Time number'],df_1_9['analogFast1'],marker='.')
plt.scatter(df_1_9['Time number'],df_1_9['analogFast3'],marker='.')



#%%

fig, axs = plt.subplots(3, 1)
axs[0].scatter(datafr1['Time number'],datafr1['Analogs.analog1'],marker='.')
axs[1].scatter(datafr1['Analogs.analog1'],datafr1['Analogs.analog3'],marker='.')
axs[2].scatter(datafr1['Time number'],datafr1['Analogs.analog3'],marker='.')

"""

