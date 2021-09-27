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
import dateutil.parser
import warnings
import scipy as sp
import statistics as st
import glob, os
import statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
path1="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/15-22 settembre dati/bson"
path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/MindsphereFleetManager"

#%%
#read excel file
#os.chdir(path)
#df_fast=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogFast.xlsx")
#df_slow=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogSlow.xlsx")
#df_power=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_Power.xlsx")
#df_power1=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_Power (1).xlsx")
#%%

"""READING AND OPENING FILES JSON AND BSON"""

"""read all the filenames in the given path and return the list of filenames""" 
def read_json_names(path):
    files=[]
    os.chdir(path)
    for file in glob.glob("*.json"):
        files.append(file)
    return files
"""create a global dataframe (pandas) for each json file read"""
def json_file_to_df(filenames, path):
    os.chdir(path)
    for file in filenames:
        file1=file[33:-5]
        file1=file1.replace(" ","")
        file1=file1.replace("(","_")
        file1=file1.replace(")","")
        globals()[file1]=pd.read_json(file)

"""create a variable for phase of interst with the name of all the file of that phase """
"""number of phases=array with the numbers of the phases of interest"""
"""example:  make_bson_list_for_phase(path_prova,[1,2,3])"""
def make_bson_list_for_phase(path,numbers_of_phases):
    files=[]
    os.chdir(path)
    for file in glob.glob("*.bson"):
        files.append(file)
    for i in numbers_of_phases:
        names=[]
        for file in files:
            if file[12:18]==('phase'+str(i)):
                names.append(file)
        globals()['phase'+str(i)]=names
        
def df_from_bson(filename,path):
    with open(path+'/'+filename, "rb") as rf:
        data = bson.decode(rf.read())
    df=pd.DataFrame(data[list(data)[0]])
    df.dropna(subset = ['timestamp'], inplace=True)
    return df

def df_from_phase_bson(filename, path):
    with open(path+'/'+filename, "rb") as rf:
        data = bson.decode(rf.read())
    df=pd.DataFrame(data[list(data)[0]])
    df.dropna(subset = ['timestamp'], inplace=True)
    df_phase=pd.DataFrame(data[list(data)[1]])
    df_phase.dropna(subset = ['timestamp'], inplace=True)
    return df,df_phase                
    
def cycle_list_file(n_cycle,path):
    files=[]
    os.chdir(path)
    for file in glob.glob("*.bson"):
        files.append(file)
    names=[]
    for file in files:
        if file[9:11]==str(n_cycle):
            names.append(file)
            
    return names


def volume_density_bson(files,path):
    volume=[]
    density=[]
    for file in files:
        df=df_from_bson(file, path)
        volume.append(max(np.array(df['PhaseVars.phaseVariable6'])))
        density.append(max(np.array(df['PhaseVars.phaseVariable9'])))
    plt.scatter(volume,density)
    return(volume,density)
def time_density_feeding_bson(files,path):
    time=[]
    density=[]
    finalfeeding=[]
    initialfeeding=[]
    for file in files:
        df=df_from_bson(file, path)
        time.append(max(np.array(df['PhaseVars.phaseVariable1'])))
        density.append(max(np.array(df['PhaseVars.phaseVariable9'])))
        finalfeeding.append(max(np.array(df['PhaseVars.phaseVariable3'])))
        initialfeeding.append(max(np.array(df['PhaseVars.phaseVariable2'])))
    return(time,density,initialfeeding,finalfeeding)


#%%
def func(x,a,b):
    return x*a+b

from scipy.optimize import curve_fit
#from pylab import *
def linear_fit(x,y):
    x=np.array(x)
    y=np.array(y)
    popt, pcov = curve_fit(func, x,y)
    perr=np.sqrt(np.diag(pcov))
    popt_up=popt+perr
    popt_dw=popt-perr
    fit = func(x, *popt)
    fit_up = func(x, *popt_up)
    fit_dw = func(x, *popt_dw)
    fig, ax = plt.subplots(1)
    plt.scatter(x, y, label='data')
    
    sorted_index = np.argsort(x)
    fit_up = [fit_up[i] for i in sorted_index]
    fit_dw = [fit_dw[i] for i in sorted_index]
    plt.plot(x, fit, 'r', lw=2, label='best fit curve')
    ax.fill_between(np.sort(x), fit_up, fit_dw, alpha=.25, label='1-sigma interval')
    
    return popt,perr
#%%
def neg_exp(x,a,b,c,d):
    return a*np.exp(-b*(x**c))+d
def exp_fit(x,y):
    x=np.array(x)
    y=np.array(y)
    popt, pcov = curve_fit(neg_exp, x,y,p0=[160,0.16,0.5,10])#p0 take from a previous fit with no p0 parameter
    perr=np.sqrt(np.diag(pcov))
    fit = neg_exp(x, *popt)
    plt.scatter(x, y, label='data',marker='.')   
    plt.plot(x, fit, 'r', lw=2, label='best fit curve')
    return popt,perr

#%%
def fitting_values_feeding_law(filenames,path,figure=False):
    fit_values=[]
    fit_err=[]
    for file in filenames:
        if figure:
            plt.figure()
        df,df_phase=df_from_phase_bson(file,path)
        df=add_time_as_number(df,'timestamp')
        x=df['Time number']
        y=df['Analogs.analog1']
        val,err=exp_fit(x,y)
        fit_values.append(val)
        fit_err.append(err)
    return fit_values,fit_err



#%%
"""analysis of feeding and pressure in a single phase""" 

def feeding_pressure_in_a_phase(phase_file,path,deg1,deg2):
#    poly2=[]
#    poly4=[]
    for file in phase_file:
        df,df_phase=df_from_phase_bson(file,path)
        plt.figure()
        df.dropna(subset = ["Analogs.analog1","Analogs.analog3"], inplace=True)       
        poly_fit(df["Analogs.analog1"],df["Analogs.analog3"],deg1,deg2)

    
#%%    
"""little pipeline"""
#make_bson_list_for_phase(path_bson,[2,3])
t,d,in_fed,fin_fed=time_density_feeding_bson(phase3,path_bson)
t1=[]
d1=[]
for i in range(0, len(t)): 
    if (in_fed[i]>230 and in_fed[i]<255 and fin_fed[i]>3 and fin_fed[i]<8):
        t1.append(t[i])
        d1.append(d[i])
plt.scatter(t1,d1)
plt.yscale("log")
#%%
    
"""TIME IN FORM OF NUMBER FROM DATES ISO OR DATETIME FORMAT"""
#take data from iso data
def take_datetime(df,timename='_time'):
    dates=[]
    for date in df[timename]:
        date=dateutil.parser.isoparse(date[:-1])
        dates.append(date)
    date=pd.DataFrame({'time':dates})
    return date

#take time from iso 8601 data
def timefromiso(dataiso):
    return datetime.datetime.fromisoformat(dataiso[:-1])

#trasforma orario formato hh,mm,ss,micros in secondi 
def time_to_num(t):
    return(t.microsecond*10**(-6)+t.second+t.minute*60+t.hour*60*60+(t.day-1)*60*60*24)
    
#add to df a column of time in second
"""if time column is in iso format"""  
def add_time_as_number(df,timename='_time'):
    if not(timename in df.columns):
        raise TypeError(timename+' do not exist in the dataframe')
    timenumberdf=pd.DataFrame({"Time number":numbers_from_time(df,timename)})
    return(pd.concat([df,timenumberdf],axis=1))
    
"""if time column is in datetime format"""    
def add_time_as_number2(df,timename='_time'):
    timenum=[]
#    t0=df[timename].iloc[0]
#    t0=time_to_num(t0)
    for time in df[timename]:
        timenum.append(time_to_num(time))#-t0)
    timenumberdf=pd.DataFrame({"Time number":timenum})
    return pd.concat([df,timenumberdf],axis=1)

def numbers_from_time(df,timename='_time'):
    timenum=[]
    if not(timename in df.columns):
        raise TypeError(timename+' do not exist in the dataframe')
    
    t0=time_to_num(timefromiso(df[timename][0]))
    for time in df[timename]:
        t=timefromiso(time)
        timenum.append(time_to_num(t)-t0)
    return (np.array(timenum))



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

#calc volume as itegral of flow

def volume_from_flow(flows,times):
    """flow must be an array"""
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

def plot_2_var_interpoled(var1,var2,time1,time2):
    var2_interp=np.interp(time1,time2,var2)
    plt.figure()
    plt.scatter(time1,var2_interp,marker='.')
    plt.scatter(time2,var2,marker='o',alpha=0.2)
    plt.figure()
    plt.scatter(var2_interp,var1)

#%%
"""scattr plot of all division of the df"""
def scatter_plot_cycles(df,indices,y_var,x_var='Time number'):
    for index in indices:
        plt.figure()
        plt.scatter(df[x_var][index[0]:index[1]],df[y_var][index[0]:index[1]],marker='.')
        
        
#%%
"""analisi densità-volume"""
"""density,time and flow should be array"""
def density_volume(density,flows,time1,time2):
    index_flow=selecting_cycle(flows,1.,3,5) #divide the points of flow in cycles in which flow is not null avoiding fluctuations
    density_interp=np.interp(time1,time2,density) #linear interpolation of density (less sample) with the time of flow (more sample)
    """time1 is the time whit higher number of point, time 2 less point"""
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
    #j= variabile di ciclo: 0= ciclo spento, 1=ciclo acceso
    start_i=[]
    end_i=[]
    times_start=[]
    times_end=[]
    for i in range(0,len(df[var])):
        if (df[var][i]!=0 and not(np.isnan(df[var][i])) and j==0):
            start_i.append(i)
            j=1
        if ((df[var][i]==0 or np.isnan(df[var][i])) and j==1):
            end_i.append(i)
            j=0
    for i in range (0,len(end_i)):
        times_start.append(df[timename][start_i[i]])
        times_end.append(df[timename][end_i[i]])
    return (times_start,times_end)



def select_cycles_indices_by_time(df, times_start,times_end, timename='Time number'):
    indices=[]
    for i in range(0,len(times_start)):
        df1=df.loc[df[timename]<=times_end[i]]
        df1=df1.loc[df1[timename]>=times_start[i]]
        indices.append([df1.index[0]+1,df1.index[len(df1)-1]+1])
    return indices

#%%

def final_feeding_delivery_on_pressure(df,indices,limit_pressure):
    feed_var='analogFast1'
    pressure_var='analogFast3'
    for i in range(0,indices[1]-indices[0]):
        j=indices[1]-i
        if df[pressure_var][j]>limit_pressure:
            return df[feed_var][j-5]

#%%

"""matching total feeding time with slurry density if  flow is almost constant"""
fed_in=[]
T_alim=[]
density=[]
fed_fin=[]
t_s,t_e=select_cycle2_time(dfSlow,'analogSlow21')
indices_slow=select_cycles_indices_by_time(dfSlow,t_s,t_e)
indices_fast=select_cycles_indices_by_time(dfFast,t_s,t_e)
indices_slow=indices_slow[1:]
indices_fast=indices_fast[1:]
for index in indices_slow:
    a=np.max(np.array(dfSlow['analogSlow19'])[index[0]:index[1]])
    fed_in.append(a)
for index in indices_fast:
    a= final_feeding_delivery_on_pressure(dfFast,index,14)
    fed_fin.append(a)  
#    c=np.max(np.array(dfSlow['analogSlow20'])[index[0]:index[1]])
#    density.append(c)
for i in range(0,len(indices_slow)):
    if (fed_in[i]>200. and fed_in[i]<260. and fed_fin[i]>0. and fed_fin[i]<10.):
        b=np.max(np.array(dfSlow['analogSlow21'])[indices_slow[i][0]:indices_slow[i][1]])
        c=np.max(np.array(dfSlow['analogSlow20'])[indices_slow[i][0]:indices_slow[i][1]])
        T_alim.append(b)
        density.append(c)
plt.scatter(T_alim,density)
#%%

import scipy
lin_reg=scipy.stats.linregress(T_alim,density)
xn=np.linspace(min(T_alim),max(T_alim),200)
yn = lin_reg.intercept+lin_reg.slope*xn

plt.plot(T_alim,density, 'or')
plt.plot(xn,yn)
plt.plot(xn,yn+lin_reg.stderr)
plt.plot(xn,yn-lin_reg.stderr)
plt.show()

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
"""polynomial fitting"""
def poly_fit(x,y,deg1=10,deg2=30):
    
    
    z=np.polyfit(x, y,deg1)
    p=np.poly1d(z)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        p30 = np.poly1d(np.polyfit(x, y, deg2))
    xp = np.linspace(0,max(x) , len(y))
    _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')

#%%
def plot_with_same_sampling(df,namevars,path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/immagini generate"):
    for name in namevars:
        for name2 in namevars:
            if name!=name2:
                plt.figure()
                plt.scatter(df[name],df[name2],marker='.')
                plt.xlabel(name)
                plt.ylabel(name2)
                plt.title(name+'-'+name2)
                plt.savefig(path+"/"+name+"-"+name2)
                plt.close()
                
                
def plot_with_different_sampling(df1,df2,var1,var2,timename='Time number',
                                     path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/immagini generate"):
    time1=df1['Time number']
    time2=df2['Time number']
    for name1 in var1:
        for name2 in var2:
            var2_interp=np.interp(time1,time2,var2)
            fig, axs = plt.subplots(2)
            axs[0]=plt.scatter(time1,var2_interp,marker='.')
            axs[0]=plt.scatter(time2,var2,marker='o',alpha=0.2)
            axs[1]=plt.scatter(var2_interp,var1)


                






#%%
"""pypeline for some analysis"""
df_fast1=add_time_as_number(df_fast)
df_slow1=add_time_as_number(df_slow)
df_fast_2_9=df_fast1[5342:19096]
df_slow_2_9=df_slow1[859:2059]

"""for function as density_volume
of volume_from_flow, we should pass an array, not a db series""" 


