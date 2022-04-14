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
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from datetime import datetime

##%%

def change_global_names(density='Analogs.analog22',flow='Analogs.analog1',pressure='Analogs.analog3'):
    """
    This function create 3 global variables (analogflow,analogpressure,analogdensity) with the name of the variables 
    used in the dataframe.  
    Parameters:
        density: string with the name of the pressure in the Dataframe. Default: 'Analogs.analog22'
        flow: string with the name of the flow in the Dataframe. Default: 'Analogs.analog1'
        pressure: string with the name of the pressure in the Dataframe. Default: 'Analogs.analog3'
    Return:
        Create (or replace) 3 global variables called analogflow, analogpressure and analogdensity that are strings
    """
    
    analogflow='analogflow'
    analogpressure='analogpressure'
    analogdensity='analogdensity'
    globals()[analogflow]=flow
    globals()[analogdensity]=density
    globals()[analogpressure]=pressure



#%%

def extract_bson_files_from_zip(zip_path,destination_path):
    """ 
    EXTRACT ZIP FILES
        
    if bson files are zipped and the contain a folder named output that contain the bson file
    getting all filename of zipfiles in the zip_path directory
    
    Parameters:
        zip_path: string of the directory where the zipped files are
        destination_path: string of the directory where unzip the files
    """ 
    import zipfile
    files=[]
    os.chdir(zip_path)
    for file in glob.glob("*.zip"):
        files.append(file)
    """extract files in the target directory"""
    for file in files:
        with zipfile.ZipFile(file,"r") as zip_ref:
            zip_ref.extractall(destination_path)

#%%
"""READING AND OPENING FILES JSON AND BSON"""


def read_json_names(path):
    
    """
    Read the name of json files in a given directory
    Parameter:
        path: string of path of a given directory
    return:
        list of strings of json files in directory""" 
    files=[]
    os.chdir(path)
    for file in glob.glob("*.json"):
        files.append(file)
    return files

def json_file_to_df(filenames, path):
    
    """
    Create a global Pandas DataFrame for each json file read
    
    Parameters:
        filenames: list of strings of json files
        path: string of the path of a given directory
    returns:
        creates global pandas DataFrames of the json files names present in filenames exixting in directory selected
    
    """
    os.chdir(path)
    for file in filenames:
        file1=file[33:-5]
        file1=file1.replace(" ","")
        file1=file1.replace("(","_")
        file1=file1.replace(")","")
        globals()[file1]=pd.read_json(file)
        


def make_bson_list_for_phase(path,numbers_of_phases):
#example:  make_bson_list_for_phase(path_prova,[1,2,3])
    """
    Create a global list of strings for each phase of interst with the name of all the bson files of that phase.
    The phase of each file should be in the name of the file. Ex: 'AQS_cycle20_phase5.bson', where the phase is 'phase5' in this case

    
    Parameters:
        path: string of the path of a given directory
        number of phase: array containing the phases of interest. Ex: [1,2,3,5]
    Return: 
        a global list with the names (strings) of all bson files in selected directory for each phase of interest.   
    """
    files=[]
    os.chdir(path)
    for file in glob.glob("*.bson"):
        files.append(file)
    for i in numbers_of_phases:
        names=[]
        for file in files:
            if ('phase'+str(i)) in file:
                names.append(file)
        globals()['phase'+str(i)]=names



def df_from_bson(filename,path):
    """
    Create a Pandas DataFrame from bson file
    
    Parameters:
        filename: name (string) of the bson file 
        path: string of the path of a given directory
    return:
        df: Pandas DataFrame of the starting bson file
    
    """    
    with open(path+'/'+filename, "rb") as rf:
        data = bson.decode(rf.read())
    df=pd.DataFrame(data[list(data)[0]])
    df.dropna(subset = ['timestamp'], inplace=True)
    return df



#generate a dataframe pandas from bson file for phase variable, in whing there are 2 db, one with analogs and another with phase
def df_from_phase_bson(filename, path):
    """
    Create 2 Pandas DataFrames from bson file of phase variable. The phase bson files provided has 2 different dataframe in same file,
    so it is necessary to separe them for the differen use of the variables
    
    Parameters:
        filename: name (string) of the bson file 
        path: string of the path of a given directory
    return:
        df: Pandas DataFrame of the "normal/analog" variables of bson file
        df_phase: Pandas DataFrame of the "phase" variables of bson file
    
    """   
    
    
    with open(path+'/'+filename, "rb") as rf:
        data = bson.decode(rf.read())
    df=pd.DataFrame(data[list(data)[0]])
    df.dropna(subset = ['timestamp'], inplace=True)
    df_phase=pd.DataFrame(data[list(data)[1]])
    df_phase.dropna(subset = ['timestamp'], inplace=True)
    return df,df_phase                
   
    

def cycle_list_file(n_cycle,path):
    """
    Create a list of names of all file in a certain directory that has a certain number in position 9-10.
    For example:  'AQS_cycle62_phase2.bson' has number 62
    Useful for selecting the cycles of interest
    
    Parameters:
        n_cycle: int of number of cycle. It must be a number between 10 and 99 ( 2 digit)
        path:string of the path of the directory where the files are
    
    Returns: 
        names: list of strings of the files with 'n_cycle' in position 9-10 of the name in the selected directory    
    
    
    """
    files=[]
    os.chdir(path)
    for file in glob.glob("*.bson"):
        files.append(file)
    names=[]
    for file in files:
        if file[9:11]==str(n_cycle):
            names.append(file)
    return names       
    
            
#TIME IN FORM OF NUMBER FROM DATES ISO OR DATETIME FORMAT


def timefromiso(dataiso):
    """function to elaborate iso8601 time format, useful for other functions"""
    return datetime.fromisoformat(dataiso[:-1])

#trasforma orario formato hh,mm,ss,micros in secondi 
def time_to_num(t):
    """return the number of seconds in TimeStamp object
    Parameter:
        t: TimeStamp object 
    Return:
        number of seconds in t as float
    """ 
    return t.timestamp()#+t.microseconds*10**(-6)
    
#add to df a column of time in second if time column is in iso format
def add_time_as_number(df,timename='_time'):
    """add in Pandas dataframe with time data in iso8601 format a column with data in seconds 
    PARAMETERS:
        df: Pandas dataframe
        timename: the column name of the iso8601 time data. default: '_time'
        
    Returns:
        The initial dataframe with a new column in last position with time in seconds, called 'time', 
        where the first value is the time starting point (0.0s) 
    Raise: 
        ValueError if the called column does not exist in dataframe
    
    """
    if not(timename in df.columns):
        raise ValueError(timename+' do not exist in the dataframe')
    timenumberdf=pd.DataFrame({"Time number":numbers_from_time(df,timename)})
    return(pd.concat([df,timenumberdf],axis=1))
    
#add to df a column of time in second if time column is in datetime format
def add_time_as_number2(df,timename='_time'):
    """
    Add in Pandas DataFrame with time data in timestamp formata a column with data in seconds 
    Parameters:
        df: Pandas dataframe
        timename: the column name of the timestamp time data. default: '_time'
        
    Returns:
        The initial dataframe with a new column in last position with time in seconds, called 'time', 
        where the first value is the time starting point (0.0s) 
    Raise: 
        ValueError if the called column does not exist in dataframe
    
    """
    if not(timename in df.columns):
        raise ValueError(timename+' do not exist in the dataframe')
    timenum=[]
    t0=time_to_num(df[timename].iloc[0])
    for time in df[timename]:
        timenum.append(time_to_num(time)-t0)
    timenumberdf=pd.DataFrame({"Time number":timenum})
    return pd.concat([df,timenumberdf],axis=1)


def numbers_from_time(df,timename='_time'):
    """
    Transfom the column of Dataframe referred to time in timestamp format in seconds
    Parameters:
        df: the Pandas DataFrame which has the time column to transform in seconds
        timename: string of the name of the time column. Default: '_time' 
    Returns:
        array of float containing the element of df[timename] transformed in seconds 
    Raise: 
        VauleError if df[timename] does not exist
    """
    timenum=[]
    if not(timename in df.columns):
        raise ValueError(timename+' do not exist in the dataframe')
    
    t0=time_to_num(timefromiso(df[timename][0]))
    for time in df[timename]:
        t=timefromiso(time)
        timenum.append(time_to_num(t)-t0)
    return (np.array(timenum))

def add_time_as_timeseries(df,timename='timestamp'):
    """
    Add in Pandas DataFrame with time data in iso8601 a column with data in timestamp format
    Parameters:
        df: Pandas dataframe which has the time column in iso8601
        timename: the column name of the timestamp time data. default: 'timestamp'
        
    Returns:
        The initial dataframe with a new column in last position with time in seconds, called 'timeserie', 
        where the first value is the time starting point (0.0s) 

    """
    timeserie=[]
    for time in df[timename]:
        timeserie.append(timefromiso(time))
    timeserie=pd.DataFrame({'timeserie':timeserie})
    return pd.concat([df,timeserie],axis=1)




def t_V_over_V_slopes(time,volume,starting_points=200,slope_points=20,limit=2):
    """
    This function find if there is a changing in the slope of time/volume over volume plot, that should be linear.
    When it's not linear and there is a changing in the slope, the 2 parts of the curve are separated in the linear and the second part.
    The assumption is that the initial part is linear and then calculate when the slopes diverge fron the initial one.
    
    Parameters:
        time: array of float containing the times at which each measure of slurry volume is collected
        volume: array of float of the measured volume of slurry
        starting_points: number (integer) of points used to determine the starting slope of the curve. the index of first point is always 0, 
                         so the firsts starting_points points will be used for the calculation of the slope of the linear part of the curve.
                         Default=200
        slope_points: integer number fo points used to evaluate the slope of the curve after the first starting_points points. The slope is
                      not the tanget of the 'time/volume over volume' curve but a mean to avoid that fluctuations can misrepresent the result.
                      Default=20
        limit: float of limit ratio between the starting slope and the slope calculated in succesive points. if the ratio is bigger than limit,
               the point in which that happens is consider the changing slope point. 
               Default=2
    
    Returns:
        if there is a change in the slope of 'time/volume over volume' curve, the index point in which that happens is returned. Otherwise,
        it is returned 0.
    
    
    """
    t_v=np.array(time/volume)[1:]
    volume=volume[1:]
    #i calculate the starting slope of the curve assuming is linear
    start_slope=sp.stats.linregress(volume[:starting_points],t_v[:starting_points])[0]  
    for i in range(200,len(t_v)-20):
        #i calculate each point the slope of the next 20 points to avoid fluctuation
        slope=sp.stats.linregress(volume[i:i+slope_points],t_v[i:i+slope_points])[0] 
        
        if slope/start_slope>limit:                                  
            # if the slope of next points are over the double of starting slopes, i break the curve 
            #print("the changing of slope occours in position",i)
            return i
    #print("no change of slope")
    return 0 
    #if there is no significative change of slope return 0

#i want that the derivate of pressure is null for a while, so i check if the next n=20 points have derivate very little (<0.01)
def limit_pressure(pressure,time,p_range=.5,n_limit=20,figure=False, path='a'):
    """
    This function finds the pressure value after which the pressure can be considered almost constant, considering fluctuations, 
    after reaching the flow rate desired. It uses the derivative of the pressure and consider when it's almost  null (<0.01)
    for a certain number of point.
    It can also show the point find plotting the graph of pressure and remarking the point found with a red dot
    
    Parameters:
        pressure: array or list of float, conteining the sequence of measured pressures 
        time: array or list of float, containing the times at which each pressure point is register. 
              len(pressure) should be equal to len(time)
        p_range: float. the distance from maximum value of pressure in which i search a region of pressure constant.
                 This is done because in some points of the curve there can be some region 
                 of constant pressure before arriving to the desired range of pressure, that should be near the maximum.
                 Default=0.5
        n_limit: integer number of consecutive points that should have simultaneusly derivative <0.01 to define the pressure is constant
                 Default=20
        figure: boolean. True to plot pressure and point of starting constant pressure, False otherwise
                Default=False
        path: string, the directory where to save the figures. Default: 'a'. If default, it is saved in the current directory
              
    Return: 
        integer, 0 if pressure not constant, >0 if constant, in particular return len(pressure)-n_limit-1
    """
    if path =='a':
        import os,inspect
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    derivate=np.gradient(pressure,time)
    counter=False
    for i in range(len(pressure)-n_limit):
        if (abs(derivate[i])<0.01 and abs(max(pressure)-pressure[i])<p_range):
            counter=True
            for j in range(i,i+n_limit):
                if (abs(derivate[j])>0.01 or abs(max(pressure)-pressure[i])>p_range):
                    counter=False
        if counter:
            if figure:
                plt.figure()
                plt.scatter(time,pressure,marker='.')
                plt.scatter(time[i],pressure[i],color='red')
                from datetime import datetime
                n= datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                plt.savefig(path+'/'+n+'.png')
                plt.close()
            if len(pressure)-n_limit==i+1:
                return i
    print("pressure not constant")
    return 0


def time_volume_over_volume(filenames, path,slope_limit=2,starting_points=200,slope_points=20,figure=False):
    """
    This function calculate the parameters and its uncertainty of the linear regression of time/volume- volume curve in the linear part (pressure constant)
    It does it for all the target files in a directory and join the results.
    In theory this relation should be linear if the pressure is constant, and after few seconds in phase 3 of the cycles it is true.
    if there is a change in the slope the function the changing point for each cycle.
    It can also  make plots of the results
    
    Parameters:
        filenames: list of names(strings) of the files corresponding to the phase of constant pressure during the pumping process 
        path: string of the directory in which the files are
        slope_limit: double, ratio of slopes beyond which the curve is considered to be no more of the initial slope (change of slope). Default:2
        starting_points: int, points used to determine the inizial slope of the curve. Default:200
        slope_points: int, points used to determine the slope of the curve after the  'starting_points' points. Default:20
        figure: boolean, True to plot the results, False otherwise. Default: False
   
   Return:
       
       slopes:list of double, the pendences of the curves calculated as if they are linear in all the reagion of constant pressure
       interceps: list of double, the intercepts of the curves calculated as if they are linear in all the reagion of constant pressure
       err: list of double, the uncertainty of the slopes  
       r_value: list of double, the r value for each curve
       times: list of double, the instant at which in each curve start the reagion of constant pressure
       indices: list of int, the indices, calculated through the function 't_V_over_V_slopes', at which there is a change in the slope 
           of the curve.
       If Figure==True, a graph of time/volume-volume curve is plotted for each file 
       
    """
    slopes=[]
    intercepts=[]
    err=[]
    r_values=[]
    times=[]
    indices=[]
    #index is where the curve is considered no more linear. if zero it is considered linear in all the region at constant pressure
    i=0
    """i is the number for the figure"""
    for file in filenames:
        i=i+1
        df,df_phase=df_from_phase_bson(file, path)
        df=add_time_as_number(df,'timestamp')
        limit=max(np.array(df[analogpressure]))-1 #limit over which the pressure remain constant
        res = next(x for x, val in enumerate(df[analogpressure]) if val > limit)
#        res=limit_pressure(df['Analogs.analog3'],df['Time number'],n_limit=30)
#        if res==0:
#            print(file+" don't have a sufficiently constant pressure" )
#            continue
#        
        #res2 = next(x for x, val in enumerate(df[pressure][res:]) if val <limit-3 ) #when pressure drops in the last seconds it cannot be more considered constant
        res2=-5
        volume=volume_from_flow(np.array(df[analogflow])[res:res2],np.array(df['Time number'])[res:res2])
        time=np.array(df['Time number'])[res:res2]-df['Time number'][res]
        #calculate T over V in the region of costant pressure 
        t_V=(df['Time number'][res:res2]-df['Time number'][res])/volume
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(volume[1:],t_V[1:])
        index=t_V_over_V_slopes(time,volume,slope_points=slope_points,limit=slope_limit,starting_points=starting_points)   #find the index in which the curve is not linear anymore, if exist   
        indices.append(index)
        
        if figure: 
            if index!=0:
                plt.figure()
                plt.scatter(volume[:index],t_V[:index],marker='.')#c=df['Time number'][res:res+i])
                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(volume[1:index],t_V[1:index])
                x=np.linspace(min(volume[:index]),max(volume[:index]),100)
                plt.plot(x,intercept+slope*x,linestyle='--',color='red')
                plt.scatter(volume[index:],t_V[index:],marker='.')#c=df['Time number'][res+i:res2])
#                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(volume[index:],t_V[index:])
#                x=np.linspace(volume[index],volume[-1],100)
#                plt.plot(x,intercept+slope*x,linestyle='--',color='green')
                
                
            else:
                plt.figure()
                plt.scatter(volume,t_V,marker='.',c=df['Time number'][res:res2])
                x=np.linspace(min(volume),max(volume),100)
                plt.plot(x,intercept+slope*x,linestyle='--')
            #plt.colorbar()
            #plt.title('final liquid concentration='+ str(liquid_concentration[-1]))
            
            #print('r_value = ',r_value)
#            x=np.linspace(min(volume),max(volume),100)
#            plt.plot(x,intercept+slope*x,linestyle='--')
            plt.xlabel('V   m3')
            plt.ylabel('t/V     s m^(-3)')
            
            
        slopes.append(slope)
        intercepts.append(intercept)
        err.append(std_err)
        r_values.append(r_value)
        times.append(timefromiso(df['timestamp'][res]))
        #date.append(df['timestamp'][res])
    return slopes,intercepts,err,r_values,times,indices
#%%

def slopes_time_volume_over_volume(flows,times):
    """
    This function calculate the derivative of the curve time/volume-volume starting from instant flows and time.
    It's used to find the change of slope in the curve
    
    Parameters:
        flows: array of double containing the the measured flows
        time: array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
    
    Return:
        numpy array of double containing the gradient of the curve time/volume-volume
    """
    
    times=times-times[0]
    volume=volume_from_flow(flows,times)
    t_V=np.array(times[1:])/np.array(volume[1:])
    return np.gradient(t_V,volume[1:])




    
#here i calculate the residual humidity, that is the mass of H2O residual in the cake
#i have only solid density, slurry density, liquid density and slurry volume


def residual_humidity_over_time(flow,time,slurry_density,figure=False,liquid_density=1.,final_volume=10.8768,solid_density=2.65746):
    """
    This function calculate the residual humidity, i.e. the residual mass of water in the cake, during the formation of the cake.
    Parameter:
        flow: array of double containing the instant flows
        time: array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
        slurry_density: array of double containing the instant densities of the slurry or double containing the mean value
        figure: boolean, True for plotting the changing of residual humidity during the time, False otherwise. Default: False
        liquid_density: density of water extracted. In general it's approximate to 1. Default: 1.
        final_volume: volume of the chamber in which the slurry is pumperd. it's the volume of the cake after the filtration. Default:10.8768
        solid_density: approximate mean density of the solid part of the slurry. Default:2.65746
        
    Return: 
        Array of double containing the residual humidity from the time in which the slurry volume pumped is major than final_volume. the
        lenght of this array is variable. 
        if Figure==True, it will also plotted the array in function of time
    """
    
    
    
    pumped_volume=np.array(volume_from_flow(flow,time))
    solid_concentration=solid_density*(liquid_density-slurry_density)/(slurry_density*(liquid_density-solid_density))
    num=(1-solid_concentration)*(pumped_volume*slurry_density)-(pumped_volume-final_volume)*liquid_density
    den=(pumped_volume*slurry_density-liquid_density*(pumped_volume-final_volume))
    liquid_concentration= num/den
    index=next(x for x, val in enumerate(pumped_volume) if val > final_volume)
    if figure:
        plt.scatter(time[index+1:],liquid_concentration[index+1:],marker='.')
    return liquid_concentration



   
def final_residual_humidity(pumped_volume,slurry_density,liquid_density=1.,final_volume=10.8768,solid_density=2.65746):
    """
    This function calculate the residual humidity, i.e. the residual mass of water in the cake, at the end of the the process.
    Parameters:
        pumperd_volume: double, the total volume of the slurry pumped
        slurry_density: double containing the mean value of the density of slurry
        liquid_density: density of water extracted. In general it's approximate to 1. Default: 1.
        final_volume: volume of the chamber in which the slurry is pumperd. it's the volume of the cake after the filtration. Default:10.8768
        solid_density: approximate mean density of the solid part of the slurry. Default:2.65746
    
    Return: double,  the value of residual humidity at the end of the process
    
    """
    
    solid_concentration=solid_density*(liquid_density-slurry_density)/(slurry_density*(liquid_density-solid_density))
    return (((1-solid_concentration)*pumped_volume*slurry_density-(pumped_volume-final_volume)*liquid_density)/(pumped_volume*slurry_density
           -liquid_density*(pumped_volume-final_volume)))

#density of the cake
def cake_density_over_time(slurry_density,flow,time,liquid_density=1.,final_volume=10.8768,solid_density=2.65746):
    """
    This function calculate the density of the cake during his formation.
    Parameters:
        slurry_density: array of double containing the instant densities of the slurry or double containing the mean value
        flow: array of double containing the instant flows
        time: array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
        liquid_density: density of water extracted. In general it's approximate to 1. Default: 1.
        final_volume: volume of the chamber in which the slurry is pumperd. it's the volume of the cake after the filtration. Default:10.8768
        solid_density: approximate mean density of the solid part of the slurry. Default:2.65746
    
    Return: array of double containing the density of the cake during his formation, from time[1] to his final value 
    """
    solid_concentration=solid_density*(liquid_density-slurry_density)/(slurry_density*(liquid_density-solid_density))
    pumped_volume=volume_from_flow(flow,time)

    cake_density=np.array(pumped_volume)*solid_concentration*slurry_density/final_volume
    return cake_density
#%%
#def solid_concentrations(filenames,path,solid_density=2.65746,liquid_density=1.):
#    solid_concentrations=[]
#    for file in filenames:
#        df,df_phase=df_from_phase_bson(file,path)
#        slurry_density=np.mean(np.array(df[analogdensity]))
#        solid_concentration=solid_density*(liquid_density-slurry_density)/(slurry_density*(liquid_density-solid_density))
#        solid_concentrations.append(solid_concentration)
#    return solid_concentrations
#%%
#
#def all_final_humidity(filenames,path):
#    liquid_concentrations=[]
#    for file in filenames:
#        df,df_phase=df_from_phase_bson(file,path)
#        liquid_concentrations.append(final_residual_humidity(np.array(df_phase['PhaseVars.phaseVariable6'])[-1],
#                                                                        np.mean(np.array(df_phase['PhaseVars.phaseVariable9']))))
#    return liquid_concentrations

                
#density should be mean cause the too big fluctuation of the read values"""
def residual_humidity_of_changing_slope(filenames2,filenames3,path,mean_density=True,solid_density=2.65746,liquid_density=1.):
    """
    This function calculate the residual humidity, i.e. the residual mass of water in the cake, in the point of the curve
    time/volume-volume in which there is the change of slope for all the cycles selected.
    
    Parameters:
        filenames2: list of string of the name of the files of phases 2 of desired cycles.
        filenames3: list of string of the name of the files of phases 3 of desired cycles. the number of elements of 
            filenames3 must be equal to the one of filenames2 and files of same cycle should be at same index in filenames2 and filenames3
        path: string of the path to the directory in which the files are
        mean_density: boolean. True if the slurry density used is the mean of the istantaneous slurry density measured. False to use all 
            the istantaneous density measured are used. Default: True
        solid_density:  solid_density: approximate mean density of the solid component of the slurry. Default:2.65746
        liquid_density: density of water extracted. In general it's approximate to 1. Default: 1.
    
    Returns:
        slope_changing_residual_humidity: array of double containing the residual humidity of each cycle in the point of changing slope
            the index of each element is the same to the one of the corrisponding cycle in filenames2 nd filenames3 
        slope_changing_control: array of integer that check for each cycle if there a point of changing slope. The element is 0 if
            there is no changing of slope in the curve, and the index of changing slope of the corrisponding cycle otherwise. 
            the indices of this array corrispond to the ones in filenames2 and filenames3 (same indices, same cycle) 
    """
    slope_changing_residual_humidity=[]

    slope_changing_control=np.zeros(len(filenames2))
    for i in range(len(filenames2)):
        df2,df_p2=df_from_phase_bson(filenames2[i],path)
        df3,df_p3=df_from_phase_bson(filenames3[i],path)
        flow=np.zeros(len(df2)+len(df3))
        time=np.zeros(len(df2)+len(df3))        
        date=list(add_time_as_timeseries(df2)['timeserie'])
        date.extend(add_time_as_timeseries(df3)['timeserie'])
        flow[:len(df2)]=np.array(df2[analogflow])
        flow[len(df2):]=np.array(df3[analogflow])
        t0=time_to_num(date[0])
        j=0
        for t in date:
            time[j]=time_to_num(t)-t0
            j=j+1
        density=np.zeros(len(df2)+len(df3))
        density[:len(df2)]=np.array(df2[analogdensity])
        density[len(df2):]=np.array(df3[analogdensity])
        if mean_density:
            mean_density=np.mean(density)
            residual_humidity=residual_humidity_over_time(flow,time,mean_density,figure=False)
        else:
            residual_humidity=residual_humidity_over_time(flow,time,density,figure=False)

        limit=max(np.array(df3[analogpressure]))-1   #limit after that i consider constant the pressure
        idx = next(x for x, val in enumerate(df3[analogpressure]) if val > limit)
        idx=idx+len(df2)

        vol=volume_from_flow(flow,time)

        vol_idx=next(x for x, val in enumerate(vol) if val > 10.8768) #index which before the pumped volume is less than the filter volume
                                                                        # and so the formula is not valid
        idx=max([idx,vol_idx])

        index_change=t_V_over_V_slopes(time[idx:],vol[idx:])
        if index_change!=0:
            slope_changing_residual_humidity.append(residual_humidity[index_change+idx])
        else:
            slope_changing_residual_humidity.append(residual_humidity[-1]) #-5 to avoid the values when the pressure is dropped
            slope_changing_control[i]=1
    return slope_changing_residual_humidity, slope_changing_control
#%%
#calculate the error of the residual humidity
def residual_humidity_error(slurry_density,delta_slurry,volume,delta_volume,solid_density=2.65746,
                            delta_solid=.9,liquid_density=1.,filter_volume=10.8768):
    
    """
    This funtion calculate the uncertainty assiociated with the residual humidity, i.e. the residual mass of water in the cake.
    In this case the values of density of water and filtered volume of water are taken without uncertainty.
    
    Parameters:
        slurry_density: double, density of the slurry
        delta_slurry: double, the uncertainty associated with the slurry density
        volume: double, the pumped volume of slurry
        delta_volume: double, the uncertainty associated with the pumped volume of slurry
        solid_density: double, the density of the solid component of the slurry. Default: 2.65746
        delta_solid: double, the uncertainty associated with the solid component of the slurry. Default: .9
        liquid_density: double, the density of the liquid component of the slurry (typically water). Default: 1.
        filter_volume: volume of the chamber in which the slurry is pumperd. it's the volume of the cake after the filtration. Default:10.8768
    
    Returns:
        double, the uncertainty associated to the residual humidity
    """
    
    
    #rh=residual_humidity
    sl_d=slurry_density
    d_sl=delta_slurry
    v=volume
    dv=delta_volume
    so_d=solid_density
    d_so=delta_solid
    ld=liquid_density
    fv=filter_volume
    #d_residual_humidity/d_slurry_density
    dC_dsl=(sl_d*so_d+so_d*(ld-so_d))/(((ld-so_d)*sl_d**2)*(sl_d*v-ld*(v-fv)))+(so_d*(ld-sl_d)*v)/(sl_d*(ld-so_d)*(sl_d*v-ld*(v-fv))**2)

    #d_residual_humidity/d_slurry_volume
    dC_dv=(so_d*(ld-sl_d)/(sl_d*(ld-so_d)))*(sl_d-ld)/((sl_d*v-ld*(v-fv))**(2)) 

    #d_residual_humidity/d_solid_density
    dC_dso=-1*((ld-sl_d)*(ld-so_d)+so_d*(ld-sl_d))/(sl_d*(sl_d*v-ld*(v-fv))*(ld-so_d)**2)

    err=((dC_dsl*d_sl)**2+(dC_dso*d_so)**2+(dC_dv*dv)**2)**0.5
    print('residual humidity error is ',err)
    return err

def volume_error(flow,time,percentual_flow_error=0.005):
    """
    This function calculate the uncertainty associated to the volume calculated as integral of the flow in time.
    In this function the time is taken without uncertainty, because the strumental error on flow is much bigger than the one on time.
    
    Parameters:
        flow: array of double containing the instant flows
        time: array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
        percentual_flow_error: the percentual uncertainty associated to the measures of flow. Default: 0.005 (0.5%)
    
    Returns:
        double, the uncertainty associated to the pumped volume 
        
    """
    err=percentual_flow_error*flow
    tot_err=0
    
    for i in range(len(flow)-1):
        tot_err=tot_err+(err[i]+err[i+1])*0.5*(time[i+1]-time[i])
    return tot_err





#calc volume as itegral of flow

def volume_from_flow(flows,times):
    """
    This function calculate the pumped volume as integral of the flow in the time.
    
    Parameters:
        flows:array of double containing the instant flows, as m^3/h
        times:array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
    Return:
        list of double, the volumes calculated as integral of flow in time. The first value is 0 
    """
    volume=[]
    volume.append(0)
    for i in range(0,len(flows)-1):
        volume.append(volume[i]+(flows[i+1]+flows[i])*(times[i+1]-times[i])/(2*3600))
    return volume

