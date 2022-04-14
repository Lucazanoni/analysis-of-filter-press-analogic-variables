# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:40:36 2022

@author: lucaz
"""
import tirocinio
import pandas as pd
import numpy as np
import datetime
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
testfiledir=currentdir+'/test_files'
bson='AQS_cycle86_16323932002.bson'
phasebson='AQS_cycle86_phase3_16323927329.bson'

analogdensity='Analogs.analog22',
analogflow='Analogs.analog1',
analogpressure='Analogs.analog3'

def test_df_from_bson():
    """
    TEST:
    if function returns a pandas DataFrame
    """
    
    assert type(tirocinio.df_from_bson(bson,testfiledir))==pd.DataFrame
    

def test_df_from_phase_bson():
    """
    Test:
    if function return 2 elements
    if the 2 elements are pandas DataFrame
    
    """
    assert len(tirocinio.df_from_phase_bson(phasebson,testfiledir))==2
    assert type(tirocinio.df_from_phase_bson(phasebson,testfiledir)[0])==pd.DataFrame
    assert type(tirocinio.df_from_phase_bson(phasebson,testfiledir)[1])==pd.DataFrame
    

    
def test_add_time_as_number():
    """
    Test:
    if function return a Dataframe
    if there is a column called 'Time number'
    
    """
    df=(tirocinio.df_from_bson(bson,testfiledir))
    assert type(tirocinio.add_time_as_number(df,'timestamp'))==pd.DataFrame
    assert 'Time number' in tirocinio.add_time_as_number(df,'timestamp').columns
    
    
def test_time_to_num():
    """
    Test:
    if function return a flot 
    """
    ct = datetime.datetime.now()
    assert type(tirocinio.time_to_num(ct))==float

def test_numbers_from_time():
    """
    Test:
    if function return a numpy array
    """
    df=(tirocinio.df_from_bson(bson,testfiledir))
    assert type(tirocinio.numbers_from_time(df,'timestamp'))==np.ndarray
    
def test_add_time_as_timeseries():
    """
    Test:
    if function return a pandas Dataframe
    """
    df=(tirocinio.df_from_bson(bson,testfiledir))
    assert type(tirocinio.add_time_as_timeseries(df,'timestamp'))==pd.DataFrame

def test_cycle_list_file():
    """
    Test:
    if function return a list
    if the lenght of the test list is equal to 2
    """
    assert type(tirocinio.cycle_list_file(86,testfiledir))==list
    assert len(tirocinio.cycle_list_file(86,testfiledir))==2

def test_t_V_over_V_slopes():
    """
    Test:
    if the function return a integer
    if the function return 0 if there is no change of slope
    """
    x=np.linspace(0,99,100)
    y=np.linspace(0,99,100)
    assert type(tirocinio.t_V_over_V_slopes(x,y,20,10,2))==int
    assert tirocinio.t_V_over_V_slopes(x,y,20,10,2)==0
    
def test_limit_pressure():
    """
    Test:
    if function return a integer
    if return len(pressure)-n_limit-1 if constant pressure case
    """
    pressure=np.ones(100)
    time=np.linspace(0,99,100)
    assert type(tirocinio.limit_pressure(pressure,time))==int
    assert tirocinio.limit_pressure(pressure,time)==79
    
def test_time_volume_over_volume():
    """
    Test:
    if each element return is a list (tested separately)
    """
    tirocinio.change_global_names()
    assert type(tirocinio.time_volume_over_volume([phasebson],testfiledir)[0])==list
    assert type(tirocinio.time_volume_over_volume([phasebson],testfiledir)[1])==list
    assert type(tirocinio.time_volume_over_volume([phasebson],testfiledir)[2])==list
    assert type(tirocinio.time_volume_over_volume([phasebson],testfiledir)[3])==list
    assert type(tirocinio.time_volume_over_volume([phasebson],testfiledir)[4])==list
    assert type(tirocinio.time_volume_over_volume([phasebson],testfiledir)[5])==list

def test_slopes_time_volume_over_volume():
    """
    Test:
    if function return a numpy array
    """
    flows=np.ones(100)
    time=np.linspace(1,100,100)
    assert type(tirocinio.slopes_time_volume_over_volume(flows,time))==np.ndarray
    
def test_residual_humidity_over_time():
    """
    Test:
    if function return a numpy array
    """
    flows=np.ones(1000)*1.5
    time=np.linspace(0,999,1000)
    assert type(tirocinio.residual_humidity_over_time(flows,time,2*flows ))==np.ndarray