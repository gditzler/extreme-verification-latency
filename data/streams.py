#!/usr/bin/env python 

import pandas as pd 
import numpy as np

def stream_file_loader(experiment_name:str='1CDT', 
                       chunk_size:int=500): 
    """read data in from a file stream
    """
    df = pd.read_csv(''.join(['data/files/', experiment_name, '.txt']), header=None)
    X, Y = df.values[:,:-1], df.values[:,-1]
    N = len(Y)

    # set Xinit and Yinit
    Xinit, Yinit = X[:chunk_size,:], Y[:chunk_size]
    Xt, Yt = [], []

    for i in range(chunk_size, N-chunk_size, chunk_size): 
        Xt.append(X[i:i+chunk_size])
        Yt.append(Y[i:i+chunk_size])
    
    return Xinit, Yinit, Xt, Yt


def generate_stream(dataset_name:str='', 
                    dataset_params:dict={'val': None}):
    """
    """
    return None 