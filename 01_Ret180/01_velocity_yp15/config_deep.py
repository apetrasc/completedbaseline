#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mon  1 10:06:34 2021

@author: ari
"""

class NN_WallRecon:
    # Problem definition
    N_VARS_IN = 3
    N_VARS_OUT = 3

    VARS_NAME_IN = ('v','u','p')
    VARS_NAME_OUT = ('v','w','u')
    
    NORMALIZE_INPUT = True        # Normalize input to have zero average and unit variance
    SCALE_OUTPUT = False           # Scale output with the ratio of the RMS values
    
    FLUCTUATIONS_PRED = True      # Predict only the fluctuations of the output
    RELU_THRESHOLD = -1.0           # Relu threshold for last layer in main network

    TRAIN_YP = 1
    TARGET_YP = (15)
    
    # Hardware definition
    ON_GPU = True
    N_GPU = 4
    WHICH_GPU_TRAIN = 0
    WHICH_GPU_TEST = 0

    # Dataset definition
    
    CUR_PATH = '/mimer/NOBACKUP/groups/kthmech/sadanori/01_Ret180/01_velocity_yp15'
    ## Ret180
    DS_PATH_TRAIN = '/mimer/NOBACKUP/groups/kthmech/argb/02_VE/n25/99_dataset/Train'
    DS_PATH_VALIDATION = '/mimer/NOBACKUP/groups/kthmech/argb/02_VE/n25/99_dataset/Validation'
    DS_PATH_TEST = '/mimer/NOBACKUP/groups/kthmech/argb/02_VE/n25/99_dataset/Test'

    N_SAMPLES_TRAIN = (2000,2000,2000,2000,2000,712,2000,2000,2000,2000,2000,712)
    N_SAMPLES_VALIDATION = (2000,1210,2000,1210)
    
    INTERV_TRAIN = 1

    N_SAMPLES_TEST = (2000,1210)

    INTERV_TEST = 1
    
    TIMESTAMP = ''  # add some information here to recognize the model
    
    FROM_CKPT = True     # to load the model saved at checkpoint CKPT
    CKPT = 50
    
    # Prediction that has to be post-processed
    PRED_NB = 0
    
    DATA_AUG = False
    
    # Network definition
    NET_MODEL = 1
    # See README for network descriptions
    
    # Model-specific options (syntax: [option]_[net_model])
    PAD_1 = 'wrap'
    
    # Training options
    INIT = 'random' # default value: 'random'
    INIT_MODEL = ''
    TRANSFER_LEARNING = False
    N_TRAINABLE_LAYERS = 3
    
    N_EPOCHS = 5
    BATCH_SIZE = 4
    VAL_SPLIT = 0.2
    
    INIT_LR = 0.001
    LR_DROP = 0.5
    LR_EPDROP = 20.0#12.0
    
    # Callbacks specifications
    TB_HIST_FREQ = 0
    CKPT_FREQ = 10
