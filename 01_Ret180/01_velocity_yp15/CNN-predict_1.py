#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:46:44 2021

@author: ari
"""

import os
import numpy as np
import math
import time
import re
import sys
os.environ["MODEL_CNN"] = "NN_WallRecon";
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE";
#%% Configuration import
import config_deep as config
from src.utils import periodic_padding, periodic_padding_z,output_parser,input_parser,eval_parser

prb_def = os.environ.get('MODEL_CNN', None)

if  prb_def == 'NN_WallRecon':
    app = config.NN_WallRecon
    print('Newtonian Wall-Recon is used')
else:
    raise ValueError('"MODEL_CNN" enviroment variable must be defined as "NN_WallRecon". Otherwise, use different train script.')

os.environ["CUDA_VISIBLE_DEVICES"]=str(app.WHICH_GPU_TEST);


#%% Tensorflow imports

import tensorflow as tf
from tensorflow.keras import layers

#device_name = tf.test.gpu_device_name()
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 
print('Using TensorFlow version:', tf.__version__, ', GPU:', availale_GPUs)
#print(tf.keras.__version__)

if physical_devices:
  try:
    for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

on_GPU = app.ON_GPU
n_gpus = app.N_GPU

distributed_training = on_GPU == True and n_gpus>1

if distributed_training:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    print('Number of devices for distributed training: {}'.format(strategy.num_replicas_in_sync))


#%% Functions for the NN  
def cnn_model():
    input_data = layers.Input(shape=input_shape, name='input_data')
    ini = 30
    dx = 4
    # ------------------------------------------------------------------
    cnv_1 = layers.Conv2D(ini, (3, 3), padding=padding,
                                data_format='channels_first')(input_data)
    bch_1 = layers.BatchNormalization(axis=1)(cnv_1)
    act_1 = layers.Activation('relu')(bch_1)
    # ------------------------------------------------------------------
    cnv_2 = layers.Conv2D(ini+dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_1)
    bch_2 = layers.BatchNormalization(axis=1)(cnv_2)
    act_2 = layers.Activation('relu')(bch_2)
    # ------------------------------------------------------------------
    cnv_3 = layers.Conv2D(ini+2*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_2)
    bch_3 = layers.BatchNormalization(axis=1)(cnv_3)
    act_3 = layers.Activation('relu')(bch_3)
    # ------------------------------------------------------------------
    cnv_4 = layers.Conv2D(ini+3*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_3)
    bch_4 = layers.BatchNormalization(axis=1)(cnv_4)
    act_4 = layers.Activation('relu')(bch_4)
    # ------------------------------------------------------------------
    cnv_5 = layers.Conv2D(ini+4*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_4)
    bch_5 = layers.BatchNormalization(axis=1)(cnv_5)
    act_5 = layers.Activation('relu')(bch_5)
    # ------------------------------------------------------------------
    cnv_6 = layers.Conv2D(ini+5*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
    bch_6 = layers.BatchNormalization(axis=1)(cnv_6)
    act_6 = layers.Activation('relu')(bch_6)
    # ------------------------------------------------------------------
    cnv_7 = layers.Conv2D(ini+6*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_6)
    bch_7 = layers.BatchNormalization(axis=1)(cnv_7)
    act_7 = layers.Activation('relu')(bch_7)
    # ------------------------------------------------------------------
    cnv_8 = layers.Conv2D(ini+7*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_7)
    bch_8 = layers.BatchNormalization(axis=1)(cnv_8)
    act_8 = layers.Activation('relu')(bch_8)
    # ------------------------------------------------------------------
    cnv_9 = layers.Conv2D(ini+8*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_8)
    bch_9 = layers.BatchNormalization(axis=1)(cnv_9)
    act_9 = layers.Activation('relu')(bch_9)
    # ------------------------------------------------------------------
    cnv_10 = layers.Conv2D(ini+9*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_9)
    bch_10 = layers.BatchNormalization(axis=1)(cnv_10)
    act_10 = layers.Activation('relu')(bch_10)
    # ------------------------------------------------------------------
    cnv_11 = layers.Conv2D(ini+10*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_10)
    bch_11 = layers.BatchNormalization(axis=1)(cnv_11)
    act_11 = layers.Activation('relu')(bch_11)
    # ------------------------------------------------------------------
    cnv_12 = layers.Conv2D(ini+11*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_11)
    bch_12 = layers.BatchNormalization(axis=1)(cnv_12)
    act_12 = layers.Activation('relu')(bch_12)
    # ------------------------------------------------------------------
    cnv_13 = layers.Conv2D(ini+12*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_12)
    bch_13 = layers.BatchNormalization(axis=1)(cnv_13)
    act_13 = layers.Activation('relu')(bch_13)
    # ------------------------------------------------------------------
    cnv_14 = layers.Conv2D(ini+13*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_13)
    bch_14 = layers.BatchNormalization(axis=1)(cnv_14)
    act_14 = layers.Activation('relu')(bch_14)
    # ------------------------------------------------------------------
    cnv_15 = layers.Conv2D(ini+14*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_14)
    bch_15 = layers.BatchNormalization(axis=1)(cnv_15)
    act_15 = layers.Activation('relu')(bch_15)
    # ------------------------------------------------------------------
    cnv_16 = layers.Conv2D(ini+14*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_15)
    bch_16 = layers.BatchNormalization(axis=1)(cnv_16)
    act_16 = layers.Activation('relu')(bch_16)
    # ------------------------------------------------------------------
    cnv_17 = layers.Conv2D(ini+13*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_16)
    bch_17 = layers.BatchNormalization(axis=1)(cnv_17)
    act_17 = layers.Activation('relu')(bch_17)
    # ------------------------------------------------------------------
    cnv_18 = layers.Conv2D(ini+12*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_17)
    bch_18 = layers.BatchNormalization(axis=1)(cnv_18)
    act_18 = layers.Activation('relu')(bch_18)
    # ------------------------------------------------------------------
    cnv_19 = layers.Conv2D(ini+11*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_18)
    bch_19 = layers.BatchNormalization(axis=1)(cnv_19)
    act_19 = layers.Activation('relu')(bch_19)
    # ------------------------------------------------------------------
    cnv_20 = layers.Conv2D(ini+10*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_19)
    bch_20 = layers.BatchNormalization(axis=1)(cnv_20)
    act_20 = layers.Activation('relu')(bch_20)
    # ------------------------------------------------------------------
    cnv_21 = layers.Conv2D(ini+9*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_20)
    bch_21 = layers.BatchNormalization(axis=1)(cnv_21)
    act_21 = layers.Activation('relu')(bch_21)
    # ------------------------------------------------------------------
    cnv_22 = layers.Conv2D(ini+8*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_21)
    bch_22 = layers.BatchNormalization(axis=1)(cnv_22)
    act_22 = layers.Activation('relu')(bch_22)
    # ------------------------------------------------------------------
    cnv_23 = layers.Conv2D(ini+7*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_22)
    bch_23 = layers.BatchNormalization(axis=1)(cnv_23)
    act_23 = layers.Activation('relu')(bch_23)
    # ------------------------------------------------------------------
    cnv_24 = layers.Conv2D(ini+6*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_23)
    bch_24 = layers.BatchNormalization(axis=1)(cnv_24)
    act_24 = layers.Activation('relu')(bch_24)
    # ------------------------------------------------------------------
    cnv_25 = layers.Conv2D(ini+5*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_24)
    bch_25 = layers.BatchNormalization(axis=1)(cnv_25)
    act_25 = layers.Activation('relu')(bch_25)
    # ------------------------------------------------------------------
    cnv_26 = layers.Conv2D(ini+4*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_25)
    bch_26 = layers.BatchNormalization(axis=1)(cnv_26)
    act_26 = layers.Activation('relu')(bch_26)
    # ------------------------------------------------------------------
    cnv_27 = layers.Conv2D(ini+3*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_26)
    bch_27 = layers.BatchNormalization(axis=1)(cnv_27)
    act_27 = layers.Activation('relu')(bch_27)
    # ------------------------------------------------------------------
    cnv_28 = layers.Conv2D(ini+2*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_27)
    bch_28 = layers.BatchNormalization(axis=1)(cnv_28)
    act_28 = layers.Activation('relu')(bch_28)
    # ------------------------------------------------------------------
    cnv_29 = layers.Conv2D(ini+dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_28)
    bch_29 = layers.BatchNormalization(axis=1)(cnv_29)
    act_29 = layers.Activation('relu')(bch_29)
    # ------------------------------------------------------------------
    cnv_30 = layers.Conv2D(ini, (3, 3), padding=padding,
                                data_format='channels_first')(act_29)
    bch_30 = layers.BatchNormalization(axis=1)(cnv_30)
    act_30 = layers.Activation('relu')(bch_30)


    # Different branches for different components
    
    # Branch 1
    cnv_b1 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
    if pred_fluct == True:
        act_b1 = layers.Lambda(lambda x: thres_relu(x))(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)
    else:
        act_b1 = layers.Activation('relu')(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)

    losses = {'output_b1':'mse'}
    
    if app.N_VARS_OUT == 2:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
        if pred_fluct == True:
            act_b2 = layers.Lambda(lambda x: thres_relu(x))(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)

        outputs_model = [output_b1, output_b2]

        losses['output_b2']='mse'
    
    elif app.N_VARS_OUT == 3:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
        if pred_fluct == True:
            act_b2 = layers.Lambda(lambda x: thres_relu(x))(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
    
        losses['output_b2']='mse'

        # Branch 3
        cnv_b3 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
        if pred_fluct == True:
            act_b3 = layers.Lambda(lambda x: thres_relu(x))(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)
        else:
            act_b3 = layers.Activation('relu')(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)
    
        outputs_model = [output_b1, output_b2, output_b3]

        losses['output_b3']='mse'
    
    else:
        outputs_model = output_b1
    
    CNN_model = tf.keras.models.Model(inputs=input_data, outputs=outputs_model)
    return CNN_model, losses


# Final ReLu function for fluctuations
def thres_relu(x):
   return tf.keras.activations.relu(x, threshold=app.RELU_THRESHOLD)

# get_custom_objects().update({'thres_relu': layers.Activation(thres_relu)})

#%% Reading configuration

cur_path = app.CUR_PATH
ds_path = app.DS_PATH_TEST
# Average profiles folder
avg_path = ds_path +'/.avg/'

train_yp = app.TRAIN_YP
target_yp = app.TARGET_YP
if not(type(target_yp) is int):
    target_yp = target_yp[0]

n_samples = np.array(app.N_SAMPLES_TEST)
n_samples_tot = np.sum(n_samples)
n_files_test = n_samples.shape[0]

interv = app.INTERV_TEST
tfr_path = ds_path+f'/.tfrecords_singlefile_test_dt{int(11.57*100*interv)}_f32/'
# epochs = app.N_EPOCHS
batch_size = app.BATCH_SIZE
if distributed_training:
    print('WARNING: The provided batch size is used in each device of the distributed training')
    batch_size *= strategy.num_replicas_in_sync
# validation_split = app.VAL_SPLIT
# Learning rate config
init_lr = app.INIT_LR
# lr_drop = app.LR_DROP
# lr_epdrop = app.LR_EPDROP

if app.NET_MODEL == 1:
    pad = tf.constant(64)
    pad_out = 2
    padding_in = 64
    padding_out = 0
else:
    pad = tf.constant(0)
    raise ValueError('NET_MODEL = 1 is the only one implentated so far')

#%% Settings for TFRecords
tfr_files_output = [os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))]

regex = re.compile(f'yp{target_yp:03d}')
regex_tr = re.compile(f'test')
regex_t = re.compile(f'yp{train_yp:03d}')
regex_tb = re.compile('Ret180')
regex_p = re.compile('velocityn25')
regex_q = re.compile('001-of-002')
print(tfr_files_output,tfr_path)
tfr_files_output = [string for string in tfr_files_output if re.search(regex,string) and re.search(regex_tr,string) and re.search(regex_tb,string) and re.search(regex_t,string) and re.search(regex_p,string) and re.search(regex_q,string)]
tfr_files_output = [string for string in tfr_files_output if int((string.split('_')[-3].split('-')[-1])[2:]) == target_yp and int((string.split('_')[-3].split('-')[0])[2:]) == train_yp]

tfr_files_output = [string for string in tfr_files_output if int(string.split('_')[-2][4:7])<n_files_test]

tfr_files_output = sorted(tfr_files_output)

Ret = (tfr_files_output[0].split('/')[-1]).split('_')[0][3:]

(nx_, ny_, nz_) = [int(val) for val in tfr_files_output[0].split('/')[-1].split('_')[1].split('x')]
nx_ = 432
nz_ = 432

#print('nx_in:'+str(nx_in))
#print('nz_in:'+str(nz_in))
#print('nx_out:'+str(nx_out))
#print('nz_out:'+str(nz_out))

# Dictionary for the statistics files from Simson
ypos_Ret180_576 = {'0':0, '1':1, '15':2, '30':3, '50':4, '310':5, '330':6, '345':7, '359':8, '360':9}
#print('WARNING: the y+ indices are computed only at Re_tau = 180')
print(Ret, ny_)
if Ret == str(180):
    if ny_ == 576:
        ypos_Ret = ypos_Ret180_576
    else:
        raise ValueError('Wall-normal resolution not supported')

# Check whether we are predicting the fluctuations
try:
    pred_fluct = app.FLUCTUATIONS_PRED
    if not(str(target_yp) in ypos_Ret):
        raise ValueError("The selected target does not have a corresponding y-index in simulation")
except NameError:
    print('Setting the prediction to full flow fields (default value)')
    pred_fluct = False

# Check whether inputs are normalized as input Gaussian
try:
    norm_input = app.NORMALIZE_INPUT
except NameError:
    norm_input = False

# Checking whether the outputs are scaled with the ratio of RMS values
try:
    scale_output = app.SCALE_OUTPUT
except NameError:
    scale_output = False

# Test-specific parameters
timestamp = app.TIMESTAMP

pred_fld = os.listdir(cur_path+'/.logs/')

for fld in pred_fld:
    if timestamp in fld and fld.split("_")[-1] != "log":
        NAME = fld
        break

try:
    print('[MODEL]')
    print(NAME)
except NameError:
    print('WARNING: Model not found in the logs folder')

pred_fld = os.listdir(cur_path+'/.saved_models/')

for fld in pred_fld:
    if timestamp in fld:
        NAME = fld
        break
        
print(NAME)

nx = nx_ + padding_in
nz = nz_ + padding_in

input_shape = (app.N_VARS_IN, nz, nx)


# Loading the mean profile and the fluctuations intensity if needed
if pred_fluct == True:
    print('The model outputs are the velocity fluctuations')
    avgs = tf.reshape(tf.constant(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_OUT[0]+'.m').astype(np.float32)[:]),(1,8))
    for i in range(1,app.N_VARS_OUT):
        avgs = tf.concat((avgs, tf.reshape(tf.constant(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_OUT[i]+'.m').astype(np.float32)[:]),(1,8))),0)

if scale_output == True:
    rms = tf.reshape(tf.constant(np.loadtxt(avg_path+app.VARS_NAME_OUT[0]+'_rms.m').astype(np.float32)[:]),(1,8))
    if prb_def == 'WallRecon':
        print('The outputs are scaled with the ratio of the RMS values, taking the first input as reference')
    for i in range(1,app.N_VARS_OUT):
        rms = tf.concat((rms, tf.reshape(tf.constant(np.loadtxt(avg_path+app.VARS_NAME_OUT[i]+'_rms.m').astype(np.float32)[:]),(1,8))),0)

if norm_input == True:
    print('The inputs are normalized to have a unit Gaussian distribution')
    avgs_in = tf.reshape(tf.constant(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_IN[0]+'.m').astype(np.float32)[:]),(1,8))
    for i in range(1,app.N_VARS_IN):
        avgs_in = tf.concat((avgs_in, tf.reshape(tf.constant(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_IN[i]+'.m').astype(np.float32)[:]),(1,8))),0)

    rms_in = tf.reshape(tf.constant(np.loadtxt(avg_path+app.VARS_NAME_IN[0]+'_rms.m').astype(np.float32)[:]),(1,8))
    for i in range(1,app.N_VARS_IN):
        rms_in = tf.concat((rms_in, tf.reshape(tf.constant(np.loadtxt(avg_path+app.VARS_NAME_IN[i]+'_rms.m').astype(np.float32)[:]),(1,8))),0)
    std_in = rms_in

print('RMS')
print(std_in)

#%% Data preprocessing with tf.data.Dataset

tfr_files_output_test_ds = tf.data.Dataset.list_files(tfr_files_output, shuffle=False)

tfr_files_output_test_ds = tfr_files_output_test_ds.interleave(lambda x : tf.data.TFRecordDataset(x).take(tf.gather(n_samples, tf.strings.to_number(tf.strings.substr(tf.strings.split(\
                       x,sep='_')[-2],4,3),tf.int64))), cycle_length=1)

dataset_test = tfr_files_output_test_ds.map(output_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
X_test = tfr_files_output_test_ds.map(input_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# Datasets size check ---------------------------------------------------------
#itr = iter(dataset_test)
#j = 0
#for i in range(n_samples_tot):
#    example = next(itr)
#    j += 1
#
#try:
#    example = next(itr)
#except StopIteration:
#    print(f'Train set over: {j}')
#
#sys.exit(0)

# Datasets for evaluation -----------------------------------------------------
#dataset_eval = tfr_files_output_test_ds.map(eval_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#dataset_eval = dataset_eval.batch(batch_size=batch_size)

# Iterating over ground truth datasets ----------------------------------------
itr = iter(dataset_test)
itrX = iter(X_test)

print('Files used for testing')
for fl in tfr_files_output:
    print(fl)
print('')

X_test = np.ndarray((n_samples_tot,app.N_VARS_IN,nz_+padding_in,nx_+padding_in),dtype='float')
Y_test = np.ndarray((n_samples_tot,app.N_VARS_OUT,nz_,nx_),dtype='float')
ii = 0
for i in range(n_samples_tot):
    X_test[i] = next(itrX)
    if app.N_VARS_OUT == 1 :
        Y_test[i,0] = next(itr)
    elif app.N_VARS_OUT == 2 :
        (Y_test[i,0], Y_test[i,1]) = next(itr)
    else:
        (Y_test[i,0], Y_test[i,1], Y_test[i,2]) = next(itr)
    ii += 1
    # print(i+1)
print(f'Iterated over {ii} samples')
print('')
# sys.exit(0)
# Configuration summary -------------------------------------------------------

print('')
print('# ====================================================================')
print('#     Summary of the options for the model                            ')
print('# ====================================================================')
print('')
print(f'Number of samples for training: {int(n_samples_tot)}')
# print(f'Number of samples for validation: {int(n_samp_valid)}')
print(f'Total number of samples: {n_samples_tot}')
# print(f'Batch size: {batch_size}')
print('')
print(f'Data augmentation: {app.DATA_AUG} (not implemented in this model)')
print(f'Initial distribution of parameters: {app.INIT}')
if app.INIT == 'random':
    print('')
    print('')
if app.INIT == 'model':
    print(f'    Timestamp: {app.INIT_MODEL[-10]}')
    print(f'    Transfer learning: {app.TRANSFER_LEARNING} (not implemented in this model)')
print(f'Prediction of fluctuation only: {app.FLUCTUATIONS_PRED}')
print(f'y- and z-output scaling with the ratio of RMS values : {app.SCALE_OUTPUT}')
print(f'Normalized input: {app.NORMALIZE_INPUT}')
print('')
print('# ====================================================================')

# =============================================================================
#   Neural network loading 
# =============================================================================

# PREPARATION FOR SAVING THE RESULTS

pred_path = cur_path+'/.predictions/'
if not os.path.exists(pred_path):
    os.mkdir(pred_path)

#pred_path = cur_path+'/CNN-'+timestamp+'-ckpt/'
pred_path = pred_path+NAME+'/'
if not os.path.exists(pred_path):
    os.mkdir(pred_path)

# Loading model trained
if app.FROM_CKPT == True:
    model_path = cur_path+'/.logs/'+NAME+'/'
    ckpt = app.CKPT
    init_model = tf.keras.models.load_model(
            model_path+f'model.ckpt.{ckpt:04d}.hdf5',
            custom_objects={"thres_relu": layers.Activation(thres_relu)}
            )
    print('[MODEL LOADING]')
    print('Loading model '+str(app.NET_MODEL)+' from checkpoint '+str(ckpt))    
    pred_path = pred_path+f'ckpt_{ckpt:04d}/'
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
else:
    model_path = cur_path+'/.saved_models/'
    init_model = tf.keras.models.load_model(
            model_path+NAME,
            custom_objects={"thres_relu": layers.Activation(thres_relu)}
            # custom_objects={"thres_relu": thres_relu}
            )
    print('[MODEL LOADING]')
    print('Loading model '+str(app.NET_MODEL)+' from saved model')
    pred_path = pred_path+'saved_model/'
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)

# If distributed training is used, we need to load only the weights
if distributed_training:
   padding = 'valid'

   print('Compiling and training the model for multiple GPU')

   with strategy.scope():

       CNN_model, losses = cnn_model()

       CNN_model.compile(loss='mse',
                     optimizer=tf.keras.optimizers.Adam(lr=init_lr))

           
       init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
       CNN_model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
       os.remove('/tmp/model_weights-CNN_keras_model.h5')
       del init_model
           

else:
    CNN_model = init_model

    CNN_model.compile(loss='mse',
                     optimizer=tf.keras.optimizers.Adam(lr=init_lr))

        
print(CNN_model.summary())

#%% Model evaluation
#print('Evaluating model performance')
#loss_values = CNN_model.evaluate(dataset_eval, batch_size=None)
#
#sys.exit(0)

#%% 
Y_pred = np.ndarray((n_samples_tot,app.N_VARS_OUT,nz_,nx_),dtype='float')

if app.N_VARS_OUT == 1:
    Y_pred[:,0,np.newaxis] = CNN_model.predict(X_test, batch_size=batch_size)
if app.N_VARS_OUT == 2:
    (Y_pred[:,0,np.newaxis], Y_pred[:,1,np.newaxis]) = CNN_model.predict(X_test, batch_size=batch_size)
if app.N_VARS_OUT == 3:
    (Y_pred[:,0,np.newaxis], Y_pred[:,1,np.newaxis], Y_pred[:,2,np.newaxis]) = CNN_model.predict(X_test, batch_size=batch_size)

print(type(Y_pred))
print(np.shape(Y_pred))

# Revert back to the flow field
# if scale_output == True:
#    for i in range(app.N_VARS_OUT):
#        print('Rescale back component '+str(i))
#    Y_pred[:,1] *= (1/17.785)
#    Y_pred[:,2] *= (1/5.5428)

# if pred_fluct == True:
#     for i in range(app.N_VARS_OUT):
#         print('Adding back mean of the component '+str(i))
#         Y_pred[:,i] = Y_pred[:,i] + avgs[i][ypos_Ret[str(target_yp)]]
#         Y_test[:,i] = Y_test[:,i] + avgs[i][ypos_Ret[str(target_yp)]]

print(Y_pred.shape)

i_set_pred = 0
while os.path.exists(pred_path+f'pred_fluct{i_set_pred:04d}.npz'):
    i_set_pred = i_set_pred + 1
print('[SAVING PREDICTIONS]')
print('Saving predictions in '+f'pred_fluct{i_set_pred:04d}')    
np.savez(pred_path+f'pred_fluct{i_set_pred:04d}', Y_test=Y_test, Y_pred=Y_pred,X_test = X_test)
