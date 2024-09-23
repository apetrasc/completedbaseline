import os
import scipy.io as sio

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import math
import time
from matplotlib import ticker
import matplotlib.colors as colors
import sys
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.lines import Line2D

cwd = os.getcwd()
batchn = cwd.split('/')[-2].replace("batch","")[:2]
casen = cwd.split('/')[-1].replace("case","")[:2]

os.makedirs(cwd+'/.jupyter_plots', exist_ok=True)

import config_deep as config

app = config.NN_WallRecon

train_yp = app.TRAIN_YP
target_yp = app.TARGET_YP

typ = app.TRAIN_YP
fyp = app.TARGET_YP
dyp = app.TARGET_YP

print_samples = 2

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{bm}')
plt.rc('font', family='serif')
plt.rc('axes', labelsize=12, titlesize=12)
plt.rc('font', size=12)
plt.rc('legend', fontsize=12)               # Make the legend/label fonts
plt.rc('xtick', labelsize=12)               # a little smaller
plt.rc('ytick', labelsize=12)

plt.close('all')

mrk_size = 3.5

N_VARS_IN = app.N_VARS_IN
VARS_NAME_IN = app.VARS_NAME_IN
N_VARS_OUT = app.N_VARS_OUT
VARS_NAME_OUT = app.VARS_NAME_OUT
ckpt = app.CKPT
avg_path = '/mimer/NOBACKUP/groups/kthmech/argb/02_VE/new_case_files/99_dataset/Train/.avg/'

pad = 64  ## Padding for base network
fl = [0,1]  ## pred_fluct{fl}.npz to be read

print('Batch = '+str(batchn))
print('Case = '+str(casen))
print('Train_yp = '+str(train_yp))
print('Target_yp = '+str(target_yp))
print('Train variables = '+str(VARS_NAME_IN))
print('Target variables = '+str(VARS_NAME_OUT))
print('No. of train variables = '+str(N_VARS_IN))
print('No. of target variables = '+str(N_VARS_OUT))
N_AUX = 3

cur_path = os.getcwd()
cur_path = os.getcwd()
pred_fld = os.listdir(cur_path+'/.predictions/')
for fld in pred_fld:
    if fld.split(".")[-1] != 'npz':
        NAME = fld
        break

if type(fl) == int:
    pred_fld = cur_path+'/.predictions/'+NAME+f'/ckpt_{ckpt:04d}/pred_fluct{fl:04d}.npz'

    data = np.load(pred_fld)

    u_test = data['Y_test']
    u_pred = data['Y_pred']
    u_input = data['X_test']
else:
    pred_fld = cur_path+'/.predictions/'+NAME+f'/ckpt_{ckpt:04d}/pred_fluct{fl[0]:04d}.npz'
    data = np.load(pred_fld)
    u_test = data['Y_test']
    u_pred = data['Y_pred']
    u_input = data['X_test']
    for i in range(1,len(fl)):
        pred_fld = cur_path+'/.predictions/'+NAME+f'/ckpt_{ckpt:04d}/pred_fluct{fl[i]:04d}.npz'
        data = np.load(pred_fld)
        u_test = np.append(u_test,data['Y_test'],0)
        u_pred = np.append(u_pred,data['Y_pred'],0)
        u_input = np.append(u_input,data['X_test'],0)

u_pred = u_pred
u_test = u_test

xd = []
u_mse = []
u_mae = []
u_mape = []
u_rms = []
u_trms = []

yd = []
v_mse = []
v_mae = []
v_mape = []
v_rms = []
v_trms = []

zd = []
w_mse = []
w_mae = []
w_mape = []
w_rms = []
w_trms = []

u_input = u_input[:,:,32:-32,32:-32]

for i in range(u_pred.shape[0]):
    i_comp = 0
    xd.append(np.mean(u_input[i,i_comp]))
    u_mse.append(np.mean((u_pred[i,i_comp]-u_test[i,i_comp])**2))
    u_mae.append(np.mean(np.abs(u_pred[i,i_comp]-u_test[i,i_comp])))
    u_mape.append(100*np.mean(np.abs((u_pred[i,i_comp]-u_test[i,i_comp])/u_test[i,i_comp])))
    u_rms.append(100*np.abs(np.sqrt(np.mean(u_pred[i,i_comp]**2))-np.sqrt(np.mean(u_test[i,i_comp]**2)))/np.sqrt(np.mean(u_test[i,i_comp]**2)))
    u_trms.append(np.sqrt(np.mean(u_test[i,i_comp]**2)))

    i_comp = 1
    yd.append(np.mean(u_input[i,i_comp]))
    v_mse.append(np.mean((u_pred[i,i_comp]-u_test[i,i_comp])**2))
    v_mae.append(np.mean(np.abs(u_pred[i,i_comp]-u_test[i,i_comp])))
    v_mape.append(100*np.mean(np.abs((u_pred[i,i_comp]-u_test[i,i_comp])/u_test[i,i_comp])))
    v_rms.append(100*np.abs(np.sqrt(np.mean(u_pred[i,i_comp]**2))-np.sqrt(np.mean(u_test[i,i_comp]**2)))/np.sqrt(np.mean(u_test[i,i_comp]**2)))
    v_trms.append(np.sqrt(np.mean(u_test[i,i_comp]**2)))

    i_comp = 2
    zd.append(np.mean(u_input[i,i_comp]))
    w_mse.append(np.mean((u_pred[i,i_comp]-u_test[i,i_comp])**2))
    w_mae.append(np.mean(np.abs(u_pred[i,i_comp]-u_test[i,i_comp])))
    w_mape.append(100*np.mean(np.abs((u_pred[i,i_comp]-u_test[i,i_comp])/u_test[i,i_comp])))
    w_rms.append(100*np.abs(np.sqrt(np.mean(u_pred[i,i_comp]**2))-np.sqrt(np.mean(u_test[i,i_comp]**2)))/np.sqrt(np.mean(u_test[i,i_comp]**2)))
    w_trms.append(np.sqrt(np.mean(u_test[i,i_comp]**2)))

xd = np.array(xd)
yd = np.array(yd)
zd = np.array(zd)

u_rms = np.array(u_rms)
v_rms = np.array(v_rms)
w_rms = np.array(w_rms)

u_trms = np.array(u_trms)
v_trms = np.array(v_trms)
w_trms = np.array(w_trms)

u_mse = np.array(u_mse)
v_mse = np.array(v_mse)
w_mse = np.array(w_mse)

u_mae = np.array(u_mae)
v_mae = np.array(v_mae)
w_mae = np.array(w_mae)

u_mape = np.array(u_mape)
v_mape = np.array(v_mape)
w_mape = np.array(w_mape)

np.savez('classify.npz',xd=xd,yd=yd,zd=zd,u_rms=u_rms,v_rms=v_rms,w_rms=w_rms,u_mse=u_mse,v_mse=v_mse,w_mse=w_mse,u_mae=u_mae,v_mae=v_mae,w_mae=w_mae,u_mape=u_mape,v_mape=v_mape,w_mape=w_mape,u_trms=u_trms,v_trms=v_trms,w_trms=w_trms)
