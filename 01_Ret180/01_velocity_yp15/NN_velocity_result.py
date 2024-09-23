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
fl = 0 # [0,1]  ## pred_fluct{fl}.npz to be read

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
"""
a = os.listdir(cur_path+'/.epoch_log/')
for i in a:
    if i.split('.')[-1] == 'npz':
        file = cur_path+'/.epoch_log/'+i
data = np.load(file)
tLoss = data['tLoss']
vLoss = data['vLoss']
epoch = np.array(range(1,len(tLoss)+1),dtype='int')
fig = plt.figure(0)
axu = fig.add_subplot(111)
p1_uu = axu.plot(epoch,tLoss,'-', linewidth=2,markersize=mrk_size,color='C1')

p1_uu = axu.plot(epoch,vLoss,'-o', linewidth=2,markersize=mrk_size,color='C1',linestyle='dotted')
plt.ylim(vLoss[-1]/1.5,vLoss[0])
plt.ylim(None,None)
#plt.xlim(0,len(tLoss)+1)

legend_elements = [Line2D([0], [0], color='C1', label='Training loss'),Line2D([4],[0],color='C1',marker='o',markersize=3.5,linestyle='dotted',label='Validation loss')]
legend1=plt.legend(handles=legend_elements,loc='best',fontsize=12)
axu.add_artist(legend1)
plt.xlabel('Epoch')
plt.ylabel(r'$\mathcal{L}(\mathbf{u}_\mathrm{FCN};\mathbf{u}_\mathrm{DNS})$')
fig.savefig('./.jupyter_plots/'+f'epoch.pdf')
"""

stat_mse = dict()
stat_rms = dict()
stat_rms_err = dict()
tstamp = dict()
actual_rms = dict()
req_loss = dict()
eplus_rms = dict()
dns_rms_plus = dict()
fcn_rms_plus = dict()

stat_mse[f'yp{fyp}'] = list()
stat_rms[f'yp{fyp}'] = list()
stat_rms_err[f'yp{fyp}'] = list()
actual_rms[f'yp{fyp}'] = list()
req_loss[f'yp{fyp}'] = list()
eplus_rms[f'yp{fyp}'] = list()
dns_rms_plus[f'yp{fyp}'] = list()
fcn_rms_plus[f'yp{fyp}'] = list()
tstamp[f'yp{fyp}'] = list()

ypos_Ret = {'1':0, '15':1, '30':2, '50':3, '310':4, '330':5, '345':6, '359':7}

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

def error():

  avgs_in = np.reshape(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_IN[0]+'.m').astype(np.float32)[:],(1,8))
  for i in range(1,app.N_VARS_IN):
      avgs_in = np.append(avgs_in, np.reshape(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_IN[i]+'.m').astype(np.float32)[:],(1,8)),0)

  std_in = np.reshape(np.loadtxt(avg_path+app.VARS_NAME_IN[0]+'_rms.m').astype(np.float32)[:],(1,8))
  for i in range(1,app.N_VARS_IN):
      std_in = np.append(std_in, np.reshape(np.loadtxt(avg_path+app.VARS_NAME_IN[i]+'_rms.m').astype(np.float32)[:],(1,8)),0)

  avgs = np.reshape(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_OUT[0]+'.m').astype(np.float32)[:],(1,8))
  for i in range(1,app.N_VARS_OUT):
      avgs = np.append(avgs, np.reshape(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_OUT[i]+'.m').astype(np.float32)[:],(1,8)),0)

  rms = np.reshape(np.loadtxt(avg_path+app.VARS_NAME_OUT[0]+'_rms.m').astype(np.float32)[:],(1,8))
  for i in range(1,app.N_VARS_OUT):
      rms = np.append(rms, np.reshape(np.loadtxt(avg_path+app.VARS_NAME_OUT[i]+'_rms.m').astype(np.float32)[:],(1,8)),0)

  ms = []
  rm_std = []
  ms_norm = []

  for i_comp in range(int(app.N_VARS_OUT)):

      pred_mean = np.mean(u_pred[:, i_comp])
      true_mean = 0.5*(avgs[i_comp,ypos_Ret[str(target_yp)]]+avgs[i_comp,ypos_Ret[str(360-target_yp)]])

      pred_rms = np.sqrt(np.mean(u_pred[:, i_comp]**2))
      act_rms = 0.5*(rms[i_comp,ypos_Ret[str(target_yp)]]+rms[i_comp,ypos_Ret[str(360-target_yp)]])

      mse = np.mean((u_pred[:, i_comp] - u_test[:, i_comp])**2)
      ms.append(mse)
      print(f'The MSE value {VARS_NAME_OUT[i_comp]} is: {mse:e}')
    
      mse_norm = mse/act_rms/act_rms
      print(mse_norm)
      ms_norm.append(mse_norm)

  rm = []
  for i_comp in range(app.N_VARS_OUT):
      act_rms = rms[i_comp,ypos_Ret[str(target_yp)]]
      pred_rms = np.sqrt(np.mean(u_pred[:,i_comp]**2))
      true_rms = np.sqrt(np.mean(u_test[:,i_comp]**2))
      err = 100*abs(pred_rms-act_rms)/act_rms
      err1 = 100*abs(pred_rms-true_rms)/true_rms
      act_mean_rms = 0.5*(rms[i_comp,ypos_Ret[str(target_yp)]]+rms[i_comp,ypos_Ret[str(360-target_yp)]])
      err2 = 100*abs(pred_rms-act_mean_rms)/act_mean_rms
      rm.append(err2)

      print(f'Mean of {VARS_NAME_OUT[i_comp]}. RMS error: {err:.2f} %  RMS dat error: {err1:.2f} % RMS mean dat error: {err2:.2f} %')
    
  np.savetxt(f'mse.dat', ms, fmt='%1.4e')
  np.savetxt(f'mse_norm.dat',ms_norm,fmt='%1.4e')
  np.savetxt(f'rms.dat', rm, fmt='%1.4e')

  rmsg = []
  for i_comp in range(app.N_VARS_OUT):
      rmse = []
      for i_samp in range(u_pred.shape[0]):
          rmse.append(100*np.abs(np.sqrt(np.mean(u_pred[i_samp,i_comp]**2))-np.sqrt(np.mean(u_test[i_samp,i_comp]**2)))/np.sqrt(np.mean(u_test[i_samp,i_comp]**2)))
      rmsg.append(np.mean(rmse))
      print(f'RMSE: {rmsg}')

  rmsh = []
  for i_comp in range(app.N_VARS_OUT):
      mm = np.mean(u_pred[:,i_comp])
      rmse = []
      for i_samp in range(u_pred.shape[0]):
          rmse.append(100*np.abs(np.sqrt(np.mean((u_pred[i_samp,i_comp]-mm)**2))-np.sqrt(np.mean((u_test[i_samp,i_comp]-mm)**2)))/np.sqrt(np.mean((u_test[i_samp,i_comp]-mm)**2)))
      rmsh.append(np.mean(rmse))
      print(f'RMSEE: {rmsh}')
  rmsi = [rmsg,rmsh]
  np.savetxt('RMS_error_file.txt',rmsi,fmt='%1.4e')

  corre = []
  corri = []
  for i in range(app.N_VARS_OUT):
      corr = np.mean(u_pred[:,i]*u_test[:,i])/(np.sqrt(np.var(u_pred[:,i]))*np.sqrt(np.var(u_test[:,i])))
      corre.append(corr)
  print(corre)

  np.savetxt('Correlation.txt',corre,fmt='%1.4e')

def best_samples():
    for i_comp in range(int(app.N_VARS_OUT)):
        bse=[]
        for j in range(u_pred.shape[0]):
            bse.append(np.mean((u_test[j,i_comp]-u_pred[j,i_comp])**2))
        print(i_comp,np.argsort(np.array(bse))[:3])

def contour():

  i_sample = [100,908,1000,3000]
  np.savez('qual_plot_yp15.npz',X=u_input[i_sample],Y=u_test[i_sample],Z=u_pred[i_sample])

def spectra(val,quant):

  Retau = 180
  Rebulk = 2800
  utau = Retau/Rebulk
  ltau = 1/Retau

  Lx = 6
  Lz = 3
  Nx = 432
  Nz = 432
  kz = np.linspace(0,int(Nz/2),int(Nz/2)+1)*2*math.pi/Lz
  kx = np.linspace(0,int(Nx/2),int(Nx/2)+1)*2*math.pi/Lx

  lambda_z = 2*math.pi/kz
  lambda_x = 2*math.pi/kx

  dkz = kz[1] - kz[0]
  dkx = kx[1] - kx[0]

  spec_xzb = np.zeros((u_test.shape[0],Nz,Nx),dtype='complex')
  for i_t in range(u_test.shape[0]):
      ff_1 = np.fft.fft2(u_test[i_t,val,:,:],(Nz,Nx),axes=(0,1))/(Nx*Nz)
      ff_1 = ff_1*np.conj(ff_1)
      spec_xzb[i_t,:,:] = ff_1
  ps_2_xz = np.ndarray((u_test.shape[0],int(Nz/2)+1,Nx),dtype='complex')
  ps_2d_xz = np.ndarray((int(Nz/2)+1,int(Nx/2)+1),dtype='float')

  ps_2_xz[:,:int(Nz/2)+1,:] = spec_xzb[:,:int(Nz/2)+1,:]
  ps_2_xz[:,1:int(Nz/2),:] = spec_xzb[:,1:int(Nz/2),:] + spec_xzb[:,Nz-1:int(Nz/2):-1,:]

  ps_2_xz[:,:,1:int(Nx/2)] = ps_2_xz[:,:,1:int(Nx/2)] + ps_2_xz[:,:,Nx-1:int(Nx/2):-1]

  ps_2d_xz = np.mean(np.real(ps_2_xz[:,:int(Nz/2)+1,:int(Nx/2)+1]),axis=0)

  kzt = np.ndarray((kz.shape[0],1),dtype='float')
  kxt = np.ndarray((kx.shape[0],1),dtype='float')
  kzt[:,0] = kz
  kxt[:,0] = kx
  coeff_xz = np.dot(kzt,np.transpose(kxt))

  pre_ps_2d_xz = np.ndarray((int(Nz/2)+1,int(Nx/2)+1),dtype='float')

  pre_ps_2d_xz = (ps_2d_xz/dkz/dkx)*coeff_xz

##

  spec_xzb_pred = np.zeros((u_pred.shape[0],Nz,Nx),dtype='complex')
  for i_t in range(u_pred.shape[0]):
      ff_1 = np.fft.fft2(u_pred[i_t,val,:,:],(Nz,Nx),axes=(0,1))/(Nx*Nz)
      ff_1 = ff_1*np.conj(ff_1)
      spec_xzb_pred[i_t,:,:] = ff_1
  ps_2_xz_pred = np.ndarray((u_pred.shape[0],int(Nz/2)+1,Nx),dtype='complex')
  ps_2d_xz_pred = np.ndarray((int(Nz/2)+1,int(Nx/2)+1),dtype='float')

  ps_2_xz_pred[:,:int(Nz/2)+1,:] = spec_xzb_pred[:,:int(Nz/2)+1,:]
  ps_2_xz_pred[:,1:int(Nz/2),:] = spec_xzb_pred[:,1:int(Nz/2),:] + spec_xzb_pred[:,Nz-1:int(Nz/2):-1,:]

  ps_2_xz_pred[:,:,1:int(Nx/2)] = ps_2_xz_pred[:,:,1:int(Nx/2)] + ps_2_xz_pred[:,:,Nx-1:int(Nx/2):-1]

  ps_2d_xz_pred = np.mean(np.real(ps_2_xz_pred[:,:int(Nz/2)+1,:int(Nx/2)+1]),axis=0)

  pre_ps_2d_xz_pred = np.ndarray((int(Nz/2)+1,int(Nx/2)+1),dtype='float')

  pre_ps_2d_xz_pred = (ps_2d_xz_pred/dkz/dkx)*coeff_xz

  xd = lambda_x[1:]/ltau
  yd = lambda_z[1:]/ltau
  dd = (ps_2d_xz[1:,1:]*coeff_xz[1:,1:]/dkz/dkx/utau**2)/np.max(ps_2d_xz[1:,1:]*coeff_xz[1:,1:]/dkz/dkx/utau**2)
  ddt = dd

  dd = (ps_2d_xz_pred[1:,1:]*coeff_xz[1:,1:]/dkz/dkx/utau**2)/np.max(ps_2d_xz_pred[1:,1:]*coeff_xz[1:,1:]/dkz/dkx/utau**2)
  ddp = dd

  np.savez(f'spectra_{quant}_yp{target_yp}.npz',xd=xd,yd=yd,ddt=ddt,ddp=ddp)

error()
best_samples()
contour()
spectra(0,'u')
spectra(1,'v')
spectra(2,'w')
