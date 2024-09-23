import numpy as np
import os

ckpt = 41
fl = [0,1]
i_sample = [100,400,500,1000,1500,1800,2100,2600,2900,3200,3500,4000,4500,5000,5500,6000,6400] 

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

np.savez('qual_plot_new.npz',X=u_input[i_sample],Y=u_test[i_sample],Z=u_pred[i_sample])
