import numpy as np
a = np.load('/mimer/NOBACKUP/groups/kthmech/argb/02_VE/n25/01_velocity_yp15/.predictions/NN_WallReconfluct1TF2_3NormIn-3Out_1-15_432x432_Ret180_lr0.001_decay20drop0.5_relu-1705416965/ckpt_0041/pred_fluct0000.npz')['X_test']
print(np.mean(a[100,0,32:-32,32:-32]))
