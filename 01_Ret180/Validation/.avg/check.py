import numpy as np
a = np.loadtxt('v_rms.m')
v = 0.5*(a[1]+a[-2])
b = np.loadtxt('w_rms.m')
w = 0.5*(b[1]+ b[-2])
c = np.loadtxt('u_rms.m')
u = 0.5*(c[1]+c[-2])
print(v+w+u)
a = np.loadtxt('v_rms.m')
v = 0.5*(a[2]+a[-3])
b = np.loadtxt('w_rms.m')
w = 0.5*(b[2]+ b[-3])
c = np.loadtxt('u_rms.m')
u = 0.5*(c[2]+c[-3])
print(v+w+u)
a = np.loadtxt('v_rms.m')
v = 0.5*(a[3]+a[-4])
b = np.loadtxt('w_rms.m')
w = 0.5*(b[3]+ b[-4])
c = np.loadtxt('u_rms.m')
u = 0.5*(c[3]+c[-4])
print(v+w+u)
