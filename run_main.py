#!/usr/bin/env python2.7

#
# conda activate py3
# run cuda in the server.
#
# Youngjoon Hong (05.15.2020)
# !!!!!!!!!!!!!!!!!!!!Important!!!!!!!!!!!!!!!!!!!!!!!
# If you want to chage the number of oscillators, Change "N" here.
# In addition, please do not forget update manually the followings in the MOCU.py file
# #define N_global 10
# 


import sys
sys.path.append("./src")
#from mocu_comp import *
from find_MOCU_seq import *
from find_Entropy_seq import *
from find_Rand_seq import *
from MOCU import *
from main_module_ij_couple import *
import time

#import scipy.fftpack as fftp

#import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
#from mlplot import *
#from scipy.sparse.linalg import dsolve
#import scipy.sparse.linalg as spla

print("################################################")
print("Test code")
print("\n")




print("Start ticking")
tt = time.time()

it_idx = 10
update_cnt = 27
# Number of equations
N = 9
K_max = 10000#10000

# Final Time
T = 4.0

# Time discretization
h = 1.0/160.0

# Time steps
M = int(T/h)
w = np.zeros(N)

w[0] = 1.19
w[1] = 3.23
w[2] = 6.34
w[3] = 7.48
w[4] = 10.9
w[5] = 11.62
w[6] = 14.74
w[7] = 25.58
w[8] = 34.88




a_upper_bound = np.zeros((N,N))
a_lower_bound = np.zeros((N,N))
a_lower_bound_update = np.zeros((N,N))
a_upper_bound_update = np.zeros((N,N))
a = np.zeros((N,N))
tata = np.zeros((N,N))


for i in range(N):
    for j in range(i+1,N):
        if (i <= 4) and (j <= 4):
            f_inv = np.abs(w[i] - w[j])/2.0
            a_upper_bound[i,j] = f_inv*1.01
            a_lower_bound[i,j] = f_inv*0.99
            a_upper_bound[j, i] = a_upper_bound[i,j]
            a_lower_bound[j, i] = a_lower_bound[i,j]
        elif (i >= 7) or (j >= 7):
            f_inv = np.abs(w[i] - w[j])/9.0
            a_upper_bound[i,j] = f_inv*1.01
            a_lower_bound[i,j] = f_inv*0.99
            a_upper_bound[j, i] = a_upper_bound[i,j]
            a_lower_bound[j, i] = a_lower_bound[i,j]
        else:
            f_inv = np.abs(w[i] - w[j])/2.0
            a_upper_bound[i,j] = f_inv*1.01
            a_lower_bound[i,j] = f_inv*0.99
            a_upper_bound[j, i] = a_upper_bound[i,j]
            a_lower_bound[j, i] = a_lower_bound[i,j]
        if j == 5:
            f_inv = np.abs(w[i] - w[j])/2.0
            a_upper_bound[i,j] = f_inv*1.01
            a_lower_bound[i,j] = f_inv*0.99
            a_upper_bound[j, i] = a_upper_bound[i,j]
            a_lower_bound[j, i] = a_lower_bound[i,j]

tata = a_upper_bound - a_lower_bound

r1 = 3.0
r2 = 3.0
r3 = 3.0

a_lower_bound_update = a_lower_bound.copy()
a_upper_bound_update = a_upper_bound.copy()


for i in range(N):
    for j in range(N):
        if i==2:
            a[i,j] = a_lower_bound[i,j] + 0.5*(a_upper_bound[i,j] - a_lower_bound[i,j])
        elif i==4:
            a[i,j] = a_lower_bound[i, j] + r2/6.0*(a_upper_bound[i, j] - a_lower_bound[i,j])
        elif i==6:
            a[i,j] = a_lower_bound[i, j] + r3/6.0 *(a_upper_bound[i,j] - a_lower_bound[i,j])
        else:
            a[i,j] = a_lower_bound[i, j] + r1/6.0*(a_upper_bound[i,j] - a_lower_bound[i,j])
        a[j,i] = a[i,j]



init_sync_check = mocu_comp(w, h, N, M, a)

if init_sync_check == 1:
    print('It is already sync system!!!!')


"""
for i in range(5):
    toto = MOCU(K_max, w, N, h , M, T, a_lower_bound, a_upper_bound)
    print("MOCU: ",toto)

sys.exit()
"""

MOCU_matrix = np.zeros((N,N))
MOCU_matrix_syn = np.zeros((N,N))
MOCU_matrix_nonsyn = np.zeros((N,N))
R = np.zeros((N,N))
R_copy = np.zeros((N,N))
D_save = np.zeros((N,N))
save_f_inv = np.zeros((N,N))
it_temp_val = np.zeros(it_idx)


for i in range(N):
    for j in range(i+1,N):

        w_i = w[i]
        w_j = w[j]
        a_ij = a[i,j]
        f_inv = 0.5*np.abs(w_i - w_j)
        save_f_inv[i,j] = f_inv

        a_lower_bound_update[i,j] = f_inv
        a_lower_bound_update[j,i] = f_inv

        a_lower_bound_update[j,i] = max(f_inv, a_lower_bound_update[i,j])
        a_lower_bound_update[j,i] = max(f_inv, a_lower_bound_update[i,j])

        a_tilde = min(max(f_inv, a_lower_bound[i, j]), a_upper_bound[i, j])
        P_syn = (a_upper_bound[i, j] - a_tilde)/(a_upper_bound[i, j] - a_lower_bound[i, j])

        for l in range(it_idx):
            it_temp_val[l] = MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update)
            print("MOCU value sync: ", it_temp_val[l])

        MOCU_matrix_syn[i,j] = np.median(it_temp_val)
        print("median = ", np.median(it_temp_val))

        a_lower_bound_update = a_lower_bound.copy()
        a_upper_bound_update = a_upper_bound.copy()


        a_upper_bound_update[i,j] = f_inv
        a_upper_bound_update[j,i] = f_inv

        a_upper_bound_update[i, j] = min(f_inv, a_upper_bound_update[i, j])
        a_upper_bound_update[j, i] = min(f_inv, a_upper_bound_update[i, j])

        P_nonsyn = (a_tilde - a_lower_bound[i,j])/(a_upper_bound[i,j] - a_lower_bound[i,j])


        for l in range(it_idx):
            it_temp_val[l] = MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update)
            #print("MOCU value non-sync: ", it_temp_val[l])

        MOCU_matrix_nonsyn[i,j] = np.median(it_temp_val)
        #print("median = ", np.median(it_temp_val))

        a_lower_bound_update = a_lower_bound.copy()
        a_upper_bound_update = a_upper_bound.copy()

        R[i, j] = P_syn*MOCU_matrix_syn[i, j] + P_nonsyn*MOCU_matrix_nonsyn[i, j]

        D = main_module_ij_couple(w_i, w_j, h, N, M, a_ij)
        if D == 1:
            MOCU_matrix[i, j] = MOCU_matrix_syn[i, j]

        else:
            MOCU_matrix[i, j] = MOCU_matrix_nonsyn[i, j]

        D_save[i,j] = D

        print("i = ",i)
        print("R = ",R)



np.savetxt('R.csv', R, delimiter=',')
np.savetxt('a_lower.csv', a_lower_bound_update, delimiter=',')
np.savetxt('a_upper.csv', a_upper_bound_update, delimiter=',')

# R = np.loadtxt('R.csv', delimiter=',')
# a_lower_bound_update = np.loadtxt('a_lower.csv', delimiter=',')
# a_upper_bound_update = np.loadtxt('a_upper.csv', delimiter=',')

MOCU_seq = np.zeros(update_cnt)
Entropy_seq = np.zeros(update_cnt)
Rand_seq = np.zeros(update_cnt)
R_copy = R.copy()

init_MOCU_val = MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update)
Entropy_seq = find_Entropy_seq(R_copy, save_f_inv, D_save, init_MOCU_val,
                                  K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update, it_idx, update_cnt)
print("Entropy",Entropy_seq)
a_lower_bound_update = a_lower_bound.copy()
a_upper_bound_update = a_upper_bound.copy()
R_copy = R.copy()
Rand_seq = find_Rand_seq(R_copy, save_f_inv, D_save, init_MOCU_val,
                                  K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update, it_idx, update_cnt)
print("Rand",Rand_seq)
a_lower_bound_update = a_lower_bound.copy()
a_upper_bound_update = a_upper_bound.copy()
R_copy = R.copy()
MOCU_seq = find_MOCU_seq(R_copy, save_f_inv, D_save, init_MOCU_val,
                                  K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update, it_idx, update_cnt)

print("MOCU",MOCU_seq)





print("time: ", time.time() - tt)

x_ax = np.arange(0,update_cnt,1)
plt.plot(x_ax, MOCU_seq, 'ro-', x_ax, Rand_seq,'bs-', x_ax, Entropy_seq, 'g^-')
plt.xlabel('Number of updates')
plt.ylabel('MOCU')
plt.legend(['MOCU based','Rand based','Entropy based'])
plt.title('MOCU Experiment N=9')
plt.grid(True)
plt.savefig("fig.eps",format='eps')
plt.show()



