import numpy as np
import time
from sampling import *
from mocu_comp import *
from MOCU import *
import numpy as np

def find_Entropy_seq(MOCU_matrix, save_f_inv, D_save, init_MOCU_val, K_max, w, N, h , M, T,
                  a_lower_bound_update, a_upper_bound_update, it_idx, update_cnt):

    a_diff = np.zeros((N,N))

    Entropy_seq = np.ones(update_cnt)*50.0
    it_temp_val = np.zeros(it_idx)
    it_temp_val_init = np.zeros(it_idx)

    Entropy_seq[0] = init_MOCU_val




    a_diff = np.triu(a_upper_bound_update - a_lower_bound_update,1)
    #print(a_diff)
    for ij in range(1,update_cnt):
        flag = 0

        max_ind = np.where(a_diff == np.max(a_diff[np.nonzero(a_diff)]))
        if len(max_ind[0]) == 1:
            i = int(max_ind[0])
            j = int(max_ind[1])
        else:
            i = int(max_ind[0][0])
            j = int(max_ind[1][0])
        a_diff[i, j] = 0.0

        #print(i,j)


        f_inv = save_f_inv[i, j]

        if D_save[i, j] == 0.0:
            a_upper_bound_update[i, j] \
                = min(a_upper_bound_update[i, j], f_inv)
            a_upper_bound_update[j, i] \
                = min(a_upper_bound_update[i, j], f_inv)
            if f_inv > a_upper_bound_update[i, j]:
                flag = 1

        else:
            a_lower_bound_update[i, j] \
                = max(a_lower_bound_update[i, j], f_inv)
            a_lower_bound_update[j, i] \
                = max(a_lower_bound_update[i, j], f_inv)
            if f_inv < a_lower_bound_update[i, j]:
                flag = 1



        cnt = 0
        while Entropy_seq[ij] > Entropy_seq[ij - 1]:
            if ij > update_cnt-3:
                K_max = K_max

            for l in range(it_idx):
                it_temp_val[l] = MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update)
            Entropy_seq[ij] = np.median(it_temp_val)
            cnt = cnt + 1
            if cnt == 2:
                Entropy_seq[ij] = Entropy_seq[ij - 1]
                break


    print("Entropy_lower = ", a_lower_bound_update)
    print("Entropy_upper = ", a_upper_bound_update)

    return Entropy_seq
