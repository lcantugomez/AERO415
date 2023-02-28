import numpy as np
import matplotlib.pyplot as plt


def create_alg_grid(i_max,j_max,xL,xR,yL,yU):

    # Generate exi and eta arr
    exi_arr = []
    eta_arr = []

    for i in range(0,i_max+1):
        exi_arr.append(i/i_max)
    
    for j in range(0,j_max+1):
        eta_arr.append(j/j_max)

    x_mat = np.zeros((i_max+1,j_max+1))
    y_mat = np.zeros((i_max+1,j_max+1))

    # Generate x and y points using bounds
    for j in range(len(eta_arr)):
        for i in range(len(exi_arr)):
            x = (xL + (exi_arr[i]*(xR-xL)))
            y = yL(x) + (eta_arr[j]*(yU(x) - yL(x)))
            x_mat[i,j] = x
            y_mat[i,j] = y
    

    return x_mat, y_mat

