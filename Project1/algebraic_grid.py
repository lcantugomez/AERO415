import numpy as np
import matplotlib.pyplot as plt


def create_alg_grid(i_max,j_max,xL,xR,yL,yU):

    # Generate exi and eta arr
    exi_arr = []
    eta_arr = []

    for i in range(0,i_max):
        exi_arr.append(i/(i_max-1))
    
    for j in range(0,j_max):
        eta_arr.append(j/(j_max-1))

    # Create matrices to store x and y values
    x_mat = np.zeros((i_max,j_max))
    y_mat = np.zeros((i_max,j_max))

    # Generate x and y points using bounds
    for j in range(len(eta_arr)):
        for i in range(len(exi_arr)):
            x = (xL + (exi_arr[i]*(xR-xL)))
            y = yL(x) + (eta_arr[j]*(yU(x) - yL(x)))
            x_mat[i,j] = x
            y_mat[i,j] = y
    

    return x_mat, y_mat