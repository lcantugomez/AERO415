import numpy as np
import matplotlib.pyplot as plt

def coeff_calc(i_max,j_max,x_mat,y_mat,d_exi,d_eta):
    # Initialize alpha, beta, gamma arrays

    alpha_arr = []
    beta_arr = []
    gamma_arr = []

    for j in range(1,j_max):
        for i in range(1,i_max):
            # Conditional to only calculate for interior nodes
                

            alpha  = (1/(4*(d_eta**2))) * ( ((x_mat[i,j+1] - x_mat[i,j-1])**2) + ((y_mat[i,j+1] - y_mat[i,j-1]**2)) )
            beta = (1/(4*(d_eta*d_exi))) * ( ((x_mat[i+1,j] - x_mat[i-1,j]) * (x_mat[i,j+1] - x_mat[i,j-1])) + ((y_mat[i+1,j] - y_mat[i-1,j]) * (y_mat[i,j+1] - y_mat[i,j-1])) )
            gamma = (1/(4*(d_exi**2))) * ( ((x_mat[i+1,j] - x_mat[i-1,j])**2) + ((y_mat[i+1,j] - y_mat[i-1,j]**2)) )

            alpha_arr.append(alpha)
            beta_arr.append(beta)
            gamma_arr.append(gamma)
    return alpha_arr, beta_arr, gamma_arr

