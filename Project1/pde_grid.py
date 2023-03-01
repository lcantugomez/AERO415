import numpy as np
import matplotlib.pyplot as plt


def coeff_calc(i_max,j_max,x_mat,y_mat,d_exi,d_eta):
    # Initialize alpha, beta, gamma arrays

    alpha_mat = np.zeros((i_max-1,j_max-1))
    beta_mat = np.zeros((i_max-1,j_max-1))
    gamma_mat = np.zeros((i_max-1,j_max-1))

    for j in range(1,j_max):            # Loop only through the interior nodes
        for i in range(1,i_max):
                

            alpha  = (1/(4*(d_eta**2))) * ( ((x_mat[i,j+1] - x_mat[i,j-1])**2) + ((y_mat[i,j+1] - y_mat[i,j-1]**2)) )
            beta = (1/(4*(d_eta*d_exi))) * ( ((x_mat[i+1,j] - x_mat[i-1,j]) * (x_mat[i,j+1] - x_mat[i,j-1])) + ((y_mat[i+1,j] - y_mat[i-1,j]) * (y_mat[i,j+1] - y_mat[i,j-1])) )
            gamma = (1/(4*(d_exi**2))) * ( ((x_mat[i+1,j] - x_mat[i-1,j])**2) + ((y_mat[i+1,j] - y_mat[i-1,j]**2)) )


            alpha_mat[i-1,j-1] = alpha
            beta_mat[i-1,j-1] = beta
            gamma_mat[i-1,j-1] = gamma

    return alpha_mat, beta_mat, gamma_mat


def pde_grid(alpha_mat,beta_mat,gamma_mat,x_mat,y_mat,d_exi,d_eta,tol,i_max,j_max):

    diff = 1
    while diff >= tol:
        for j in range(0,j_max+1):
            for i in range(0,i_max+1):
                if i == 0:
                    if j == 0:
                        pass
                    elif j == j_max:
                        pass
                    else:
                        x_mat[i,j] = x_mat[i+1,j]
                elif i == i_max:
                    if j == 0:
                        pass
                    elif j == j_max:
                        pass
                    else:
                        x_mat[i,j] = x_mat[i-1,j]
                elif j == 0:
                    print("Bottom wall", i, " ",j)
                    if 2 < x_mat[i,j] < 3:
                        print("On the curve ",x_mat[i,j])
                    else:
                        print("Off the curve ", x_mat[i,j])
                elif j == j_max:
                    pass
                else:
                    i_coeff = i - 1
                    j_coeff = j - 1
                    print("inner node",i, "  ", j)
                    
        diff = 10**-10

    pass