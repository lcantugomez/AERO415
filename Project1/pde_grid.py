import numpy as np
import matplotlib.pyplot as plt


class PDE_Grid():

    # Init class with set of parameters
    def __init__(self,params) -> None:      
        for kw,arg in params.items():
            setattr(PDE_Grid,kw,arg)
    


    def coeff_calc(self):

        i_max = self.i_max
        j_max = self.j_max      # Init variables to use in function (is in this format due to legacy code)
        x_mat = self.x_mat
        y_mat = self.y_mat
        d_exi = self.d_exi
        d_eta = self.d_eta

        # Initialize alpha, beta, gamma arrays

        alpha_mat = np.zeros((i_max-1,j_max-1))
        beta_mat = np.zeros((i_max-1,j_max-1))
        gamma_mat = np.zeros((i_max-1,j_max-1))

        for j in range(1,j_max):            # Loop only through the interior nodes
            for i in range(1,i_max):
                    
                

                # Calc the alpha beta and gamma vals

                alpha  = (1/(4*(d_eta**2))) * ( ((x_mat[i,j+1] - x_mat[i,j-1])**2) + ((y_mat[i,j+1] - y_mat[i,j-1]**2)) )
                beta = (1/(4*(d_eta*d_exi))) * ( ((x_mat[i+1,j] - x_mat[i-1,j]) * (x_mat[i,j+1] - x_mat[i,j-1])) + ((y_mat[i+1,j] - y_mat[i-1,j]) * (y_mat[i,j+1] - y_mat[i,j-1])) )
                gamma = (1/(4*(d_exi**2))) * ( ((x_mat[i+1,j] - x_mat[i-1,j])**2) + ((y_mat[i+1,j] - y_mat[i-1,j]**2)) )



                # Adjust indices to use in matrix (python uses index 0, notes use index 1)

                alpha_mat[i-1,j-1] = alpha
                beta_mat[i-1,j-1] = beta
                gamma_mat[i-1,j-1] = gamma

        # Init mats into class

        self.alpha_mat = alpha_mat
        self.beta_mat = beta_mat
        self.gamma_mat = gamma_mat


        # Return variables for debugging
        return alpha_mat, beta_mat, gamma_mat


    def pde_grid(self,tol):

        alpha_mat = self.alpha_mat
        beta_mat = self.beta_mat
        gamma_mat = self.gamma_mat
        i_max = self.i_max
        j_max = self.j_max              # Init variables to use in function (is in this format due to legacy code)
        x_mat = self.x_mat
        y_mat = self.y_mat
        d_exi = self.d_exi
        d_eta = self.d_eta

        
        diff = 1
        while diff >= tol:
            diff += -1
            for j in range(0,j_max+1):
                for i in range(0,i_max+1):

                    if i == 0:
                        if j == 0:
                            pass            # Bottom left corner
                        elif j == j_max:
                            pass            # Top left corner
                        else:
                            x_mat[i,j] = x_mat[i+1,j]          # Left wall
                            y_mat[i,j] = y_mat[i+1,j]

                    elif i == i_max:
                        if j == 0:
                            pass            # Bottom right corner
                        elif j == j_max:
                            pass            # Top right corner
                        else:
                            x_mat[i,j] = x_mat[i-1,j]       # Right wall
                            y_mat[i,j] = y_mat[i-1,j]
                    
                    elif j == 0:
                        x_mat[i,j] = x_mat[i,j+2] * (2 * d_eta * self.__class__.yp_lower(x_mat[i,j]))     # On the bottom curve
                        y_mat[i,j] = self.__class__.y_lower(x_mat[i,j-2])
                        
                    elif j == j_max:
                        x_mat[i,j] = x_mat[i,j-2] * (2 * d_eta * self.__class__.yp_upper(x_mat[i,j]))     # On the upper curve
                        y_mat[i,j] = self.__class__.y_upper(x_mat[i,j-2])
                        
                    else:
                        i_coeff = i - 1
                        j_coeff = j - 1
                        
                        phi = (2*self.alpha_mat[i_coeff,j_coeff]/(self.d_exi**2)) + (2*self.gamma_mat[i_coeff,j_coeff]/(self.d_eta**2))

                        a1 = -(self.beta_mat[i_coeff,j_coeff]/(2*self.d_exi*self.d_eta)) / phi
                        a2 = -(self.gamma_mat[i_coeff,j_coeff]/(2*self.d_eta**2)) / phi
                        a3 = -a1
                        a4 = -(self.alpha_mat[i_coeff,j_coeff]/(2*self.d_exi**2)) / phi
                        a5 = a4
                        a6 = a3
                        a7 = a2
                        a8 = a1

                        x_mat[i,j] = a1*x_mat[i-1,j] + a2*x_mat[i,j-1] + a3*x_mat[i+1,j-1] + a4*x_mat[i-1,j] + a5*x_mat[i+1,j] + a6*x_mat[i-1,j+1] + a7*x_mat[i,j+1] + a8*x_mat[i+1,j+1]

                        y_mat[i,j] = a1*y_mat[i-1,j] + a2*y_mat[i,j-1] + a3*y_mat[i+1,j-1] + a4*y_mat[i-1,j] + a5*y_mat[i+1,j] + a6*y_mat[i-1,j+1] + a7*y_mat[i,j+1] + a8*y_mat[i+1,j+1]

        return x_mat,y_mat