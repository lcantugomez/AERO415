import numpy as np
import matplotlib.pyplot as plt
from plotting_func import plot_grid
import imageio.v2 as imageio
import os

class PDE_Grid():

    # Init class with set of parameters
    def __init__(self,params) -> None:      
        for kw,arg in params.items():
            setattr(PDE_Grid,kw,arg)


    def coeff_calc(self):

        i_max = self.i_max
        j_max = self.j_max      # Init variables to use in function (is in this format due to legacy code)
        nx_mat = self.x_mat
        ny_mat = self.y_mat
        d_exi = self.d_exi
        d_eta = self.d_eta

        # Initialize alpha, beta, gamma arrays

        alpha_mat = np.zeros((i_max,j_max))
        beta_mat = np.zeros((i_max,j_max))
        gamma_mat = np.zeros((i_max,j_max))

        for j in range(1,j_max-1):            # Loop only through the interior nodes
            for i in range(1,i_max-1):
                    
                

                # Calc the alpha beta and gamma vals
                
                alpha  = ((1/4)*((j_max-1)**2)) * ( ((nx_mat[i,j+1] - nx_mat[i,j-1])**2) + ((ny_mat[i,j+1] - ny_mat[i,j-1])**2) )
                beta = ((1/4)*((j_max-1)*(i_max-1))) * ( ((nx_mat[i+1,j] - nx_mat[i-1,j]) * (nx_mat[i,j+1] - nx_mat[i,j-1])) + ((ny_mat[i+1,j] - ny_mat[i-1,j]) * (ny_mat[i,j+1] - ny_mat[i,j-1])) )
                gamma = ((1/4)*((i_max-1)**2)) * ( ((nx_mat[i+1,j] - nx_mat[i-1,j])**2) + ((ny_mat[i+1,j] - ny_mat[i-1,j])**2) )



                # Fill coeff matrix

                alpha_mat[i,j] = alpha
                beta_mat[i,j] = beta
                gamma_mat[i,j] = gamma

        # Init mats into class
        self.alpha_mat = alpha_mat
        self.beta_mat = beta_mat
        self.gamma_mat = gamma_mat



    def pde_grid(self,tol,gif_bool=False):

        if gif_bool:
            writer = imageio.get_writer('mygif.gif', mode='I')


        i_max = self.i_max
        j_max = self.j_max              # Init variables to use in function (is in this format due to legacy code)
        nx_mat = self.x_mat
        ny_mat = self.y_mat
        d_eta = self.d_eta

        fig_count = 0

        prev_x = np.zeros(nx_mat.shape)
        prev_y = np.zeros(ny_mat.shape)
        for j in range(0,j_max):
            for i in range(0,i_max):
                prev_x[i,j] = nx_mat[i,j]
                prev_y[i,j] = ny_mat[i,j]
            
        


        diff_x = 1
        diff_y = 1
        while diff_x >= tol or diff_y >= tol:
            
            fig_count += 1

            # Calculate alpha,beta,gamma
            self.coeff_calc()
            for j in range(0,j_max):
                for i in range(0,i_max):

                    if i == 0:
                        if j == 0:
                            pass            # Bottom left corner
                        elif j == j_max-1:
                            pass            # Top left corner
                        else:
                            ny_mat[i,j] = ny_mat[i+2,j]   # Left wall

                    elif i == i_max-1:
                        if j == 0:
                            pass            # Bottom right corner
                        elif j == j_max:
                            pass            # Top right corner
                        else:       
                            ny_mat[i,j] = ny_mat[i-2,j]       # Right wall
                    
                    elif j == 0:
                        nx_mat[i,j] = nx_mat[i,j+2] + (2*d_eta * self.__class__.yp_lower(nx_mat[i,j+2]))     # On the bottom curve
                        ny_mat[i,j] = self.__class__.y_lower(nx_mat[i,j])
                        
                    elif j == j_max-1:
                        nx_mat[i,j] = nx_mat[i,j-2] - (2*d_eta * self.__class__.yp_upper(nx_mat[i,j-2]))     # On the upper curve
                        ny_mat[i,j] = self.__class__.y_upper(nx_mat[i,j])
                        
                    else:
                        
                        nu = self.alpha_mat[i,j]*((i_max-1)**2)             # Where nu is alpha/(delta_exi^2) (using original definition for floating point error)
                        theta = -self.beta_mat[i,j]*((j_max-1)*(i_max-1))   # Where theta is -beta/(2*delta_exi*delta_eta) (using original definition for floating point error)
                        lam = self.gamma_mat[i,j]*((j_max-1)**2)            # Where lambda is gamma/(delta_eta^2) (using original definition for floating point error)
                        phi = 2*(nu + lam)                                  # Where phi is the coefficient of x_ij 


                        a1 = theta/phi
                        a2 = lam/phi
                        a3 = -a1
                        a4 = nu/phi
                        a5 = a4                 # Set realtions for the a coefficents when solving for the current point
                        a6 = a3
                        a7 = a2
                        a8 = a1
                        
                        # Calculating points

                        nx_mat[i,j] = a1*nx_mat[i-1,j-1] + a2*nx_mat[i,j-1] + a3*nx_mat[i+1,j-1] + a4*nx_mat[i-1,j] + a5*nx_mat[i+1,j] + a6*nx_mat[i-1,j+1] + a7*nx_mat[i,j+1] + a8*nx_mat[i+1,j+1]

                        ny_mat[i,j] = a1*ny_mat[i-1,j-1] + a2*ny_mat[i,j-1] + a3*ny_mat[i+1,j-1] + a4*ny_mat[i-1,j] + a5*ny_mat[i+1,j] + a6*ny_mat[i-1,j+1] + a7*ny_mat[i,j+1] + a8*ny_mat[i+1,j+1]
            

            diff_x_mat = abs(prev_x - nx_mat)
            diff_y_mat = abs(prev_y - ny_mat)
            prev_x = np.zeros(nx_mat.shape)
            prev_y = np.zeros(ny_mat.shape)
            diff_x = diff_x_mat.mean()
            diff_y = diff_y_mat.mean()
            print(diff_x,diff_y)

            for j in range(0,j_max):
                for i in range(0,i_max):
                    prev_x[i,j] = nx_mat[i,j]
                    prev_y[i,j] = ny_mat[i,j]
            

            # Update Matrices
            if gif_bool:
                plot_grid(nx_mat,ny_mat,fig_count,gif_bool)
                image = imageio.imread(f'Fig{fig_count}.png')
                writer.append_data(image)
                os.remove(f'Fig{fig_count}.png')

            self.x_mat = nx_mat
            self.y_mat = ny_mat
        
        # Declare variables to move point to corners of the bump

        diff1 = 1
        diff2 = 1
        i_index1 = 0
        i_index2 = 0
        for i in range(0,i_max):
            pt1 = 2
            pt2 = 3

            if nx_mat[i,0] < pt1:
                temp_diff1 = abs(2 - nx_mat[i,0])
                if temp_diff1 < diff1:                  # Finding smallest differenc between points and only cheking for points outside of the curve
                    diff1 = temp_diff1
                    i_index1 = i

            if nx_mat[i,0] > pt2:
                temp_diff2 = abs(nx_mat[i,j_max-1] - 3)
                if temp_diff2 < diff2:                  # Finding smallest differenc between points and only cheking for points outside of the curve
                    diff2 = temp_diff2
                    i_index2 = i
        
        nx_mat[i_index1,0] = pt1
        nx_mat[i_index2,0] = pt2            # Update index closest to corner of the bump
        nx_mat[i_index1,j_max-1] = pt1
        nx_mat[i_index2,j_max-1] = pt2

        if gif_bool:
            plot_grid(nx_mat,ny_mat,fig_count+1,gif_bool)
            image = imageio.imread(f'Fig{fig_count+1}.png')
            writer.append_data(image)
            os.remove(f'Fig{fig_count+1}.png')

        print(f'Number of iterations: {fig_count} for tolerance = {tol}')
        return nx_mat,ny_mat