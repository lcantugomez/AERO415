import numpy as np
import matplotlib.pyplot as plt
from plotting_func import plot_grid
import imageio.v2 as imageio
import os

class PDE_Grid():

    # Init class with set of parameters
    # (if not familiar with Python, this may be confusing at first, but it is simnply unpacking the arguments and making them attributes of the class)
    def __init__(self,params) -> None:      
        for kw,arg in params.items():
            setattr(PDE_Grid,kw,arg)


    def coeff_calc(self):

        # Initialize alpha, beta, gamma arrays

        alpha_mat = np.zeros((self.i_max,self.j_max))
        beta_mat = np.zeros((self.i_max,self.j_max))
        gamma_mat = np.zeros((self.i_max,self.j_max))

        for j in range(1,self.j_max-1):            # Loop only through the interior nodes
            for i in range(1,self.i_max-1):
                    
                

                # Calc the alpha beta and gamma vals
                
                alpha  = ((1/4)*((self.j_max-1)**2)) * ( ((self.x_mat[i,j+1] - self.x_mat[i,j-1])**2)
                                                         + ((self.y_mat[i,j+1] - self.y_mat[i,j-1])**2) )
                
                beta = ((1/4)*((self.j_max-1)*(self.i_max-1))) * ( ((self.x_mat[i+1,j] - self.x_mat[i-1,j]) * 
                (self.x_mat[i,j+1] - self.x_mat[i,j-1])) + ((self.y_mat[i+1,j] - self.y_mat[i-1,j]) * (self.y_mat[i,j+1] - self.y_mat[i,j-1])) )
                
                gamma = ((1/4)*((self.i_max-1)**2)) * ( ((self.x_mat[i+1,j] - self.x_mat[i-1,j])**2) + 
                                                       ((self.y_mat[i+1,j] - self.y_mat[i-1,j])**2) )



                # Fill coeff matrix

                alpha_mat[i,j] = alpha
                beta_mat[i,j] = beta
                gamma_mat[i,j] = gamma

        # Init mats into class
        self.alpha_mat = alpha_mat
        self.beta_mat = beta_mat
        self.gamma_mat = gamma_mat



    def pde_grid(self,tol,gif_bool=False):

        # Conditional for making gif writer and figure count of each plot of the grid as it changes. Only used for debugging and troubleshooting

        fig_count = 0

        if gif_bool:
            writer = imageio.get_writer('mygif.gif', mode='I')


        # Setting the algebraic grid as the first matrix to compare
        prev_x = np.copy(self.x_mat)
        prev_y = np.copy(self.y_mat)
        


        diff_x = 1
        diff_y = 1
        while diff_x >= tol or diff_y >= tol:
            
            # Updateing figure count per iteration
            fig_count += 1
            if fig_count % 200 == 0:
                print(f'Running... Iteration = {fig_count}')

            # Calculate alpha,beta,gamma
            self.coeff_calc()
            for j in range(0,self.j_max):
                for i in range(0,self.i_max):

                    if i == 0:
                        if j == 0:
                            pass            # Bottom left corner
                        elif j == self.j_max-1:
                            pass            # Top left corner
                        else:
                            self.y_mat[i,j] = self.y_mat[i+2,j]   # Left wall

                    elif i == self.i_max-1:
                        if j == 0:
                            pass            # Bottom right corner
                        elif j == self.j_max:
                            pass            # Top right corner
                        else:       
                            self.y_mat[i,j] = self.y_mat[i-2,j]       # Right wall
                    
                    elif j == 0:
                        self.x_mat[i,j] = self.x_mat[i,j+2] + (2*self.d_eta * self.__class__.yp_lower(self.x_mat[i,j+2]))     # On the bottom curve
                        self.y_mat[i,j] = self.__class__.y_lower(self.x_mat[i,j])
                        
                    elif j == self.j_max-1:
                        self.x_mat[i,j] = self.x_mat[i,j-2] - (2*self.d_eta * self.__class__.yp_upper(self.x_mat[i,j-2]))     # On the upper curve
                        self.y_mat[i,j] = self.__class__.y_upper(self.x_mat[i,j])
                        
                    else:

                        # Where nu is alpha/(delta_exi^2) (using original definition to prevent floating point error)
                        # Where theta is -beta/(2*delta_exi*delta_eta) (using original definition to prevent floating point error)
                        # Where lambda is gamma/(delta_eta^2) (using original definition to prevent floating point error)
                        # Where phi is the coefficient of x_ij
                        nu = self.alpha_mat[i,j]*((self.i_max-1)**2)                    
                        theta = -self.beta_mat[i,j]*((self.j_max-1)*(self.i_max-1))     
                        lam = self.gamma_mat[i,j]*((self.j_max-1)**2)                   
                        phi = 2*(nu + lam)                                              


                        a1 = theta/phi
                        a2 = lam/phi
                        a3 = -a1
                        a4 = nu/phi
                        a5 = a4                 # Set realtions for the a coefficents when solving for the current point
                        a6 = a3
                        a7 = a2
                        a8 = a1
                        
                        # Calculating points
                        self.x_mat[i,j] = a1*self.x_mat[i-1,j-1] + a2*self.x_mat[i,j-1] + a3*self.x_mat[i+1,j-1] + a4*self.x_mat[i-1,j]
                        + a5*self.x_mat[i+1,j] + a6*self.x_mat[i-1,j+1] + a7*self.x_mat[i,j+1] + a8*self.x_mat[i+1,j+1]

                        self.y_mat[i,j] = a1*self.y_mat[i-1,j-1] + a2*self.y_mat[i,j-1] + a3*self.y_mat[i+1,j-1] + a4*self.y_mat[i-1,j]
                        + a5*self.y_mat[i+1,j] + a6*self.y_mat[i-1,j+1] + a7*self.y_mat[i,j+1] + a8*self.y_mat[i+1,j+1]
            

            # Subtracting new iteration from previous iteration and getting absoulte values
            diff_x_mat = abs(prev_x - self.x_mat)
            diff_y_mat = abs(prev_y - self.y_mat)

            # Finding the max difference between iterations and replacing the difference variables to continue loop
            diff_x = diff_x_mat.max()
            diff_y = diff_y_mat.max()

            # Making the previous iteration matrix the current iteration matrix
            prev_x = np.copy(self.x_mat)
            prev_y = np.copy(self.y_mat)
            
            
            # Condiitonal to write the figure onto the gif and remove the file to prevent cluttering in the folder
            if gif_bool:
                plot_grid(self.x_mat,self.y_mat,fig_count,gif_bool)
                image = imageio.imread(f'Fig{fig_count}.png')
                writer.append_data(image)
                os.remove(f'Fig{fig_count}.png')


            
        # Declare variables to move point to corners of the bump
        diff1 = 1
        diff2 = 1
        i_index1 = 0
        i_index2 = 0
        for i in range(0,self.i_max):
            pt1 = 2
            pt2 = 3

            if self.x_mat[i,0] < pt1:
                temp_diff1 = abs(2 - self.x_mat[i,0])
                if temp_diff1 < diff1:                  # Finding smallest difference between points and only checking for points outside of the curve
                    diff1 = temp_diff1
                    i_index1 = i

            if self.x_mat[i,0] > pt2:
                temp_diff2 = abs(self.x_mat[i,0] - 3)
                if temp_diff2 < diff2:                  # Finding smallest difference between points and only checking for points outside of the curve
                    diff2 = temp_diff2
                    i_index2 = i
        
        self.x_mat[i_index1,0] = pt1
        self.x_mat[i_index2,0] = pt2            # Update index closest to corner of the bump
        self.x_mat[i_index1,self.j_max-1] = pt1
        self.x_mat[i_index2,self.j_max-1] = pt2

        if gif_bool:
            plot_grid(self.x_mat,self.y_mat,fig_count+1,gif_bool)
            image = imageio.imread(f'Fig{fig_count+1}.png')
            writer.append_data(image)
            os.remove(f'Fig{fig_count+1}.png')

        print(f'Number of iterations: {fig_count} for tolerance = {tol}')
        return self.x_mat,self.y_mat,fig_count