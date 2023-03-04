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
                
                alpha  = (1/(4*(d_eta**2))) * ( ((nx_mat[i,j+1] - nx_mat[i,j-1])**2) + ((ny_mat[i,j+1] - ny_mat[i,j-1])**2) )
                beta = (1/(4*(d_eta*d_exi))) * ( ((nx_mat[i+1,j] - nx_mat[i-1,j]) * (nx_mat[i,j+1] - nx_mat[i,j-1])) + ((ny_mat[i+1,j] - ny_mat[i-1,j]) * (ny_mat[i,j+1] - ny_mat[i,j-1])) )
                gamma = (1/(4*(d_exi**2))) * ( ((nx_mat[i+1,j] - nx_mat[i-1,j])**2) + ((ny_mat[i+1,j] - ny_mat[i-1,j])**2) )



                # Fill coeff matrix

                alpha_mat[i,j] = alpha
                beta_mat[i,j] = beta
                gamma_mat[i,j] = gamma

        # Init mats into class

        self.alpha_mat = alpha_mat
        self.beta_mat = beta_mat
        self.gamma_mat = gamma_mat



    def pde_grid(self,tol):

        

        i_max = self.i_max
        j_max = self.j_max              # Init variables to use in function (is in this format due to legacy code)
        nx_mat = self.x_mat
        ny_mat = self.y_mat
        d_exi = self.d_exi
        d_eta = self.d_eta

        i_index_diff = int((2/5)*i_max)

        j_index_diff = int((0.3/1)*j_max)

        prev_x = nx_mat[i_index_diff,j_index_diff]
        prev_y = ny_mat[i_index_diff,j_index_diff]
        diff_y = 1
        diff_x = 1


        diff = 1
        while diff >= tol:

            # Calculate alpha,beta,gamma
            self.coeff_calc()
            diff += -0.005
            for j in range(0,j_max):
                for i in range(0,i_max):

                    if i == 0:
                        if j == 0:
                            pass            # Bottom left corner
                        elif j == j_max-1:
                            pass            # Top left corner
                        else:
                            ny_mat[i,j] = ny_mat[i+1,j]   # Left wall

                    elif i == i_max-1:
                        if j == 0:
                            pass            # Bottom right corner
                        elif j == j_max:
                            pass            # Top right corner
                        else:       
                            ny_mat[i,j] = ny_mat[i-1,j]       # Right wall
                    
                    elif j == 0:
                        nx_mat[i,j] = nx_mat[i,j+1] + (d_eta * self.__class__.yp_lower(nx_mat[i,j]))     # On the bottom curve
                        ny_mat[i,j] = self.__class__.y_lower(nx_mat[i,j])
                        
                    elif j == j_max-1:
                        nx_mat[i,j] = nx_mat[i,j-1] - (d_eta * self.__class__.yp_upper(nx_mat[i,j]))     # On the upper curve
                        ny_mat[i,j] = self.__class__.y_upper(nx_mat[i,j])
                        
                    else:
                        
                        nu = self.alpha_mat[i,j]/(d_exi**2)             # Where nu is alpha/(delta_exi^2)
                        theta = -self.beta_mat[i,j]/(2*d_exi*d_eta)     # Where theta is -beta/(2*delta_exi*delta_eta)
                        lam = self.gamma_mat[i,j]/(d_eta**2)            # Where lambda is gamma/(delta_eta^2)
                        phi = 2*(nu + lam)                              # Where phi is the coefficient of x_ij


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
            
            # Update matrices
            
            diff_x = (prev_x - nx_mat[i_index_diff,j_index_diff])
            diff_y = (prev_y - ny_mat[i_index_diff,j_index_diff])
            prev_x = nx_mat[i_index_diff,j_index_diff]
            prev_y = ny_mat[i_index_diff,j_index_diff]
            print(diff_x," ",diff_y)

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

        return nx_mat,ny_mat