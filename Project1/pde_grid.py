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

        self.coeff_calc()
        

        i_max = self.i_max
        j_max = self.j_max              # Init variables to use in function (is in this format due to legacy code)
        nx_mat = self.x_mat
        ny_mat = self.y_mat
        d_exi = self.d_exi
        d_eta = self.d_eta

        diff = 1
        while diff >= tol:
            diff += -0.01
            print(diff)
            for j in range(j_max-1,-1,-1):
                for i in range(i_max-1,-1,-1):

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
                        nx_mat[i,j] = nx_mat[i,j+2] + (2 * d_eta * self.__class__.yp_lower(nx_mat[i,j]))     # On the bottom curve
                        ny_mat[i,j] = self.__class__.y_lower(nx_mat[i,j])
                        
                    elif j == j_max-1:
                        nx_mat[i,j] = nx_mat[i,j-2] + (2 * d_eta * self.__class__.yp_upper(nx_mat[i,j]))     # On the upper curve
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
                        a5 = a4
                        a6 = a3
                        a7 = a2
                        a8 = a1
                        
                        nx_mat[i,j] = a1*nx_mat[i-1,j-1] + a2*nx_mat[i,j-1] + a3*nx_mat[i+1,j-1] + a4*nx_mat[i-1,j] + a5*nx_mat[i+1,j] + a6*nx_mat[i-1,j+1] + a7*nx_mat[i,j+1] + a8*nx_mat[i+1,j+1]

                        ny_mat[i,j] = a1*ny_mat[i-1,j-1] + a2*ny_mat[i,j-1] + a3*ny_mat[i+1,j-1] + a4*ny_mat[i-1,j] + a5*ny_mat[i+1,j] + a6*ny_mat[i-1,j+1] + a7*ny_mat[i,j+1] + a8*ny_mat[i+1,j+1]
            
            self.x_mat = nx_mat
            self.y_mat = ny_mat
            self.coeff_calc()
            
            x_arr = []
            y_arr = []

            """for i in range(0,i_max + 1):
                for j in range(0, j_max + 1):
                    x_arr.append(nx_mat[i,j])
                    y_arr.append(ny_mat[i,j])

            plt.scatter(x_arr,y_arr)
            plt.show()"""

        return nx_mat,ny_mat