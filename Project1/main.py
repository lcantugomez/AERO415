import numpy as np
import matplotlib.pyplot as plt

from algebraic_grid import create_alg_grid
from pde_grid import coeff_calc, pde_grid

y_lower = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: (0.15*np.sin((t-2)*np.pi)), lambda t: 0])
y_upper = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:1, lambda t: 1 - (0.15*np.sin((t-2)*np.pi)), lambda t: 1])
x_left = 0
x_right = 5
i_max = 20
j_max = 4
d_exi = 1/i_max
d_eta = 1/j_max

x_pts,y_pts = create_alg_grid(i_max,j_max,x_left,x_right,y_lower,y_upper)



alpha_mat, beta_mat, gamma_mat = coeff_calc(i_max,j_max,x_pts,y_pts,d_exi,d_eta)

pde_grid(alpha_mat,beta_mat,gamma_mat,x_pts,y_pts,d_exi,d_eta,10**-7,i_max,j_max)
