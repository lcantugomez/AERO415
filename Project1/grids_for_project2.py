# Local Imports
import matplotlib.pyplot as plt
from plotting_func import plot_grid
from algebraic_grid import create_alg_grid
from plotting_func import plot_grid
from pde_grid import PDE_Grid
import pickle
import numpy as np

tol = 10**-7
i_max = 41
j_max = 9

# Boundary conditions on lower and upper wall
y_lower = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: (0.10*np.sin((t-2)*np.pi)), lambda t: 0])
y_upper = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:1, lambda t: 1 - (0.10*np.sin((t-2)*np.pi)), lambda t: 1])
yp_lower = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: (0.10*np.pi*np.cos((t-2)*np.pi)), lambda t: 0])
yp_upper = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: -(0.10*np.pi*np.cos((t-2)*np.pi)), lambda t: 0])

# Boundary conditions on left and right wall
x_left = 0
x_right = 5

# Defining delta xi and delta eta
d_exi = 1/(i_max-1)
d_eta = 1/(j_max-1)


x_pts,y_pts = create_alg_grid(i_max,j_max,x_left,x_right,y_lower,y_upper)

# Init parameters needed to compute PDE grid
params = {"i_max":i_max,"j_max":j_max,
            "x_left":x_left,"x_right":x_right,
            "y_lower":y_lower,"y_upper":y_upper,
            "yp_lower":yp_lower,"yp_upper":yp_upper,
            "x_mat":x_pts,"y_mat":y_pts,
            "d_exi":d_exi,"d_eta":d_eta}


# Create grid object with params
grid_gen = PDE_Grid(params)

# Calculate coefficients
grid_gen.coeff_calc()

        
# Saving the 80x16 grid iterations per tolerance for use in the report
print('-----------Begin saving tolerance comparisons-----------\n')

# Generate Grid and save iteration count
nx_pts,ny_pts,iters1 = grid_gen.pde_grid(tol,False)

grid = [nx_pts,ny_pts]
with open('grid1_r.pkl','wb') as file:
    pickle.dump(grid,file)

plot_grid(nx_pts,ny_pts)
plt.show()