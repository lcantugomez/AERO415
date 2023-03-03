import numpy as np
import matplotlib.pyplot as plt

from algebraic_grid import create_alg_grid
from pde_grid import PDE_Grid

y_lower = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: (0.15*np.sin((t-2)*np.pi)), lambda t: 0])
y_upper = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:1, lambda t: 1 - (0.15*np.sin((t-2)*np.pi)), lambda t: 1])
yp_lower = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: (0.15*np.pi*np.cos((t-2)*np.pi)), lambda t: 0])
yp_upper = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: -(0.15*np.pi*np.cos((t-2)*np.pi)), lambda t: 0])

x_left = 0
x_right = 5

i_max = 100
j_max = 20

d_exi = 1/(i_max-1)
d_eta = 1/(j_max-1)



x_pts,y_pts = create_alg_grid(i_max,j_max,x_left,x_right,y_lower,y_upper)

print(x_pts)

x_arr = []
y_arr = []

for i in range(0,i_max):
    for j in range(0, j_max):
        x_arr.append(x_pts[i,j])
        y_arr.append(y_pts[i,j])

plt.scatter(x_arr,y_arr)
plt.show()

params = {"i_max":i_max,"j_max":j_max,
         "x_left":x_left,"x_right":x_right,
         "y_lower":y_lower,"y_upper":y_upper,
         "yp_lower":yp_lower,"yp_upper":yp_upper,
         "x_mat":x_pts,"y_mat":y_pts,
         "d_exi":d_exi,"d_eta":d_eta}


grid_gen = PDE_Grid(params)

nx_pts,ny_pts = grid_gen.pde_grid(10**-7)


x_arr = []
y_arr = []

for i in range(0,i_max):
    for j in range(0, j_max):
        x_arr.append(nx_pts[i,j])
        y_arr.append(ny_pts[i,j])

plt.scatter(x_arr,y_arr,s=5)
plt.show()
