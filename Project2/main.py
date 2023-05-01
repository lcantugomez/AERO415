# Global imports
import numpy as np
import pickle
from plotting_func import plot_grid
import matplotlib.pyplot as plt
import time

start = time.time()
# Local imports 
from struct_2d_euler import Struct2DEulerChannel

with open('grid1.pkl','rb') as file:
    grid1 = pickle.load(file)

# Inlet conditions
Mach = 0.7
inlet_angle = 0
rho_inf = 1
gamma = 1.4
cfl = 0.7

rho_inf = 1
rho_u_inf = Mach * np.cos(inlet_angle)
rho_v_inf = Mach * np.sin(inlet_angle)
rho_E = (1/(gamma*(gamma - 1))) + (0.5*(Mach**2))
q_inf = np.matrix([[rho_inf],[rho_u_inf],[rho_v_inf],[rho_E]])
euler_solver = Struct2DEulerChannel(grid1,q_inf,gamma,cfl)
euler_solver.create_cell_matrices()
euler_solver.bcWall()
euler_solver.bcInletOutlet()

try:
    euler_solver.runge_kutta(0,0.35,0.07)
except Exception as e:
    print(e)

print(euler_solver.resid1_list,'Resid 1 \n')
print(euler_solver.resid2_list,'Resid 2 \n')
print(euler_solver.resid3_list,'Resid 3 \n')
print(euler_solver.resid4_list,'Resid 4 \n')
iter_list = np.arange(1,len(euler_solver.resid1_list)+1,1)

plt.figure(1)
plt.plot(iter_list,euler_solver.resid1_list)
plt.show()

x = euler_solver.grid[0]
y = euler_solver.grid[1]
z = np.zeros(shape=euler_solver.cell_pressure_mat.shape)
z2 = np.zeros(shape=euler_solver.cell_pressure_mat.shape)
z3 = np.zeros(shape=euler_solver.cell_pressure_mat.shape)
M_mat = np.zeros(shape=euler_solver.cell_pressure_mat.shape)
for j in range(2,euler_solver.gc_j_max-2):
    for i in range(2,euler_solver.gc_i_max-2):
        q_ij = euler_solver.ghost_cell_state_vector_mat[i,j]
        q0 = q_ij[0,0]
        q1 = q_ij[1,0]
        q2 = q_ij[2,0]
        p_ij = euler_solver.cell_pressure_mat[i,j]
        c = np.sqrt(euler_solver.gamma *(p_ij/q0))
        V_ij = np.sqrt((q1**2)+(q2**2))/q0
        M_ij = V_ij/c
        M_mat[i,j] = M_ij
        z[i,j] = q0

z = M_mat[2:euler_solver.gc_i_max-2,2:euler_solver.gc_j_max-2]
p = euler_solver.cell_pressure_mat[2:euler_solver.gc_i_max-2,2:euler_solver.gc_j_max-2]
"""print(z,'Mach\n\n')
print(p,'Pressure')"""
plot_grid(x,y,'cool',z,title='Mach Number')
plot_grid(x,y,'hot_r',p,title='Pressure')
end = time.time()
delta_time = round((end - start),2)
print(f'Time elapsed {delta_time}')
plt.show()