from struct_2d_euler_mat import Struct2DEulerChannelMat
import numpy as np
import pickle
from plotting_func import plot_grid
import matplotlib.pyplot as plt
import time


start = time.time()
with open('grid2.pkl','rb') as file:
    grid1 = pickle.load(file)


Mach = 0.7
inlet_angle = 0
gamma = 1.4
rho_inf = 1
cfl = 0.7

rho_u_inf = Mach * np.cos(inlet_angle)
rho_v_inf = Mach * np.sin(inlet_angle)
rho_E = (1/(gamma*(gamma - 1))) + (0.5*(Mach**2))
q_inf = np.matrix([[rho_inf],[rho_u_inf],[rho_v_inf],[rho_E]])
euler_solver = Struct2DEulerChannelMat(grid1,q_inf,gamma,cfl)
euler_solver.create_cell_matrices()
euler_solver.bcWall()
euler_solver.bcInletOutlet()
euler_solver.runge_kutta(0,0.25,0.05)
end = time.time()

x = euler_solver.grid[0]
y = euler_solver.grid[1]
z = np.zeros(shape=euler_solver.cell_pressure_mat.shape)
M_mat = np.zeros(shape=euler_solver.cell_pressure_mat.shape)
for j in range(2,euler_solver.gc_j_max-2):
    for i in range(2,euler_solver.gc_i_max-2):
        q_ij = euler_solver.ghost_cell_state_vector_mat[i,j]
        q0 = q_ij[0]
        q1 = q_ij[1]
        q2 = q_ij[2]
        p_ij = euler_solver.cell_pressure_mat[i,j]
        c = np.sqrt(euler_solver.gamma *(p_ij/q0))
        V_ij = np.sqrt((q1**2)+(q2**2))/q0
        M_ij = V_ij/c
        M_mat[i,j] = M_ij
        z[i,j] = q0

z = M_mat[2:euler_solver.gc_i_max-2,2:euler_solver.gc_j_max-2]
p = euler_solver.cell_pressure_mat[2:euler_solver.gc_i_max-2,2:euler_solver.gc_j_max-2]
plot_grid(x,y,'cool',z,title='Mach Number')
plot_grid(x,y,'hot_r',p,title='Pressure')

diff1 = euler_solver.diff1_list
diff2 = euler_solver.diff2_list
diff3 = euler_solver.diff3_list
diff4 = euler_solver.diff4_list
iters = np.arange(1,len(diff1) + 1,1)
plt.figure('Residuals')
plt.plot(iters,diff1,label='Diff1')
plt.plot(iters,diff2,label='Diff2')
plt.plot(iters,diff3,label='Diff3')
plt.plot(iters,diff4,label='Diff4')
plt.legend()

delta_t = round(end - start,3)
print(f'Elapsed time {delta_t}')
plt.show()