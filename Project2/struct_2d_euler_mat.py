# Import libraries
import numpy as np
from plotting_func import plot_grid
import matplotlib.pyplot as plt
import time


# Init solver class
class Struct2DEulerChannelMat():
    def __init__(self,grid,qminf,gamma,cfl,inlet_angle = 0) -> None:
        self.grid = grid
        self.qminf = qminf
        self.gamma = gamma
        self.cfl = cfl
        self.inlet_angle = inlet_angle
        self.grid_shape = grid[0].shape
        self.n_i_max = self.grid_shape[0]
        self.n_j_max = self.grid_shape[1]
        self.c_i_max = self.n_i_max - 1
        self.c_j_max = self.n_j_max - 1
        self.gc_i_max = self.c_i_max + 4
        self.gc_j_max = self.c_j_max + 4
        self.d_xi = 1/(self.n_i_max-1)
        self.d_eta = 1/(self.n_j_max-1)
        self.cell_area_mat = None
        self.ghost_cell_state_vector_mat = None
        self.f_ghost_mat = None
        self.g_ghost_mat = None
        self.cell_grid_y_mat = None
        self.cell_grid_x_mat = None
        self.cell_pressure_mat = None
        self.residuals = None
        self.diss_mat = None
        self.dt_mat = None
        self.resid1_list = None
        self.resid2_list = None
        self.resid3_list = None
        self.resid4_list = None
        self.diff1_list = None
        self.diff2_list = None
        self.diff3_list = None
        self.diff4_list = None


    # Init and calculate cell areas, init cell grid, init cell pressure matrix, and init ghost cells state matrix
    def create_cell_matrices(self):
        cell_area_mat = np.zeros(shape=(self.gc_i_max,self.gc_j_max),dtype=float)
        ghost_cell_state_vector_mat = np.zeros(shape=(self.gc_i_max, self.gc_j_max,4),dtype=float)
        cell_grid_y_mat = np.zeros(shape=(self.gc_i_max,self.gc_j_max,4),dtype=float)
        cell_grid_x_mat = np.zeros(shape=(self.gc_i_max,self.gc_j_max,4),dtype=float)
        cell_pressure_mat = np.zeros(shape=(self.gc_i_max, self.gc_j_max),dtype=float)

        for i in range(2,self.gc_i_max-2):
            for j in range(2,self.gc_j_max-2):
                i_grid = i - 2
                j_grid = j - 2

                x1 = self.grid[0][i_grid,j_grid]
                x2 = self.grid[0][i_grid+1,j_grid]
                x3 = self.grid[0][i_grid,j_grid+1]
                x4 = self.grid[0][i_grid+1,j_grid+1]

                y1 = self.grid[1][i_grid,j_grid]
                y2 = self.grid[1][i_grid+1,j_grid]
                y3 = self.grid[1][i_grid,j_grid+1]
                y4 = self.grid[1][i_grid+1,j_grid+1]

                area_ij = 0.5*(((x4-x1)*(y3-y2)) - ((y4-y1)*(x3-x2)))

                cell_area_mat[i,j] = area_ij

                cell_grid_y_mat[i,j,0] = y1
                cell_grid_y_mat[i,j,1] = y2
                cell_grid_y_mat[i,j,2] = y3
                cell_grid_y_mat[i,j,3] = y4

                cell_grid_x_mat[i,j,0] = x1
                cell_grid_x_mat[i,j,1] = x2
                cell_grid_x_mat[i,j,2] = x3
                cell_grid_x_mat[i,j,3] = x4

        for i in range(0,self.gc_i_max):
            for j in range(0,self.gc_j_max):

                q0 = self.qminf[0,0]
                q1 = self.qminf[1,0]
                q2 = self.qminf[2,0]
                q3 = self.qminf[3,0]

                ghost_cell_state_vector_mat[i,j,0] = q0
                ghost_cell_state_vector_mat[i,j,1] = q1
                ghost_cell_state_vector_mat[i,j,2] = q2
                ghost_cell_state_vector_mat[i,j,3] = q3 

                p = (self.gamma - 1)*(q3 - (0.5*((q1**2) + (q2**2)/q0)))
                cell_pressure_mat[i,j] = p

        self.cell_area_mat = np.copy(cell_area_mat)
        self.ghost_cell_state_vector_mat = np.copy(ghost_cell_state_vector_mat)
        self.cell_grid_y_mat = np.copy(cell_grid_y_mat)
        self.cell_grid_x_mat = np.copy(cell_grid_x_mat)
        self.cell_pressure_mat = np.copy(cell_pressure_mat)
    
    # Create method for updating pressures
    def update_pressures(self):
        q0 = self.ghost_cell_state_vector_mat[:,:,0]
        q1 = self.ghost_cell_state_vector_mat[:,:,1]
        q2 = self.ghost_cell_state_vector_mat[:,:,2]
        q3 = self.ghost_cell_state_vector_mat[:,:,3]

        p = (self.gamma - 1)*(q3 - (0.5*((q1**2) + (q2**2)/q0)))

        self.cell_pressure_mat = p
    
    # Init fluxes
    def init_fluxes(self):
        f_ghost_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape,dtype=float)
        g_ghost_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape,dtype=float)

        q0 = np.copy(self.ghost_cell_state_vector_mat[:,:,0])
        q1 = np.copy(self.ghost_cell_state_vector_mat[:,:,1])
        q2 = np.copy(self.ghost_cell_state_vector_mat[:,:,2])
        q3 = np.copy(self.ghost_cell_state_vector_mat[:,:,3])
        p = np.copy(self.cell_pressure_mat)

        f0 = q1
        f1 = ((q1**2)/q0) + p
        f2 = (q1*q2)/q0
        f3 = (q1/q0)*(q3 + p)

        g0 = q2
        g1 = (q1*q2)/q0
        g2 = ((q2**2)/q0) + p
        g3 = (q2/q0)*(q3 + p)

        f_ghost_mat[:,:,0] = f0
        f_ghost_mat[:,:,1] = f1
        f_ghost_mat[:,:,2] = f2
        f_ghost_mat[:,:,3] = f3

        g_ghost_mat[:,:,0] = g0
        g_ghost_mat[:,:,1] = g1
        g_ghost_mat[:,:,2] = g2
        g_ghost_mat[:,:,3] = g3

        

        self.f_ghost_mat = np.copy(f_ghost_mat)
        self.g_ghost_mat = np.copy(g_ghost_mat)
    

    # Init residual matrix
    def init_residuals(self):
        residuals = np.zeros(shape=self.ghost_cell_state_vector_mat.shape,dtype=float)

        f_m = self.f_ghost_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
        f_r = self.f_ghost_mat[3:self.gc_i_max-1,2:self.gc_j_max-2,:]
        f_t = self.f_ghost_mat[2:self.gc_i_max-2,3:self.gc_j_max-1,:]
        f_b = self.f_ghost_mat[2:self.gc_i_max-2,1:self.gc_j_max-3,:]
        f_l = self.f_ghost_mat[1:self.gc_i_max-3,2:self.gc_j_max-2,:]

        g_m = self.g_ghost_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
        g_r = self.g_ghost_mat[3:self.gc_i_max-1,2:self.gc_j_max-2,:]
        g_t = self.g_ghost_mat[2:self.gc_i_max-2,3:self.gc_j_max-1,:]
        g_b = self.g_ghost_mat[2:self.gc_i_max-2,1:self.gc_j_max-3,:]
        g_l = self.g_ghost_mat[1:self.gc_i_max-3,2:self.gc_j_max-2,:]

        dy_ihalf = self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,3] - self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,1]
        dx_ihalf = self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,3] - self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,1]

        dy_jhalf = self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,2] - self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,3]
        dx_jhalf = self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,2] - self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,3]

        dy_imhalf = self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0] - self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,2]
        dx_imhalf = self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0] - self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,2]

        dy_jmhalf = self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,1] - self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0]
        dx_jmhalf = self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,1] - self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0]

        for i in range(4):
            f_ij_E_avg = 0.5*(f_m[:,:,i] + f_r[:,:,i])*dy_ihalf
            f_ij_W_avg = 0.5*(f_m[:,:,i] + f_l[:,:,i])*dy_imhalf
            f_ij_N_avg = 0.5*(f_m[:,:,i] + f_t[:,:,i])*dy_jhalf
            f_ij_S_avg = 0.5*(f_m[:,:,i] + f_b[:,:,i])*dy_jmhalf

            g_ij_E_avg = 0.5*(g_m[:,:,i] + g_r[:,:,i])*dx_ihalf
            g_ij_W_avg = 0.5*(g_m[:,:,i] + g_l[:,:,i])*dx_imhalf
            g_ij_N_avg = 0.5*(g_m[:,:,i] + g_t[:,:,i])*dx_jhalf
            g_ij_S_avg = 0.5*(g_m[:,:,i] + g_b[:,:,i])*dx_jmhalf

            residuals[2:self.gc_i_max-2,2:self.gc_j_max-2,i] = (f_ij_E_avg - g_ij_E_avg) + (f_ij_N_avg - g_ij_N_avg) + (f_ij_W_avg - g_ij_W_avg) + (f_ij_S_avg - g_ij_S_avg)
        
        self.residuals = np.copy(residuals)


# -------------------- Begin Methods for calculating the dissipation matrix -------------------- #

    # Create method for length scale
    def length(self,face,with_dx_dy = False):
        if face.upper() == 'N':
            dx = self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,2] - self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,3]
            dy = self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,2] - self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,3]
        elif face.upper() == 'S':
            dx = self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,1] - self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0]
            dy = self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,1] - self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0]
        elif face.upper() == 'E':
            dx = self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,3] - self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,1]
            dy = self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,3] - self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,1]
        elif face.upper() == 'W':
            dx = self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0] - self.cell_grid_x_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,2]
            dy = self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0] - self.cell_grid_y_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,2]
        else:
            raise ValueError

        dx_sqrd = (dx)**2
        dy_sqrd = (dy)**2
        face_length = np.sqrt(dx_sqrd + dy_sqrd)

        if with_dx_dy:
            return dx,dy,face_length
        else:
            return face_length

    # Create method for delta_xi operation on q
    def d_xi_q(self,face):
        if face.upper() == 'E':
            q1 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
            q2 = self.ghost_cell_state_vector_mat[3:self.gc_i_max-1,2:self.gc_j_max-2,:]
        elif face.upper() == 'W':
            q1 = self.ghost_cell_state_vector_mat[1:self.gc_i_max-3,2:self.gc_j_max-2,:]
            q2 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
        return q2-q1

    # Create method for delta_eta operation on q
    def d_eta_q(self,face):
        if face.upper() == 'N':
            q1 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
            q2 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,3:self.gc_j_max-1,:]
        elif face.upper() == 'S':
            q1 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,1:self.gc_j_max-3,:]
            q2 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
        return q2-q1
    
    # Create method for delta_xi^2 operation on q
    def d_xi_sqrd_q(self,cell):
        if cell.upper() == 'M':
            q1 = self.ghost_cell_state_vector_mat[3:self.gc_i_max-1,2:self.gc_j_max-2,:]
            q2 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
            q3 = self.ghost_cell_state_vector_mat[1:self.gc_i_max-3,2:self.gc_j_max-2,:]
        elif cell.upper() == 'R':
            q1 = self.ghost_cell_state_vector_mat[4:self.gc_i_max-0,2:self.gc_j_max-2,:]
            q2 = self.ghost_cell_state_vector_mat[3:self.gc_i_max-1,2:self.gc_j_max-2,:]
            q3 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
        elif cell.upper() == 'L':
            q1 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
            q2 = self.ghost_cell_state_vector_mat[1:self.gc_i_max-3,2:self.gc_j_max-2,:]
            q3 = self.ghost_cell_state_vector_mat[0:self.gc_i_max-4,2:self.gc_j_max-2,:]
        return np.copy(q1 - (2*q2) + q3)

    # Create method for delta_eta^2 operation on q
    def d_eta_sqrd_q(self,cell):
        if cell.upper() == 'M':
            q1 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,3:self.gc_j_max-1,:]
            q2 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
            q3 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,1:self.gc_j_max-3,:]
        elif cell.upper() == 'T':
            q1 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,4:self.gc_j_max-0,:]
            q2 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,3:self.gc_j_max-1,:]
            q3 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
        elif cell.upper() == 'B':
            q1 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
            q2 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
            q3 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,:]
        return np.copy(q1 - (2*q2) + q3)
    
    # Create method for delta_xi^2 operation on pressure
    def d_xi_sqrd_p(self,cell):
        if cell.upper() == 'M':
            p1 = self.cell_pressure_mat[3:self.gc_i_max-1,2:self.gc_j_max-2]
            p2 = self.cell_pressure_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
            p3 = self.cell_pressure_mat[1:self.gc_i_max-3,2:self.gc_j_max-2]
        elif cell.upper() == 'R':
            p1 = self.cell_pressure_mat[4:self.gc_i_max-0,2:self.gc_j_max-2]
            p2 = self.cell_pressure_mat[3:self.gc_i_max-1,2:self.gc_j_max-2]
            p3 = self.cell_pressure_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
        elif cell.upper() == 'L':
            p1 = self.cell_pressure_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
            p2 = self.cell_pressure_mat[1:self.gc_i_max-3,2:self.gc_j_max-2]
            p3 = self.cell_pressure_mat[0:self.gc_i_max-4,2:self.gc_j_max-2]
        return (p1 - (2*p2) + p3),p1,p2,p3

    # Create method for delta_eta^2 operation on pressure
    def d_eta_sqrd_p(self,cell):
        if cell.upper() == 'M':
            p1 = self.cell_pressure_mat[2:self.gc_i_max-2,3:self.gc_j_max-1]
            p2 = self.cell_pressure_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
            p3 = self.cell_pressure_mat[2:self.gc_i_max-2,1:self.gc_j_max-3]
        elif cell.upper() == 'T':
            p1 = self.cell_pressure_mat[2:self.gc_i_max-2,4:self.gc_j_max-0]
            p2 = self.cell_pressure_mat[2:self.gc_i_max-2,3:self.gc_j_max-1]
            p3 = self.cell_pressure_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
        elif cell.upper() == 'B':
            p1 = self.cell_pressure_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
            p2 = self.cell_pressure_mat[2:self.gc_i_max-2,1:self.gc_j_max-3]
            p3 = self.cell_pressure_mat[2:self.gc_i_max-2,0:self.gc_j_max-4]
        return (p1 - (2*p2) + p3),p1,p2,p3

    # Create method for delta_xi^3 operation on q
    def d_xi_cubed_q(self,face):
        if face.upper() == 'E':
            q1 = self.d_xi_sqrd_q(cell='M')
            q2 = self.d_xi_sqrd_q(cell='R')
        elif face.upper() == 'W':
            q1 = self.d_xi_sqrd_q(cell='L')
            q2 = self.d_xi_sqrd_q(cell='M')
        return np.copy(q2 - q1)

    # Create method for delta_eta^3 operation on q
    def d_eta_cubed_q(self,face):
        if face.upper() == 'N':
            q1 = self.d_eta_sqrd_q(cell='M')
            q2 = self.d_eta_sqrd_q(cell='T')
        elif face.upper() == 'S':
            q1 = self.d_eta_sqrd_q(cell='B')
            q2 = self.d_eta_sqrd_q(cell='M')
        return np.copy(q2 - q1)
    
    # Create method for s2 switch in xi for the cell
    def s2_xi_cell(self,nu2,cell):
        num,p1,p2,p3 = self.d_xi_sqrd_p(cell=cell)
        denom = p1 + (2*p2) + p3
        s2 = nu2 * (abs(num)/denom)
        return s2

    # Create method for s2 switch in eta for the cell
    def s2_eta_cell(self,nu2,cell):
        num,p1,p2,p3 = self.d_eta_sqrd_p(cell=cell)
        denom = p1 + (2*p2) + p3
        s2 = nu2 * (abs(num)/denom)
        return s2

    # Create method for s2 switch in xi at the face
    def s2_xi_face(self,nu2,face):
        if face == 'E':
            s2 = 0.5*(self.s2_xi_cell(nu2,cell='M') + self.s2_xi_cell(nu2,cell='R'))
        elif face == 'W':
            s2 = 0.5*(self.s2_xi_cell(nu2,cell='L') + self.s2_xi_cell(nu2,cell='M'))
        return s2

    # Create method for s2 switch in eta at the face
    def s2_eta_face(self,nu2,face):
        if face.upper() == 'N':
            s2 = 0.5*(self.s2_eta_cell(nu2,cell='M') + self.s2_eta_cell(nu2,cell='T'))
        elif face.upper() == 'S':
            s2 = 0.5*(self.s2_eta_cell(nu2,cell='B') + self.s2_eta_cell(nu2,cell='M'))
        return s2
    
    # Create method for s4 switch in xi at the face
    def s4_xi_face(self,nu2,nu4,face):
        s4 = nu4 - self.s2_xi_face(nu2,face=face)
        s4.clip(min=0)
        return s4

    # Create method for s4 switch in eta at the face
    def s4_eta_face(self,nu2,nu4,face):
        s4 = nu4 - self.s2_eta_face(nu2,face)
        s4.clip(min=0)
        return s4

    # Create method for eigen values at constant xi face
    def lam(self,face):
        
        dx,dy,ds = self.length(face,with_dx_dy=True)
        
        p = np.copy(self.cell_pressure_mat[2:self.gc_i_max-2,2:self.gc_j_max-2])

        q0 = np.copy(self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0])
        q1 = np.copy(self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,1])
        q2 = np.copy(self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,2])

        u = q1/q0
        v = q2/q0
        c = np.sqrt(self.gamma*(p/q0))

        u_n = (u*dy) - (v*dx)/ds

        return abs(u_n) + c
    
    # Create method to calculate the first term of the dissipation term using all the previous functions
    def D1(self,nu2):
        d_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)
        s2_xi_lhs = self.s2_xi_face(nu2,face='E')
        length_lhs = self.length(face='E')
        lam_lhs = self.lam(face='E')

        s2_xi_rhs = self.s2_xi_face(nu2,face='W')
        length_rhs = self.length(face='W')
        lam_rhs = self.lam(face='W')

        q_lhs = self.d_xi_q(face='E')
        q_rhs = self.d_xi_q(face='W')

        for i in range(0,4):
            lhs = s2_xi_lhs*length_lhs*lam_lhs*q_lhs[:,:,i]
            rhs = s2_xi_rhs*length_rhs*lam_rhs*q_rhs[:,:,i]
            d_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,i] = lhs - rhs

        return d_mat

    # Create method to calculate the second term of the dissipation term using all the previous functions
    def D2(self,nu2):
        d_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)
        s2_eta_lhs = self.s2_eta_face(nu2,face='N')
        length_lhs = self.length(face='N')
        lam_lhs = self.lam(face='N')

        s2_eta_rhs = self.s2_eta_face(nu2,face='S')
        length_rhs = self.length(face='S')
        lam_rhs = self.lam(face='S')

        q_lhs = self.d_eta_q(face='N')
        q_rhs = self.d_eta_q(face='S')

        for i in range(0,4):
            lhs = s2_eta_lhs*length_lhs*lam_lhs*q_lhs[:,:,i]
            rhs = s2_eta_rhs*length_rhs*lam_rhs*q_rhs[:,:,i]
            d_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,i] = lhs - rhs

        return d_mat
    
    # Create method to calculate the first term of the dissipation term using all the previous functions
    def D3(self,nu2,nu4):
        d_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)
        s2_xi_lhs = self.s4_xi_face(nu2,nu4,face='E')
        length_lhs = self.length(face='E')
        lam_lhs = self.lam(face='E')

        s2_xi_rhs = self.s4_xi_face(nu2,nu4,face='W')
        length_rhs = self.length(face='W')
        lam_rhs = self.lam(face='W')

        q_lhs = self.d_xi_cubed_q(face='E')
        q_rhs = self.d_xi_cubed_q(face='W')

        for i in range(0,4):
            lhs = s2_xi_lhs*length_lhs*lam_lhs*q_lhs[:,:,i]
            rhs = s2_xi_rhs*length_rhs*lam_rhs*q_rhs[:,:,i]
            d_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,i] = lhs - rhs

        return d_mat

    # Create method to calculate the second term of the dissipation term using all the previous functions
    def D4(self,nu2,nu4):
        d_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)
        s2_xi_lhs = self.s4_eta_face(nu2,nu4,face='N')
        length_lhs = self.length(face='N')
        lam_lhs = self.lam(face='N')

        s2_xi_rhs = self.s4_eta_face(nu2,nu4,face='S')
        length_rhs = self.length(face='S')
        lam_rhs = self.lam(face='S')

        q_lhs = self.d_eta_cubed_q(face='N')
        q_rhs = self.d_eta_cubed_q(face='S')

        for i in range(0,4):
            lhs = s2_xi_lhs*length_lhs*lam_lhs*q_lhs[:,:,i]
            rhs = s2_xi_rhs*length_rhs*lam_rhs*q_rhs[:,:,i]
            d_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,i] = lhs - rhs

        return d_mat

# -------------------- End Methods for calculating the dissipation matrix -------------------- #


    # Init dissipation matrix
    def init_dissipation(self,nu2,nu4):
        diss_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape,dtype=object)

        d1_term = self.D1(nu2)
        d2_term = self.D2(nu2)
        d3_term = self.D3(nu2,nu4)
        d4_term = self.D4(nu2,nu4)

        diss_mat = (d1_term + d2_term) - (d3_term + d4_term)
        
        self.diss_mat = np.copy(diss_mat)


    # Boundary conditions for the wall
    def bcWall(self):

        dx,dy,ds = self.length('S',with_dx_dy=True)
        dx_bot = dx[:,0:1]
        dy_bot = dy[:,0:1]
        ds_bot = ds[:,0:1]

        dx,dy,ds = self.length('N',with_dx_dy=True)
        dx_top = dx[:,self.gc_j_max-5:self.gc_j_max-4]
        dy_top = dy[:,self.gc_j_max-5:self.gc_j_max-4]
        ds_top = ds[:,self.gc_j_max-5:self.gc_j_max-4]
        
        real_bot1 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2,:]
        real_bot2 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,3,:]

        real_top1 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,self.gc_j_max - 3,:]
        real_top2 = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,self.gc_j_max - 4,:]

        ghost_bot1 = np.copy(real_bot1)
        ghost_bot2 = np.copy(real_bot2)
        
        ghost_top1 = np.copy(real_top1)
        ghost_top2 = np.copy(real_top2)

        num_u_bot1 = ((real_bot1[:,1:2]*dx_bot) + (real_bot1[:,2:3]*dy_bot))*2*dx_bot
        num_u_bot2 = ((real_bot2[:,1:2]*dx_bot) + (real_bot2[:,2:3]*dy_bot))*2*dx_bot

        num_v_bot1 = ((real_bot1[:,1:2]*dx_bot) + (real_bot1[:,2:3]*dy_bot))*2*dy_bot
        num_v_bot2 = ((real_bot2[:,1:2]*dx_bot) + (real_bot2[:,2:3]*dy_bot))*2*dy_bot

        num_u_top1 = ((real_top1[:,1:2]*dx_top) + (real_top1[:,2:3]*dy_top))*2*dx_top
        num_u_top2 = ((real_top2[:,1:2]*dx_top) + (real_top2[:,2:3]*dy_top))*2*dx_top

        num_v_top1 = ((real_top1[:,1:2]*dx_top) + (real_top1[:,2:3]*dy_top))*2*dy_top
        num_v_top2 = ((real_top2[:,1:2]*dx_top) + (real_top2[:,2:3]*dy_top))*2*dy_top
        
        ghost_bot1[:,1] = ((num_u_bot1/(ds_bot**2)) - real_bot1[:,1:2])[:,0]
        ghost_bot1[:,2] = ((num_v_bot1/(ds_bot**2)) - real_bot1[:,2:3])[:,0]
        self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,1,:] = ghost_bot1

        ghost_bot2[:,1] = ((num_u_bot2/(ds_bot**2)) - real_bot2[:,1:2])[:,0]
        ghost_bot2[:,2] = ((num_v_bot2/(ds_bot**2)) - real_bot2[:,2:3])[:,0]
        self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,0,:] = ghost_bot2

        ghost_top1[:,1] = ((num_u_top1/(ds_top**2)) - real_top1[:,1:2])[:,0]
        ghost_top1[:,2] = ((num_v_top1/(ds_top**2)) - real_top1[:,2:3])[:,0]
        self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,self.gc_j_max - 2,:] = ghost_top1

        ghost_top2[:,1] = ((num_u_top2/(ds_top**2)) - real_top2[:,1:2])[:,0]
        ghost_top2[:,2] = ((num_v_top2/(ds_top**2)) - real_top2[:,2:3])[:,0]
        self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,self.gc_j_max - 1,:] = ghost_top2

        
    
    # Create method to calculate Riem 1
    def riem1(self,i,j):
        q_ij = self.ghost_cell_state_vector_mat[i,j]
        q0 = q_ij[0]
        q1 = q_ij[1]
        q2 = q_ij[2]
        p = self.cell_pressure_mat[i,j]
        

        c = np.sqrt(self.gamma*(p/q0))

        V = np.sqrt((q1**2) + (q2**2))/q0
        return V + (2*c/(self.gamma - 1))
    
    # Create method to calculate Riem 2
    def riem2(self,i,j):
        q_ij = self.ghost_cell_state_vector_mat[i,j]
        q0 = q_ij[0]
        q1 = q_ij[1]
        q2 = q_ij[2]
        p = self.cell_pressure_mat[i,j]

        c = np.sqrt(self.gamma*(p/q0))

        V = np.sqrt((q1**2) + (q2**2))/q0
        return V - (2*c/(self.gamma - 1))
    
    # Create methhod for finding stagnation pressure
    def p0(self,i,j):
        q_ij = self.ghost_cell_state_vector_mat[i,j]
        p = self.cell_pressure_mat[i,j]
        q0 = q_ij[0]
        q1 = q_ij[1]
        q2 = q_ij[2]
        c = np.sqrt(self.gamma*(p/q0))

        V = np.sqrt((q1**2) + (q2**2))/q0
        M = V/c

        p_stag = p*(1 + (((self.gamma - 1)/2)*(M**2)))**(self.gamma/(self.gamma-1))
        return p_stag
    
    # Create method for single cell inlet boundary condition
    def inlet_condition(self,j):
        riem1_minf = self.riem1(1,j)
        riem1_2 = riem1_minf
        p0_2j = self.p0(1,j)

        riem2_3 = self.riem2(3,j)
        riem2_2 = riem2_3

        V_2j = 0.5*(riem1_2 + riem2_2)
        u_2j = V_2j*np.cos(self.inlet_angle)
        v_2j = V_2j*np.sin(self.inlet_angle)
        c_2j = 0.25*(self.gamma - 1)*(riem1_2 - riem2_2)
        M_2j = V_2j/c_2j
        p_2j = p0_2j/((1 + (((self.gamma - 1)/2)*(M_2j**2)))**(self.gamma/(self.gamma-1)))
        rho_2j = (self.gamma * p_2j)/(c_2j**2)
        
        q = np.zeros(shape=(4,))
        q[0] = rho_2j
        q[1] = u_2j * rho_2j
        q[2] = v_2j * rho_2j
        q[3] = (p_2j/(rho_2j*(self.gamma - 1))) + (0.5*(V_2j**2))

        return q

    # Create method for single cell outlet boundary condition
    def outlet_condition(self,j):
        q_icmax_j = np.copy(self.ghost_cell_state_vector_mat[self.gc_i_max - 3,j])
        q_icmaxm1_j = np.copy(self.ghost_cell_state_vector_mat[self.gc_i_max - 4,j])
        q_icmax1_j = (2*q_icmax_j) - q_icmaxm1_j

        p = self.cell_pressure_mat[self.gc_i_max - 3,j]
        q0 = q_icmax_j[0]
        q1 = q_icmax_j[1]
        q2 = q_icmax_j[2]

        c = np.sqrt(self.gamma*(p/q0))

        V = np.sqrt((q1**2) + (q2**2))/q0
        M = V/c

        if M >= 1:
            p_forE = self.cell_pressure_mat[self.gc_i_max - 4,j]
        elif 0 <= M < 1:
            p_forE = self.cell_pressure_mat[self.gc_i_max - 1,j]
        else:
            raise ValueError
        
        rhoE = (p_forE/(self.gamma - 1)) + (0.5*(V**2))

        q_icmax1_j[3] = rhoE
        return q_icmax1_j

    # Inlet and outlet boundary conditions
    def bcInletOutlet(self):
        for j in range(0,self.gc_j_max):

            # Inlet boundary conditions
            q_2_j = self.inlet_condition(j)
            self.ghost_cell_state_vector_mat[2,j] = np.copy(q_2_j)

            # Outlet boundary condition
            q_icmax1_j = self.outlet_condition(j)
            self.ghost_cell_state_vector_mat[self.gc_i_max - 2,j] = np.copy(q_icmax1_j)
        
        self.update_pressures()
    
    # Create method for timestep
    def max_timestep(self):
        area = self.cell_area_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
        lam1 = self.lam('N')
        lam2 = self.lam('S')
        lam3 = self.lam('E')
        lam4 = self.lam('W')
        length1 = self.length('N')
        length2 = self.length('S')
        length3 = self.length('E')
        length4 = self.length('W')

        denom = (lam1*length1) + (lam2*length2) * (lam3*length3) + (lam4*length4)
        num =  2*area
        dt = num/denom
        dt_mat = self.cfl*dt
        
        self.dt_mat = np.copy(dt_mat)

    # Create method for calculating the force on the bump
    def compute_force_bot_bump(self):
        dx,dy,ds = self.length('S',with_dx_dy=True)
        dx_bot = dx[:,0:1]
        dy_bot = dy[:,0:1]
        ds_bot = ds[:,0:1]
        p_bot = self.cell_pressure_mat[2:self.gc_i_max-2,2]
        force_x = 0
        force_y = 0
        for i in range(len(dx_bot)):
            if abs(dy_bot[i]) > 0:
                force_x += p_bot[i]*dx_bot[i]
                force_y += p_bot[i]*dy_bot[i]

        return force_x,force_y
        


    def runge_kutta(self,tol,nu2,nu4):
        diff = 1
        count = 0
        alphas = [1/4,1/3,1/2,1]
        diff1_list = []
        diff2_list = []
        diff3_list = []
        diff4_list = []

        while diff >= tol:

            count += 1
            print(f'iter {count}')
            diff -= 1e-2

            self.max_timestep()
            self.init_fluxes()
            self.init_residuals()
            self.init_dissipation(nu2,nu4)

            qrk_mat = np.copy(self.ghost_cell_state_vector_mat)[2:self.gc_i_max-2,2:self.gc_j_max-2]
            q_new = np.copy(self.ghost_cell_state_vector_mat)[2:self.gc_i_max-2,2:self.gc_j_max-2]
            for alpha in alphas:
                q0_ij = qrk_mat
                dt = self.dt_mat
                A_ij = self.cell_area_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
                R_ij = self.residuals[2:self.gc_i_max-2,2:self.gc_j_max-2]
                D_ij = self.diss_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
                for i in range(0,4):
                    q_new[:,:,i] = q0_ij[:,:,i] - ((alpha*dt/A_ij)*(R_ij[:,:,i]-D_ij[:,:,i]))

                self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2] = np.copy(q_new)
                self.bcWall()
                self.bcInletOutlet()
                self.init_fluxes()
                self.init_residuals()
            
            diff_mat = abs(q_new - qrk_mat)
            diff1_list.append(np.log(diff_mat[1:,:,0].mean()))
            diff2_list.append(np.log(diff_mat[1:,:,1].mean()))
            diff3_list.append(np.log(diff_mat[1:,:,2].mean()))
            diff4_list.append(np.log(diff_mat[1:,:,3].mean()))

            if count % 10000 == 0:
                print(diff_mat[:,:,0])
                print(diff_mat[:,:,0].shape)
                print(diff_mat[0].max())
                print(diff_mat[0].min())
                print(diff_mat[0].mean())
                x = self.grid[0]
                y = self.grid[1]
                z = self.cell_pressure_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
                rho = self.ghost_cell_state_vector_mat[2:self.gc_i_max-2,2:self.gc_j_max-2,0]
                rho_residual = diff_mat[:,:,3]
                plot_grid(x,y,'hot_r',z,title='Pressure')
                plot_grid(x,y,'hot_r',rho,title='Density')
                plot_grid(x,y,'hot_r',rho_residual,title='Density Residuals') 
                plt.show()
        
        self.bcWall()
        self.bcInletOutlet()

        self.diff1_list = diff1_list
        self.diff2_list = diff2_list
        self.diff3_list = diff3_list
        self.diff4_list = diff4_list