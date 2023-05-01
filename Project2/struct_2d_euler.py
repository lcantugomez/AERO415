# Import libraries
import numpy as np
from plotting_func import plot_grid
import matplotlib.pyplot as plt
import time


# Init solver class
class Struct2DEulerChannel():
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
        self.cell_grid_mat = None
        self.cell_pressure_mat = None
        self.residuals = None
        self.diss_mat = None
        self.dt_mat = None
        self.resid1_list = None
        self.resid2_list = None
        self.resid3_list = None
        self.resid4_list = None


    # Init and calculate cell areas, init cell grid, init cell pressure matrix, and init ghost cells state matrix
    def create_cell_matrices(self):
        cell_area_mat = np.zeros(shape=(self.gc_i_max,self.gc_j_max),dtype=object)
        ghost_cell_state_vector_mat = np.zeros(shape=(self.gc_i_max, self.gc_j_max),dtype=object)
        cell_grid_mat = np.zeros(shape=(self.gc_i_max,self.gc_j_max),dtype=object)
        cell_pressure_mat = np.zeros(shape=(self.gc_i_max, self.gc_j_max),dtype=float)

        for j in range(2,self.gc_j_max-2):
            for i in range(2,self.gc_i_max-2):
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
                
                n1 = (x1,y1)
                n2 = (x2,y2)
                n3 = (x3,y3)
                n4 = (x4,y4)

                local_nodes = [n1,n2,n3,n4]

                area_ij = 0.5*(((x4-x1)*(y3-y2)) - ((y4-y1)*(x3-x2)))
                cell_area_mat[i,j] = area_ij
                cell_grid_mat[i,j] = local_nodes

        for j in range(0,self.gc_j_max):
            for i in range(0,self.gc_i_max):

                q0 = self.qminf[0,0]
                q1 = self.qminf[1,0]
                q2 = self.qminf[2,0]
                q3 = self.qminf[3,0]

                ghost_cell_state_vector_mat[i,j] = np.matrix([[q0],[q1],[q2],[q3]])

                p = (self.gamma - 1)*(q3 - (0.5*((q1**2) + (q2**2)/q0)))
                cell_pressure_mat[i,j] = p

        self.cell_area_mat = np.copy(cell_area_mat)
        self.ghost_cell_state_vector_mat = np.copy(ghost_cell_state_vector_mat)
        self.cell_grid_mat = np.copy(cell_grid_mat)
        self.cell_pressure_mat = np.copy(cell_pressure_mat)
    
    # Create method for updating pressures
    def update_pressures(self):
        for j in range(0,self.gc_j_max):
            for i in range(0,self.gc_i_max):
                q_ij = self.ghost_cell_state_vector_mat[i,j]
                q0 = q_ij[0,0]
                q1 = q_ij[1,0]
                q2 = q_ij[2,0]
                q3 = q_ij[3,0]

                p = (self.gamma - 1)*(q3 - (0.5*((q1**2) + (q2**2)/q0)))
                if p <= 0:
                    print(q3,'\n',q1,q2,q0,'\n',i,j,'\n\n')
                    raise ValueError
                self.cell_pressure_mat[i,j] = p
    
    # Init fluxes
    def init_fluxes(self):
        f_ghost_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape,dtype=object)
        g_ghost_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape,dtype=object)

        for j in range(0,self.gc_j_max):
            for i in range(0,self.gc_i_max):
                q_ij = self.ghost_cell_state_vector_mat[i,j]

                p = self.cell_pressure_mat[i,j]
                q0 = q_ij[0,0]
                q1 = q_ij[1,0]
                q2 = q_ij[2,0]
                q3 = q_ij[3,0]

                f0 = q1
                f1 = ((q1**2)/q0) + p
                f2 = (q1*q2)/q0
                f3 = (q1/q0)*(q3 + p)

                f = np.matrix([[f0],[f1],[f2],[f3]])

                g0 = q2
                g1 = (q1*q2)/q0
                g2 = ((q2**2)/q0) + p
                g3 = (q2/q0)*(q3 + p)

                g = np.matrix([[g0],[g1],[g2],[g3]])


                f_ghost_mat[i,j] = f
                g_ghost_mat[i,j] = g

        self.f_ghost_mat = np.copy(f_ghost_mat)
        self.g_ghost_mat = np.copy(g_ghost_mat)
    

    # Init residual matrix
    def init_residuals(self):
        residuals = np.zeros(shape=self.cell_grid_mat.shape,dtype=object)
        for j in range(2,self.gc_j_max-2):
            for i in range(2,self.gc_i_max-2):
                
                f_i_j = self.f_ghost_mat[i,j]
                f_i1_j =  self.f_ghost_mat[i+1,j]
                f_i_j1 = self.f_ghost_mat[i,j+1]
                f_m1i_j = self.f_ghost_mat[i-1,j]
                f_i_m1j = self.f_ghost_mat[i,j-1]

                g_i_j = self.g_ghost_mat[i,j]
                g_i1_j =  self.g_ghost_mat[i+1,j]
                g_i_j1 = self.g_ghost_mat[i,j+1]
                g_m1i_j = self.g_ghost_mat[i-1,j]
                g_i_m1j = self.g_ghost_mat[i,j-1]
                
                local_nodes = self.cell_grid_mat[i,j]

                x1,y1= local_nodes[0]
                x2,y2= local_nodes[1]
                x3,y3= local_nodes[2]
                x4,y4= local_nodes[3]

                dy_ihalf_j = y4-y2
                dx_ihalf_j = x4-x2

                dy_i_jhalf = y3-y4
                dx_i_jhalf = x3-x4

                dy_imhalf_j = y1-y3
                dx_imhalf_j = x1-x3

                dy_i_jmhalf = y2-y1
                dx_i_jmhalf = x2-x1

                f_ij_E_avg = 0.5*(f_i_j + f_i1_j)*dy_ihalf_j
                f_ij_W_avg = 0.5*(f_i_j + f_m1i_j)*dy_imhalf_j
                f_ij_N_avg = 0.5*(f_i_j + f_i_j1)*dy_i_jhalf
                f_ij_S_avg = 0.5*(f_i_j + f_i_m1j)*dy_i_jmhalf

                g_ij_E_avg = 0.5*(g_i_j + g_i1_j)*dx_ihalf_j
                g_ij_W_avg = 0.5*(g_i_j + g_m1i_j)*dx_imhalf_j
                g_ij_N_avg = 0.5*(g_i_j + g_i_j1)*dx_i_jhalf
                g_ij_S_avg = 0.5*(g_i_j + g_i_m1j)*dx_i_jmhalf

                R_ij = (f_ij_E_avg - g_ij_E_avg) + (f_ij_N_avg - g_ij_N_avg) + (f_ij_W_avg - g_ij_W_avg) + (f_ij_S_avg - g_ij_S_avg)

                residuals[i,j] = R_ij
                """if j == self.gc_j_max-3:
                    print(R_ij,i,j,'\n\n')"""
        
        self.residuals = np.copy(residuals)


# -------------------- Begin Methods for calculating the dissipation matrix -------------------- #

    # Create method for length scale
    def length(self,i,j,face,with_dx_dy = False):
        local_nodes = self.cell_grid_mat[i,j]
        if face.upper() == 'N':
            x1,y1 = local_nodes[3]
            x2,y2 = local_nodes[2]
        elif face.upper() == 'S':
            x1,y1 = local_nodes[0]
            x2,y2 = local_nodes[1]
        elif face.upper() == 'E':
            x1,y1 = local_nodes[1]
            x2,y2 = local_nodes[3]
        elif face.upper() == 'W':
            x1,y1 = local_nodes[2]
            x2,y2 = local_nodes[0]
        else:
            raise ValueError

        dx_sqrd = (x2-x1)**2
        dy_sqrd = (y2-y1)**2
        face_length = np.sqrt(dx_sqrd + dy_sqrd)

        if with_dx_dy:
            dx = x2-x1
            dy = y2-y1
            return dx,dy,face_length
        else:
            return face_length

    # Create method for delta_xi operation on q
    def d_xi_q(self,i,j):
        q1 = self.ghost_cell_state_vector_mat[i,j]
        q2 = self.ghost_cell_state_vector_mat[i+1,j]
        return q2-q1

    # Create method for delta_eta operation on q
    def d_eta_q(self,i,j):
        q1 = self.ghost_cell_state_vector_mat[i,j]
        q2 = self.ghost_cell_state_vector_mat[i,j+1]
        return q2-q1
    
    # Create method for delta_xi^2 operation on q
    def d_xi_sqrd_q(self,i,j):
        q1 = self.ghost_cell_state_vector_mat[i+1,j]
        q2 = self.ghost_cell_state_vector_mat[i,j]
        q3 = self.ghost_cell_state_vector_mat[i-1,j]
        return np.copy(q1 - (2*q2) + q3)

    # Create method for delta_eta^2 operation on q
    def d_eta_sqrd_q(self,i,j):
        q1 = self.ghost_cell_state_vector_mat[i,j+1]
        q2 = self.ghost_cell_state_vector_mat[i,j]
        q3 = self.ghost_cell_state_vector_mat[i,j-1]
        return np.copy(q1 - (2*q2) + q3)
    
    # Create method for delta_xi^2 operation on pressure
    def d_xi_sqrd_p(self,i,j):
        p1 = self.cell_pressure_mat[i+1,j]
        p2 = self.cell_pressure_mat[i,j]
        p3 = self.cell_pressure_mat[i-1,j]
        return p1 - (2*p2) + p3

    # Create method for delta_eta^2 operation on pressure
    def d_eta_sqrd_p(self,i,j):
        p1 = self.cell_pressure_mat[i,j+1]
        p2 = self.cell_pressure_mat[i,j]
        p3 = self.cell_pressure_mat[i,j-1]
        return p1 - (2*p2) + p3

    # Create method for delta_xi^3 operation on q
    def d_xi_cubed_q(self,i,j):
        q1 = self.d_xi_sqrd_q(i,j)
        q2 = self.d_xi_sqrd_q(i+1,j)
        return np.copy(q2 - q1)

    # Create method for delta_eta^3 operation on q
    def d_eta_cubed_q(self,i,j):
        q1 = self.d_eta_sqrd_q(i,j)
        q2 = self.d_eta_sqrd_q(i,j+1)
        return np.copy(q2 - q1)
    
    # Create method for s2 switch in xi for the cell
    def s2_xi_cell(self,i,j,nu2):
        num = abs(self.d_xi_sqrd_p(i,j))
        p1 = self.cell_pressure_mat[i+1,j]
        p2 = self.cell_pressure_mat[i,j]
        p3 = self.cell_pressure_mat[i-1,j]
        denom = p1 + (2*p2) + p3
        s2 = nu2 * (num/denom)
        return s2

    # Create method for s2 switch in eta for the cell
    def s2_eta_cell(self,i,j,nu2):
        num = abs(self.d_eta_sqrd_p(i,j))
        p1 = self.cell_pressure_mat[i,j+1]
        p2 = self.cell_pressure_mat[i,j]
        p3 = self.cell_pressure_mat[i,j-1]
        denom = p1 + (2*p2) + p3
        s2 = nu2 * (num/denom)
        return s2

    # Create method for s2 switch in xi at the face
    def s2_xi_face(self,i,j,nu2):
        s2 = 0.5*(self.s2_xi_cell(i,j,nu2) + self.s2_xi_cell(i+1,j,nu2))
        return s2

    # Create method for s2 switch in eta at the face
    def s2_eta_face(self,i,j,nu2):
        s2 = 0.5*(self.s2_eta_cell(i,j,nu2) + self.s2_eta_cell(i,j+1,nu2))
        return s2
    
    # Create method for s4 switch in xi at the face
    def s4_xi_face(self,i,j,nu2,nu4):
        s4 = nu4 - self.s2_xi_face(i,j,nu2)
        if s4 < 0:
            s4 = 0
        return s4

    # Create method for s4 switch in eta at the face
    def s4_eta_face(self,i,j,nu2,nu4):
        s4 = nu4 - self.s2_eta_face(i,j,nu2)
        if s4 < 0:
            s4 = 0
        return s4

    # Create method for eigen values at constant xi face
    def lam(self,i,j,face):
        
        dx,dy,ds = self.length(i,j,face,with_dx_dy=True)

        q = self.ghost_cell_state_vector_mat[i,j]
        p = self.cell_pressure_mat[i,j]

        rho = q[0,0]
        u = q[1,0]/rho
        v = q[2,0]/rho
        c = np.sqrt(self.gamma*(p/rho))

        u_n = (u*dy) - (v*dx)/ds

        return abs(u_n) + c
    
    # Create method to calculate the first term of the dissipation term using all the previous functions
    def D1(self,i,j,nu2):
        lhs = self.s2_xi_face(i,j,nu2)*self.length(i,j,'E')*self.lam(i,j,'E')*self.d_xi_q(i,j)
        rhs = self.s2_xi_face(i-1,j,nu2)*self.length(i,j,'W')*self.lam(i,j,'W')*self.d_xi_q(i-1,j)
        return lhs - rhs

    # Create method to calculate the second term of the dissipation term using all the previous functions
    def D2(self,i,j,nu2):
        lhs = self.s2_eta_face(i,j,nu2)*self.length(i,j,'N')*self.lam(i,j,'N')*self.d_eta_q(i,j)
        rhs = self.s2_eta_face(i,j-1,nu2)*self.length(i,j,'S')*self.lam(i,j,'S')*self.d_eta_q(i,j-1)
        return lhs - rhs
    
    # Create method to calculate the first term of the dissipation term using all the previous functions
    def D3(self,i,j,nu2,nu4):
        lhs = self.s4_xi_face(i,j,nu2,nu4)*self.length(i,j,'E')*self.lam(i,j,'E')*self.d_xi_cubed_q(i,j)
        rhs = self.s4_xi_face(i-1,j,nu2,nu4)*self.length(i,j,'W')*self.lam(i,j,'W')*self.d_xi_cubed_q(i-1,j)
        return lhs - rhs

    # Create method to calculate the second term of the dissipation term using all the previous functions
    def D4(self,i,j,nu2,nu4):
        lhs = self.s4_eta_face(i,j,nu2,nu4)*self.length(i,j,'N')*self.lam(i,j,'N')*self.d_eta_cubed_q(i,j)
        rhs = self.s4_eta_face(i,j-1,nu2,nu4)*self.length(i,j,'S')*self.lam(i,j,'S')*self.d_eta_cubed_q(i,j-1)
        return lhs - rhs

# -------------------- End Methods for calculating the dissipation matrix -------------------- #


    # Init dissipation matrix
    def init_dissipation(self,nu2,nu4):
        diss_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape,dtype=object)

        for j in range(2, self.gc_j_max - 2):
            for i in range(2, self.gc_i_max - 2):

                d1_term = self.D1(i,j,nu2)
                d2_term = self.D2(i,j,nu2)
                d3_term = self.D3(i,j,nu2,nu4)
                d4_term = self.D4(i,j,nu2,nu4)
                diss_mat[i,j] = (d1_term + d2_term) - (d3_term + d4_term)
        
        self.diss_mat = np.copy(diss_mat)


    # Create method for mirroring the velocities at the walls
    def u_mirrored(self,u,v,dx,dy,ds):
        num = ((u*dx) + (v*dy))*2*dx
        denom = ds**2
        return (num/denom) - u
    
    # Create method for mirroring the velocities at the walls
    def v_mirrored(self,u,v,dx,dy,ds):
        num = ((u*dx) + (v*dy))*2*dy
        denom = ds**2
        return (num/denom) - v

    # Boundary conditions for the wall
    def bcWall(self):
        for i in range(2, self.gc_i_max - 2):
            dom_cell_1_bot = self.ghost_cell_state_vector_mat[i,2]
            dom_cell_2_bot = self.ghost_cell_state_vector_mat[i,3]
            ghost_cell_1_bot = np.copy(dom_cell_1_bot)
            ghost_cell_2_bot = np.copy(dom_cell_2_bot)

            dom_cell_1_top = self.ghost_cell_state_vector_mat[i,self.gc_j_max - 3]
            dom_cell_2_top = self.ghost_cell_state_vector_mat[i,self.gc_j_max - 4]
            ghost_cell_1_top = np.copy(dom_cell_1_top)
            ghost_cell_2_top = np.copy(dom_cell_2_top)

            dx1_bot,dy1_bot,ds1_bot = self.length(i,2,'S',with_dx_dy=True)

            dx1_top,dy1_top,ds1_top = self.length(i,self.gc_j_max - 3,'N',with_dx_dy=True)
            
            u_bot1 = dom_cell_1_bot[1,0]
            v_bot1 = dom_cell_1_bot[2,0]
            u_bot2 = dom_cell_2_bot[1,0]
            v_bot2 = dom_cell_2_bot[2,0]

            u_top1 = dom_cell_1_top[1,0]
            v_top1 = dom_cell_1_top[2,0]
            u_top2 = dom_cell_2_top[1,0]
            v_top2 = dom_cell_2_top[2,0]

            u_ghost_bot1 = self.u_mirrored(u_bot1,v_bot1,dx1_bot,dy1_bot,ds1_bot)
            u_ghost_bot2 = self.u_mirrored(u_bot2,v_bot2,dx1_bot,dy1_bot,ds1_bot)
            v_ghost_bot1 = self.v_mirrored(u_bot1,v_bot1,dx1_bot,dy1_bot,ds1_bot)
            v_ghost_bot2 = self.v_mirrored(u_bot2,v_bot2,dx1_bot,dy1_bot,ds1_bot)

            u_ghost_top1 = self.u_mirrored(u_top1,v_top1,dx1_top,dy1_top,ds1_top)
            u_ghost_top2 = self.u_mirrored(u_top2,v_top2,dx1_top,dy1_top,ds1_top)
            v_ghost_top1 = self.v_mirrored(u_top1,v_top1,dx1_top,dy1_top,ds1_top)
            v_ghost_top2 = self.v_mirrored(u_top2,v_top2,dx1_top,dy1_top,ds1_top)

            ghost_cell_1_bot[1,0] = u_ghost_bot1
            ghost_cell_1_bot[2,0] = v_ghost_bot1
            ghost_cell_2_bot[1,0] = u_ghost_bot2
            ghost_cell_2_bot[2,0] = v_ghost_bot2

            ghost_cell_1_top[1,0] = u_ghost_top1
            ghost_cell_1_top[2,0] = v_ghost_top1
            ghost_cell_2_top[1,0] = u_ghost_top2
            ghost_cell_2_top[2,0] = v_ghost_top2

            self.ghost_cell_state_vector_mat[i,1] = np.copy(ghost_cell_1_bot)
            self.ghost_cell_state_vector_mat[i,0] = np.copy(ghost_cell_2_bot)

            self.ghost_cell_state_vector_mat[i,self.gc_j_max - 2] = np.copy(ghost_cell_1_top)
            self.ghost_cell_state_vector_mat[i,self.gc_j_max - 1] = np.copy(ghost_cell_2_top)
    
    # Create method to calculate Riem 1
    def riem1(self,i,j):
        q_ij = self.ghost_cell_state_vector_mat[i,j]
        q0 = q_ij[0,0]
        q1 = q_ij[1,0]
        q2 = q_ij[2,0]
        p = self.cell_pressure_mat[i,j]

        c = np.sqrt(self.gamma*(p/q0))

        V = np.sqrt((q1**2) + (q2**2))/q0
        return V + (2*c/(self.gamma - 1))
    
    # Create method to calculate Riem 2
    def riem2(self,i,j):
        q_ij = self.ghost_cell_state_vector_mat[i,j]
        q0 = q_ij[0,0]
        q1 = q_ij[1,0]
        q2 = q_ij[2,0]
        p = self.cell_pressure_mat[i,j]

        c = np.sqrt(self.gamma*(p/q0))

        V = np.sqrt((q1**2) + (q2**2))/q0
        return V - (2*c/(self.gamma - 1))
    
    # Create methhod for finding stagnation pressure
    def p0(self,i,j):
        q_ij = self.ghost_cell_state_vector_mat[i,j]
        p = self.cell_pressure_mat[i,j]
        q0 = q_ij[0,0]
        q1 = q_ij[1,0]
        q2 = q_ij[2,0]
        c = np.sqrt(self.gamma*(p/q0))

        V = np.sqrt((q1**2) + (q2**2))/q0
        M = V/c

        p_stag = p*(1 + (((self.gamma - 1)/2)*(M**2)))**(self.gamma/(self.gamma-1))
        return p_stag
    
    # Create method for single cell inlet boundary condition
    def inlet_condition(self,j):
        riem1_minf = self.riem1(0,j)
        riem1_2 = riem1_minf
        p0_2j = self.p0(0,j)

        riem2_3 = self.riem2(3,j)
        riem2_2 = riem2_3

        V_2j = 0.5*(riem1_2 + riem2_2)
        u_2j = V_2j*np.cos(self.inlet_angle)
        v_2j = V_2j*np.sin(self.inlet_angle)
        c_2j = 0.25*(self.gamma - 1)*(riem1_2 - riem2_2)
        M_2j = V_2j/c_2j
        p_2j = p0_2j/((1 + (((self.gamma - 1)/2)*(M_2j**2)))**(self.gamma/(self.gamma-1)))
        rho_2j = (self.gamma * p_2j)/(c_2j**2)
        
        """print(M_2j)
        print(p_2j)
        print(rho_2j,'\n\n')"""
        q0 = rho_2j
        q1 = u_2j * rho_2j
        q2 = v_2j * rho_2j
        q3 = (p_2j/(rho_2j*(self.gamma - 1))) + (0.5*(V_2j**2))
        q = np.matrix([[q0],[q1],[q2],[q3]])

        return q

    # Create method for single cell outlet boundary condition
    def outlet_condition(self,j):
        q_icmax_j = np.copy(self.ghost_cell_state_vector_mat[self.gc_i_max - 3,j])
        q_icmaxm1_j = np.copy(self.ghost_cell_state_vector_mat[self.gc_i_max - 4,j])
        q_icmax1_j = (2*q_icmax_j) - q_icmaxm1_j

        p = self.cell_pressure_mat[self.gc_i_max - 3,j]
        q0 = q_icmax_j[0,0]
        q1 = q_icmax_j[1,0]
        q2 = q_icmax_j[2,0]

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

        q_icmax1_j[3,0] = rhoE
        return q_icmax1_j

    # Inlet boundary conditions
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
        dt_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape,dtype=float)
        for j in range(2,self.gc_j_max - 2):
            for i in range(2,self.gc_i_max - 2):
                area = self.cell_area_mat[i,j]
                lam1 = self.lam(i,j,'N')
                lam2 = self.lam(i,j,'S')
                lam3 = self.lam(i,j,'E')
                lam4 = self.lam(i,j,'W')
                length1 = self.length(i,j,'N')
                length2 = self.length(i,j,'S')
                length3 = self.length(i,j,'E')
                length4 = self.length(i,j,'W')

                denom = (lam1*length1) + (lam2*length2) * (lam3*length3) + (lam4*length4)
                num =  2*area
                dt = num/denom
                dt_mat[i,j] = self.cfl*dt
        
        self.dt_mat = np.copy(dt_mat)




    def runge_kutta(self,tol,nu2,nu4):
        diff = 1
        count = 0
        alphas = [1/4,1/3,1/2,1]

        resid1_list = []
        resid2_list = []
        resid3_list = []
        resid4_list = []

        diff_mat = np.ones(shape=self.ghost_cell_state_vector_mat.shape)
        diff_test = diff_mat.min()
        while diff >= tol:

            resid1_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)
            resid2_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)
            resid3_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)
            resid4_mat = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)

            count += 1
            print(f'iter {count}')
            diff -= 0.001

            self.max_timestep()
            self.init_fluxes()
            self.init_residuals()
            self.init_dissipation(nu2,nu4)
            test_mat1 = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)
            test_mat2 = np.zeros(shape=self.ghost_cell_state_vector_mat.shape)

            qrk_mat = np.copy(self.ghost_cell_state_vector_mat)
            q_new_mat = np.copy(self.ghost_cell_state_vector_mat)
            for alpha in alphas:
                for j in range(2,self.gc_j_max - 2):
                    for i in range(2,self.gc_i_max - 2):
                        q0_ij = qrk_mat[i,j]
                        dt = self.dt_mat[i,j]
                        A_ij = self.cell_area_mat[i,j]
                        R_ij = self.residuals[i,j]
                        D_ij = self.diss_mat[i,j]
                        q_new_ij = q0_ij - ((alpha*dt/A_ij)*(R_ij-D_ij))
                        q_new_mat[i,j] = q_new_ij
                        diff_ij = abs(q_new_ij - q0_ij).mean()
                        diff_mat[i,j] = diff_ij
                        test_mat1[i,j] = R_ij[3,0]
                        test_mat2[i,j] = D_ij[3,0]
                        
                        resid1_mat[i,j] = R_ij[0,0]
                        resid2_mat[i,j] = R_ij[1,0]
                        resid3_mat[i,j] = R_ij[2,0]
                        resid4_mat[i,j] = R_ij[3,0]

                self.ghost_cell_state_vector_mat = np.copy(q_new_mat)
                self.bcWall()
                self.bcInletOutlet()
                self.init_fluxes()
                self.init_residuals()
            
            diff_test = diff_mat[2:self.gc_i_max - 2,2:self.gc_j_max - 2].mean()
            print(diff_test)
            resid1_list.append(resid1_mat[2:self.gc_j_max - 2,2:self.gc_i_max - 2].mean())
            resid2_list.append(resid2_mat[2:self.gc_j_max - 2,2:self.gc_i_max - 2].mean())
            resid3_list.append(resid3_mat[2:self.gc_j_max - 2,2:self.gc_i_max - 2].mean())
            resid4_list.append(resid4_mat[2:self.gc_j_max - 2,2:self.gc_i_max - 2].mean())
            
            if count % 50 == 0:
                x = self.grid[0]
                y = self.grid[1]
                z = self.cell_pressure_mat[2:self.gc_i_max-2,2:self.gc_j_max-2]
                z2 = test_mat1[2:self.gc_i_max-2,2:self.gc_j_max-2]
                z3 = test_mat2[2:self.gc_i_max-2,2:self.gc_j_max-2]
                print(z)
                plot_grid(x,y,'hot_r',z)
                plot_grid(x,y,'seismic',z2,title='Residual 3')
                plot_grid(x,y,'seismic',z3,title='Diss 3')
                plt.show()
        
        self.bcWall()
        self.bcInletOutlet()
        self.resid1_list = resid1_list
        self.resid2_list = resid2_list
        self.resid3_list = resid3_list
        self.resid4_list = resid4_list