
# Global Imports
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import time
import pandas as pd

# Local Imports
from plotting_func import plot_grid
from algebraic_grid import create_alg_grid
from pde_grid import PDE_Grid

# Init arrays for generating tables and grids
i_max_arr = [80,100,120,140]
j_max_arr = [6,8,12,16,20,24,28]

print('Would you like to plot only one grid of custom size?\n')
print('----------------------------WARNING---------------------------------------')
print('Selecting no will run the code to provide all the data used in the report.\n This will take upwards of 100,000 iterations of the PDE solver.')
print('--------------------------------------------------------------------------')
selction = input('(y/n): ')

if selction.upper() == 'Y':
    i_max_arr = [int(input('I Max = '))]
    j_max_arr = [int(input('J Max = '))]
    table_bool = False
else:
    print('Are you sure you wish to proceed. This will take a while...')
    selction = input('(y/n): ')
    if selction.upper() == 'Y':
        table_bool = True
    else:
        i_max_arr = [int(input('I Max = '))]
        j_max_arr = [int(input('J Max = '))]
        table_bool = False

# Init the tables for the report
table1 = pd.DataFrame()
table2 = pd.DataFrame()

# Boundary conditions on lower and upper wall
y_lower = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: (0.15*np.sin((t-2)*np.pi)), lambda t: 0])
y_upper = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:1, lambda t: 1 - (0.15*np.sin((t-2)*np.pi)), lambda t: 1])
yp_lower = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: (0.15*np.pi*np.cos((t-2)*np.pi)), lambda t: 0])
yp_upper = lambda x: np.piecewise(x,[x<=2,(x>2)*(x<3),x>=3],[lambda t:0, lambda t: -(0.15*np.pi*np.cos((t-2)*np.pi)), lambda t: 0])

# Boundary conditions on left and right wall
x_left = 0
x_right = 5

# Init tolerance arrays for table in report
tol_arr = [10**-4,10**-5,10**-6,10**-7,10**-8,10**-9,10**-10,10**-11,10**-12]

# Add fiest columns for both tables
table1['Tolerance Values'] = tol_arr
table2['I_max Values'] = i_max_arr

# Array for tolerance comparison
iter1_arr = []

# Iterating first through j_max since those are hte columns of the table
for j_max in j_max_arr:

    # Arays for columns for tables 1 and 2 respectively
    iter2_arr = []

    for i_max in i_max_arr:
        
        # Start a timer
        start_time = time.time()

        # Name the file for the figures used in the report
        filename = f'Imax{i_max}_Jmax{j_max}_grid'

        # Defining delta xi and delta eta
        d_exi = 1/(i_max-1)
        d_eta = 1/(j_max-1)

        # Create algebraic grid and plot with the title
        plot_title = f'I Max = {i_max} and J Max = {j_max}'
        print(plot_title)

        x_pts,y_pts = create_alg_grid(i_max,j_max,x_left,x_right,y_lower,y_upper)
        plot_grid(x_pts,y_pts,title=plot_title,color="lightgrey")

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
        if i_max == 80 and j_max == 16:
            print('-----------Begin saving tolerance comparisons-----------\n')
            for tol in tol_arr:

                # Generate Grid and save iteration count
                nx_pts,ny_pts,iters1 = grid_gen.pde_grid(tol,False)
                iter1_arr.append(iters1)

                delta_t = round(time.time() - start_time,5)
                print(f'Time to complete  = {delta_t} seconds\n')

                # Conditional to catch the tolerance = 10^-6 case
                if tol == 10**-6:
                    plot_grid(nx_pts,ny_pts,save_fig=True,filename=filename)
                    iter2_arr.append(iters1)

            print('-----------End saving tolerance comparisons-----------\n')
        else:

            # Generate Grid, save iteration time, plot the figure, and save the file (happens in plot_grid function)
            nx_pts,ny_pts,iters2 = grid_gen.pde_grid(10**-6,False)
            iter2_arr.append(iters2)

            delta_t = round(time.time() - start_time,5)
            print(f'Time to complete  = {delta_t} seconds\n')

            plot_grid(nx_pts,ny_pts,save_fig=True,filename=filename)

    if table_bool:
        table2[f'Jmax = {j_max}'] = iter2_arr


if table_bool:
    table1['Iterations'] = iter1_arr

    writer = pd.ExcelWriter('Tables_415_Project1.xlsx',engine="xlsxwriter")

    table1.to_excel(writer,'Table1',index=False)
    table2.to_excel(writer,'Table2',index=False)

    writer.save()
else:
    plt.show()