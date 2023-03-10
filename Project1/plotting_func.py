
# Global imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Create funtion that does plotting and autoscaling for a mesh type array
def plot_grid(x,y,filename=None,save_fig=False,ax=None,title = None, **kwargs):

    # Pass existing figure or create new one
    ax = ax or plt.gca()

    # Set title if given
    if title != None:
        ax.set_title(str(title))

    # Add both x and y segments and directions
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)

    # Drawing line segments connecting the x and y points
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))

    # Autoscale the axis size
    ax.autoscale()

    # Set width to be 5 times the height
    ax.figure.set_size_inches(20,4)

    # If saving the figure, name it the number or name of the figure
    if save_fig and filename != None:
        if type(filename) == str:
            plt.savefig(f'{filename}.png')
        else:
            plt.savefig(f'Fig{filename}.png')
        plt.close()
    
    # Raise error if not fig name or file name is passed
    elif save_fig and filename == None:
        raise Exception('No figure number was provided to save the file appropriately')