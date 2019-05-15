import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import matplotlib.cm as cmx


def plot_size(figsize, rotation = None):
    def decorator(f):
        def func(*args, **kwargs):
            plt.figure(figsize = figsize)
            f(*args, **kwargs)
            if rotation is not None:
                plt.yticks(rotation = 0)
        return func
    return decorator


def grid(cluster):
    """deprecated."""
    fig, ax = plt.subplots()
    ax.axis('off')
    
    pattern = list(cluster)[-1]
    n_days = len(pattern)
    #cluster = cluster.drop(columns = 'pattern')
    cluster.pop('pattern')
    
    mapping = cmx.ScalarMappable(norm = colors.Normalize(0, 100), cmap = plt.get_cmap('Reds'))
    color = lambda x : mapping.to_rgba(int(x)) if x > 1 else 'white'
    
    d = np.zeros((n_days, 24))
    for (x, y), val in cluster.items():
        d[x - 1][y] = val
        
    table = ax.table(
        rowLabels = ['day_{}'.format(i + 1) for i in range(n_days)],
        colLabels = range(24),
        cellColours = [
            [color(value * 100) for value in x] for x in d
        ],
        rowLoc = 'right'
    )
    
    for cell in table.properties()['child_artists']:
        cell.set_width(0.1)
        cell.set_height(0.15)
        
      
@plot_size((15, 6), rotation = 0)
def heatmap(cluster, legend = False):
    """produces a cluster heatmap based on seaborn.heatmap"""
    pattern = list(cluster)[-1]
    n_days = len(pattern)
    cluster.pop('pattern')
    
    d = np.zeros((n_days, 24))
    for (x, y), val in cluster.items():
        d[x - 1][y] = val
    
    kwargs = {
        'data' : d,
        'vmin' : 0,
        'vmax' : 1,
        'cmap' : 'Reds',
        'xticklabels' : range(24), 
        'yticklabels' : ['day_{}'.format(i + 1) for i in range(n_days)],
        'square' : True,
        'linecolor' : 'gray',
        'linewidths' : 0.05,
        'cbar' : False
    }
    
    ax = sns.heatmap(**kwargs)
    if legend:
        cbar = ax.figure.colorbar(ax.collections[0])
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["0%", "50%", "100%"])
        

@plot_size((15, 6))
def draw_profile(profile):
    """draws a heatmap representing the temporal profile of a single passenger."""
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    days = [x.capitalize() for x in days]
    d = np.zeros((7, 24))
    
    profile.pop('pattern')
    profile.pop('schedule')
    
    for i, arr in enumerate(np.split(profile.values, 14)):
        d[i % 7] = d[i % 7] + arr
    
    mask = np.fromfunction(np.vectorize(lambda i, j: d[i][j] == 0), (7, 24), dtype = int)
    sns.heatmap(
        d, 
        vmin = 0, 
        vmax = max(4, np.max(d)), 
        cmap = 'Reds',
        square = True, 
        xticklabels = range(24), 
        yticklabels = days, 
        cbar = False, 
        annot = True, 
        mask = mask, 
        linecolor = 'gray', 
        linewidth = 0.05
    )