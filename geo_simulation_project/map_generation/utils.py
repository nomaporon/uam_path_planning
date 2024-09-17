import random
from matplotlib import patches

def generate_random_colors(n):
    return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(n)]

def add_legend(ax, colors):
    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', alpha=0.5, label=f'Polygon {i+1}') 
                       for i, color in enumerate(colors)]
    ax.legend(handles=legend_elements, loc='upper right')