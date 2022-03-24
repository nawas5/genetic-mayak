from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

vertex = ((0, 1), (1, 1), (0.5, 0.8), (0.1, 0.5), (0.8, 0.2), (0.4, 0))

vx = [v[0] for v in vertex]
vy = [v[1] for v in vertex]

def show_graph(ax, best):
    ax.add_line(Line2D((vertex[0][0], vertex[1][0]), (vertex[0][1], vertex[1][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[0][0], vertex[2][0]), (vertex[0][1], vertex[2][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[0][0], vertex[3][0]), (vertex[0][1], vertex[3][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[1][0], vertex[2][0]), (vertex[1][1], vertex[2][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[2][0], vertex[5][0]), (vertex[2][1], vertex[5][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[2][0], vertex[4][0]), (vertex[2][1], vertex[4][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[3][0], vertex[5][0]), (vertex[3][1], vertex[5][1]), color='#aaa'))
    ax.add_line(Line2D((vertex[4][0], vertex[5][0]), (vertex[4][1], vertex[5][1]), color='#aaa'))

    startV = 0
    for i, v in enumerate(best):
        if i == 0:
            continue

        prev = startV
        v = v[:v.index(i)+1]
        for j in v:
            ax.add_line(Line2D((vertex[prev][0], vertex[j][0]), (vertex[prev][1], vertex[j][1]), color='r'))
            prev = j

    ax.plot(vx, vy, ' ob', markersize=15)



v_ship = np.array([0, 1, 2, 3])
h_ship = np.array([0, 0, 0, 0])
type_ship=[1, 1, 1, 1]
colors=['y']

def show_ships(ax, best, pole_size):
    rect = Rectangle((0, 0), pole_size+1, pole_size+1, fill=None, edgecolor='r')

    t_n = 0
    for i in range(0, len(best), 3):
        x = best[i]
        y = best[i+1]
        r = best[i+2]
        t = type_ship[t_n]
        t_n += 1

        if r == 1:
            ax.plot(v_ship[:t] + x, h_ship[:t] + y, ' sb', markersize=18, alpha=0.8, markerfacecolor=colors[t-1])
        else:
            ax.plot(h_ship[:t] + x, v_ship[:t] + y, ' sb', markersize=18, alpha=0.8, markerfacecolor=colors[t-1])

    ax.add_patch(rect)
