import networkx as nx
import numpy as np
import string
from random import sample
import torch
from torchvision.ops import box_convert
import math


import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

IMG_LEN=924

# draw graph
DRAW_OPTIONS = {
    "font_size": 5,
    "font_family": 'Bradley Hand',
    "node_size": 125,
    # "arrowsize": 25,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
}


object_to_code = {}
for i in range(48, 58):
    object_to_code[chr(i)] = i - 48
for i in range(97, 123):
    object_to_code[chr(i)] = i - 87

all_labels = list(string.ascii_lowercase) + [chr(num) for num in range(48, 58)]

n = 3
p = 1

def gen_graph(n,p, id=0):
    # create graph
    G = nx.fast_gnp_random_graph(n, p, directed=False)


    # relabel nodes
    labels = dict(zip(nx.spring_layout(G).keys(), sample(all_labels, n)))
    backwards_labels = {labels[key]: key for key in labels}
    nx.relabel_nodes(G, labels, copy=False)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    adjacency = nx.adjacency_matrix(G).toarray()

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, **DRAW_OPTIONS)
    # set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.axis("off")
    plt.savefig(f'{id}.png', bbox_inches='tight', pad_inches=0)

    # Generate bounding box points for training data
    radius = [.11 for i in range(n)]
    def format_box(x1, y1, x2, y2):
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    letters_radius = .05

    boxes = []
    training_labels = []
    for k in pos.keys():
        i = backwards_labels[k]
        boxes.append(format_box(pos[k][0] - radius[i], pos[k][1] - radius[i], pos[k][0] + radius[i], pos[k][1] + radius[i]))
        training_labels.append(36)

    for k in pos.keys():
        boxes.append(format_box(pos[k][0] - letters_radius, pos[k][1] - letters_radius, pos[k][0] + letters_radius, pos[k][1] + letters_radius))
        if k in object_to_code:
            training_labels.append(object_to_code[k])

    for i in range(len(adjacency)):
        for j in range(0, i + 1):
            letter = labels[i]
            letter_2 = labels[j]
            if adjacency[i][j] == 1:
                x1 = pos[letter][0]
                y1 = pos[letter][1]
                x2 = pos[letter_2][0]
                y2 = pos[letter_2][1]

                second_above = y2 > y1
                second_right = x2 > x1
                slope = (y2 - y1) / (x2 - x1)

                x_delta_abs1 = math.sqrt(radius[i]**2 / (1 + slope**2))
                y_delta_abs1 = abs(slope * x_delta_abs1)

                x_delta_abs2 = math.sqrt(radius[j]**2 / (1 + slope**2))
                y_delta_abs2 = abs(slope * x_delta_abs2)

                box_x1 = x1 + x_delta_abs1 if second_right else x1 - x_delta_abs1
                box_y1 = y1 + y_delta_abs1 if second_above else y1 - y_delta_abs1

                box_x2 = x2 - x_delta_abs2 if second_right else x2 + x_delta_abs2
                box_y2 = y2 - y_delta_abs2 if second_above else y2 + y_delta_abs2

                boxes.append(format_box(box_x1, box_y1, box_x2, box_y2))
                
                if slope > 0:
                    training_labels.append(38)
                elif slope <= 0:
                    training_labels.append(39)

    out = {"labels": torch.Tensor([training_labels]), "boxes": torch.Tensor(np.array(boxes))}

    shape = [3,1,1]

    # Transform coordinates
    for i in range(len(out["boxes"])):
        out["boxes"][i][0] = shape[2] / 2 + out["boxes"][i][0] * shape[2] / 4
        out["boxes"][i][2] = shape[2] / 2 + out["boxes"][i][2] * shape[2] / 4
        temp = shape[1] / 2 - out["boxes"][i][1] * shape[1] / 4
        out["boxes"][i][1] = shape[1] / 2 - out["boxes"][i][3] * shape[1] / 4
        out["boxes"][i][3] = temp


    out['boxes'] = box_convert(out['boxes'], 'xyxy', 'cxcywh')

    rows = []
    print(out['boxes'].shape)
    print(out['labels'].shape)
    for l, b in zip(out['labels'][0], out['boxes']):
        # image_id,width,height,bbox,class
        rows.append([id, IMG_LEN, IMG_LEN, b, l])
    
    return rows


