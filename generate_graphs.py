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


object_to_code = {'node':36, 
                  'filled_node':37,
                  'undirected_BL_TR':38,
                  'undirected_BR_TL':39,
                  'directed_BL_TR':40,
                  'directed_TR_BL':41,
                  'directed_BR_TL':42,
                  'directed_TL_BR':43,}
for i in range(48, 58):
    object_to_code[chr(i)] = i - 48
for i in range(97, 123):
    object_to_code[chr(i)] = i - 87

all_labels = list(string.ascii_lowercase) + [chr(num) for num in range(48, 58)]



# n = number of vertices
# p = probability of an edge
# f = font size
# directed = directed/undirected graph
# show = whether to display graph
def gen_graph(n, p, f, directed = False, show = False, id=0):
    # create graph
    G = nx.fast_gnp_random_graph(n, p, directed=directed)

    # relabel nodes
    labels = dict(zip(nx.spring_layout(G).keys(), sample(all_labels, n)))
    backwards_labels = {labels[key]: key for key in labels}
    nx.relabel_nodes(G, labels, copy=False)
    
    # remove loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    
    # adjacency matrix
    adjacency = nx.adjacency_matrix(G).toarray()

    # coordinate dictionary
    pos = nx.spring_layout(G)
    
    # draw settings
    radii = list(np.arange(f + 5, f + 15))
    sample_radii = np.random.choice(radii, n)
    DRAW_OPTIONS = {
        "font_size": f,
        "font_family": 'Bradley Hand',
        "node_size": sample_radii**2,
        # "arrowsize": 25,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }
    
    # draw graph
    fig = nx.draw_networkx(G, pos, **DRAW_OPTIONS)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.axis("off")
    plt.savefig(f'{id}.png', bbox_inches='tight', pad_inches=0)
    if not show:
        plt.close(fig)

    # radii for bounding boxes
    radius = sample_radii/100
    letters_radius = f/100
    
    # Generate bounding box points for training data
    def format_box(x1, y1, x2, y2):
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


    boxes = []
    training_labels = []
    # node boxes
    for k in pos.keys():
        i = backwards_labels[k]
        boxes.append(format_box(pos[k][0] - radius[i], pos[k][1] - radius[i], pos[k][0] + radius[i], pos[k][1] + radius[i]))
        training_labels.append(object_to_code['node'])

    # label boxes
    for k in pos.keys():
        boxes.append(format_box(pos[k][0] - letters_radius, pos[k][1] - letters_radius, pos[k][0] + letters_radius, pos[k][1] + letters_radius))
        if k in object_to_code:
            training_labels.append(object_to_code[k])

    # edge boxes
    for i in range(len(adjacency)):
        end = i + 1
        if directed: 
            end = len(adjacency[i])
        for j in range(0, end):
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
                
                if not directed:
                    if slope > 0:
                        training_labels.append(object_to_code['undirected_BL_TR'])
                    elif slope <= 0:
                        training_labels.append(object_to_code['undirected_BR_TL'])
                else:
                    if second_above and second_right: 
                        training_labels.append(object_to_code['directed_BL_TR'])
                    elif second_above and not second_right:
                        training_labels.append(object_to_code['directed_BR_TL'])
                    elif not second_above and second_right:
                        training_labels.append(object_to_code['directed_TL_BR'])
                    elif not second_above and not second_right:
                        training_labels.append(object_to_code['directed_TR_BL'])

    out = {"labels": torch.Tensor([training_labels]), "boxes": torch.Tensor(np.array(boxes))}

#     if show:
#         out1 = out
#         for i in range(len(out1["boxes"])):
#             out1["boxes"][i][0] = t.shape[2] / 2 + out1["boxes"][i][0] * t.shape[2] / 4
#             out1["boxes"][i][2] = t.shape[2] / 2 + out1["boxes"][i][2] * t.shape[2] / 4
#             temp = t.shape[1] / 2 - out1["boxes"][i][1] * t.shape[1] / 4
#             out1["boxes"][i][1] = t.shape[1] / 2 - out1["boxes"][i][3] * t.shape[1] / 4
#             out1["boxes"][i][3] = temp
#         t_boxes = draw_bounding_boxes(t, out1["boxes"], colors = 'red', width=2)
#         img_boxes = T.ToPILImage()(t_boxes)
#         plt.imshow(img_boxes)
    
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
    for l, b in zip(out['labels'][0], out['boxes']):
        # image_id,width,height,bbox,class
        rows.append([id, IMG_LEN, IMG_LEN, b, l])
    
    return rows