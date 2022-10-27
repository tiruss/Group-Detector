from collections import Counter

import numpy as np
import torch
import torchvision.ops.boxes as bops

import networkx
from networkx.algorithms.components.connected import connected_components


def detect_group(outputs, threshold, frame_idx):
    count_ids = Counter()
    ids = []
    ious = torch.triu(bops.box_iou(torch.tensor(outputs[:, 0:4]), torch.tensor(outputs[:, 0:4])),
                      diagonal=1)
    # ious = ious > 0.1
    iou_idx = torch.nonzero(ious).cpu()

    center_x, center_y = (outputs[:, 0] + outputs[:, 2]) / 2, (outputs[:, 1] + outputs[:, 3]) / 2
    centers = np.stack([center_x, center_y, outputs[:, 4]], axis=1)

    for v in iou_idx:
        # count += 1
        interact1, interact2 = centers[int(v[0])], centers[int(v[1])]
        idx = [int(interact1[2]), int(interact2[2])]
        ids.append(idx)
        # s += f" interact success: {count} "

    if frame_idx != 0 and frame_idx % 100 == 0:
        str_ids = list(map(str, ids))
        count_ids = Counter(str_ids)
        ids = []

    return count_ids, iou_idx, centers

def check_group(id, group_list):
    group_key = [k for k, v in group_list.items() if id in v]
    if group_key:
        return group_key
    else:
        return 0

def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current