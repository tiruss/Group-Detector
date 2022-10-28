from ast import literal_eval
from collections import Counter

import numpy as np
import torch
import torchvision.ops.boxes as bops

import networkx
from networkx.algorithms.components.connected import connected_components

import os
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import cv2
from glob import glob


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

def unify_group(groups, threshold=None):
    # unify groups
    group_list = []
    for k, v in groups.items():
        if v > threshold:
            group_list.append(k)
        else:
            return

    group_list = [literal_eval(i) for i in group_list]
    G = to_graph(group_list)
    group_list = list(connected_components(G))
    group_list = [list(i) for i in group_list]

    return group_list

def merge_unify_group(group_list):
    # unify groups
    groups = []
    G = to_graph(group_list)
    group_list = list(connected_components(G))
    group_list = [list(i) for i in group_list]

    return group_list

def check_group_key(id, group_list):
    group_key = [k for k, v in group_list.items() if id in v]
    if group_key:
        return group_key[0]
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

def count_all_group(folder):
    files = glob(folder + "/*.xml")
    print(len(files))
    g_idss = []
    for file in files:
        # print(file)
        # parser = etree.XMLParser(remove_blank_text=True)
        tree = ElementTree.parse(file)
        root = tree.getroot()
        g_ids = []
        for obj in root.findall('object'):
            attributes = obj.find('attributes')
            if attributes is not None:
                person = attributes.find('person')
                group_id = person.find('group_id').text
                # print(group_id)
                # print(group_id)
                if group_id is not None:
                    if len(group_id) == 1:
                        g_id = int(group_id)
                    elif len(group_id) == 2:
                        g_id = int(group_id[1])
                    # g_id = int(group_id[1])
                    g_ids.append(g_id)

        g_idss.append(len(set(g_ids))+1)
        # print(g_idss)
    print(sum(g_idss))


if __name__ == '__main__':

    folder = "/home/bronze9/Group-Detector/test_data/airport_test_entry1_mask_on-005"
    count_all_group(folder)
