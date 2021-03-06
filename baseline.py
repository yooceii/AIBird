from random import randint
from random import uniform
from math import sqrt, ceil
from pathlib import Path
from shutil import copyfile
import pickle
import matlab.engine
import copy
from sympy import *
import numpy as np
import pandas as pd
import os.path
import time
import logging
import sys
import os

# blocks number and size
blocks = {'1': [0.85, 0.85], '2': [0.85, 0.43], '3': [0.43, 0.85], '4': [0.43, 0.43],
          '5': [0.22, 0.22], '6': [0.43, 0.22], '7': [0.22, 0.43], '8': [0.85, 0.22],
          '9': [0.22, 0.85], '10': [1.68, 0.22], '11': [0.22, 1.68],
          '12': [2.06, 0.22], '13': [0.22, 2.06]}

# blocks number and name
# (blocks 3, 7, 9, 11 and 13) are their respective block names rotated 90 derees clockwise
block_names = {'1': "SquareHole", '2': "RectFat", '3': "RectFat", '4': "SquareSmall",
               '5': "SquareTiny", '6': "RectTiny", '7': "RectTiny", '8': "RectSmall",
               '9': "RectSmall", '10': "RectMedium", '11': "RectMedium",
               '12': "RectBig", '13': "RectBig"}

# additional objects number and name
additional_objects = {'1': "TriangleHole",
                      '2': "Triangle", '3': "Circle", '4': "CircleSmall"}

# additional objects number and size
additional_object_sizes = {'1': [0.82, 0.82], '2': [
    0.82, 0.82], '3': [0.8, 0.8], '4': [0.45, 0.45]}

# blocks number and probability of being selected
probability_table_blocks = {'1': 0.08, '2': 0.08, '3': 0.08, '4': 0.08,
                            '5': 0.08, '6': 0.08, '7': 0.08, '8': 0.08,
                            '9': 0.08, '10': 0.08, '11': 0.08,
                            '12': 0.08, '13': 0.04}
# probability_table_blocks = {'1':0, '2':0, '3':0, '4':0,
#                             '5':0, '6':0.25, '7':0, '8':0.25,
#                             '9':0, '10':0.25, '11':0,
#                             '12':0.25, '13':0}


# materials that are available
materials = ["wood", "stone", "ice"]

# bird types number and name
bird_names = {'1': "BirdRed", '2': "BirdBlue",
              '3': "BirdYellow", '4': "BirdBlack", '5': "BirdWhite"}

# bird types number and probability of being selected
bird_probabilities = {'1': 0.35, '2': 0.2, '3': 0.2, '4': 0.15, '5': 0.1}

TNT_block_probability = 0.3

pig_size = [0.5, 0.5]    # size of pigs

platform_size = [0.62, 0.62]     # size of platform sections

# buffer uesd to push edge blocks further into the structure center
# (increases stability)
edge_buffer = 0

absolute_ground = -3.5          # the position of ground within level

# maximum number of peaks a structure can have (up to 5)
max_peaks = 5
min_peak_split = 10     # minimum distance between two peak blocks of structure
max_peak_split = 50     # maximum distance between two peak blocks of structure

minimum_height_gap = 3.5        # y distance min between platforms
# x_distance min between platforms / y_distance min between platforms and
# ground structures
platform_distance_buffer = 0.4

# defines the levels area (ie. space within which structures/platforms can
# be placed)
level_width_min = -3.0
level_width_max = 9.0
# only used by platforms, ground structures use absolute_ground to
# determine their lowest point
level_height_min = -2.0
level_height_max = 15.0

# how precise to check for possible pig positions on ground
pig_precision = 0.01

# minimum amount of space allocated to ground structure
min_ground_width = 2.5
# desired height limit of ground structures
ground_structure_height_limit = (
    (level_height_max - minimum_height_gap) - absolute_ground)/1.5

# number of times to attempt to place a platform before abandoning it
max_attempts = 100

# step
gap = 0.45

number_pigs = 4

x = symbols("x")
y = symbols("y")


class Node(object):
    """docstring for Node"""

    def __init__(self, arg=None):
        super(Node, self).__init__()
        self.g = 0
        self.h = 0
        self.f = self.g + self.h
        self.parent = arg
        self.block = str(0)
        self.current_structure_height = 0
        self.position = 0  # left
        self.point = 0  # bottom
        self.max_height = 0
        self.is_start = 0  # test if the node is the structure's head
        self.is_head = 0  # test if the node is the row's head

    def print(self):
        print(*self.__dict__.items(), sep=' ')


# generates a list of all possible subsets for structure bottom

def generate_subsets(current_tree_bottom):
    current_distances = []
    subsets = []
    current_point = 0
    while current_point < len(current_tree_bottom)-1:
        current_distances.append(current_tree_bottom[
                                 current_point+1][1] - current_tree_bottom[current_point][1])

        current_point = current_point + 1

    # remove similar splits causesd by floating point imprecision
    for i in range(len(current_distances)):
        current_distances[i] = round(current_distances[i], 10)

    # all possible x-distances between bottom blocks
    split_points = list(set(current_distances))  # delete duplicate x-distances

    for i in split_points:      # subsets based on differences between x-distances
        current_subset = []
        start_point = 0
        end_point = 1
        for j in current_distances:
            if j >= i:
                current_subset.append(current_tree_bottom[
                                      start_point:end_point])
                start_point = end_point
            end_point = end_point + 1

        current_subset.append(current_tree_bottom[start_point:end_point])

        subsets.append(current_subset)

    subsets.append([current_tree_bottom])

    return subsets


# finds the center positions of the given subset

def find_subset_center(subset):
    if len(subset) % 2 == 1:
        return subset[(len(subset)-1)//2][1]
    else:
        return (subset[len(subset)//2][1] - subset[(len(subset)//2)-1][1])/2.0 + subset[(len(subset)//2)-1][1]


# finds the edge positions of the given subset

def find_subset_edges(subset):
    edge1 = subset[0][1] - (blocks[str(subset[0][0])][0])/2.0 + edge_buffer
    edge2 = subset[-1][1] + (blocks[str(subset[-1][0])][0])/2.0 - edge_buffer
    return[edge1, edge2]


# checks that positions for new block dont overlap and support the above blocks

def check_valid(grouping, choosen_item, current_tree_bottom, new_positions):

    # check no overlap
    i = 0
    while i < len(new_positions)-1:
        if (new_positions[i] + (blocks[str(choosen_item)][0])/2) > (new_positions[i+1] - (blocks[str(choosen_item)][0])/2):
            return False
        i = i + 1

    # check if each structural bottom block's edges supported by new blocks
    for item in current_tree_bottom:
        edge1 = item[1] - (blocks[str(item[0])][0])/2
        edge2 = item[1] + (blocks[str(item[0])][0])/2
        edge1_supported = False
        edge2_supported = False
        for new in new_positions:
            if ((new - (blocks[str(choosen_item)][0])/2) <= edge1 and (new + (blocks[str(choosen_item)][0])/2) >= edge1):
                edge1_supported = True
            if ((new - (blocks[str(choosen_item)][0])/2) <= edge2 and (new + (blocks[str(choosen_item)][0])/2) >= edge2):
                edge2_supported = True
        if edge1_supported == False or edge2_supported == False:
            return False
    return True


# check if new block can be placed under center of bottom row blocks validly

def check_center(grouping, choosen_item, current_tree_bottom):
    new_positions = []
    for subset in grouping:
        new_positions.append(find_subset_center(subset))
    return check_valid(grouping, choosen_item, current_tree_bottom, new_positions)


# check if new block can be placed under edges of bottom row blocks validly

def check_edge(grouping, choosen_item, current_tree_bottom):
    new_positions = []
    for subset in grouping:
        new_positions.append(find_subset_edges(subset)[0])
        new_positions.append(find_subset_edges(subset)[1])
    return check_valid(grouping, choosen_item, current_tree_bottom, new_positions)


# check if new block can be placed under both center and edges of bottom
# row blocks validly

def check_both(grouping, choosen_item, current_tree_bottom):
    new_positions = []
    for subset in grouping:
        new_positions.append(find_subset_edges(subset)[0])
        new_positions.append(find_subset_center(subset))
        new_positions.append(find_subset_edges(subset)[1])
    return check_valid(grouping, choosen_item, current_tree_bottom, new_positions)


# choose a random item/block from the blocks dictionary based on
# probability table

def choose_item(table):
    ran_num = uniform(0.0, 1.0)
    selected_num = 0
    while ran_num > 0:
        selected_num = selected_num + 1
        ran_num = ran_num - table[str(selected_num)]
    return selected_num


# finds the width of the given structure

def find_structure_width(structure):
    min_x = 999999.9
    max_x = -999999.9
    for block in structure:
        if round((block[1]-(blocks[str(block[0])][0]/2)), 10) < min_x:
            min_x = round((block[1]-(blocks[str(block[0])][0]/2)), 10)
        if round((block[1]+(blocks[str(block[0])][0]/2)), 10) > max_x:
            max_x = round((block[1]+(blocks[str(block[0])][0]/2)), 10)
    return (round(max_x - min_x, 10))


# finds the height of the given structure

def find_structure_height(structure):
    min_y = 999999.9
    max_y = -999999.9
    for block in structure:
        if round((block[2]-(blocks[str(block[0])][1]/2)), 10) < min_y:
            min_y = round((block[2]-(blocks[str(block[0])][1]/2)), 10)
        if round((block[2]+(blocks[str(block[0])][1]/2)), 10) > max_y:
            max_y = round((block[2]+(blocks[str(block[0])][1]/2)), 10)
    return (round(max_y - min_y, 10))


def generate(structure_height):
    structures = []
    start = Node(None)
    start.g = 0
    start.h = 0
    start.is_start = 1
    start.is_head = 1
    leaf_node = []
    temp_leaf = []
    temp_leaf.append(start)
    leaf_node.append(start)
    step = 1
    height = 0
    signal = True
    while signal:

        leaf_node = copy.copy(temp_leaf)
        temp_leaf.clear()

        print("\n\nstep:", step, "\n\n")
        for leaf in leaf_node:
            # return structure when cannot add more blocks
            print('\n', leaf.current_structure_height,
                  leaf.max_height, structure_height)
            if leaf.current_structure_height+leaf.max_height+0.22 > structure_height or (leaf.max_height == 0 and leaf.is_start != 1):
                # data_logger.info(leaf.current_structure_height+' '+leaf.max_height)
                print("\nstructure\n")
                structures.append(leaf)
                continue
            leaf.print()
            parents = []
            temp_parents = []
            # x1, x2 = round(x1, 2), round(x2, 2)
            x1, x2 = limit_boundary(
                leaf.current_structure_height+leaf.max_height)
            print(type(x1), x1, type(x2), x2)
            x1, x2 = round(x1, 2), round(x2, 2)
            sections = np.arange(x1, x2, gap)
            parents.append(leaf)
            # each position
            for position in sections:
                position = round(position, 2)
                print('\n', position)
                temp_parents.clear()
                # blocks in the same position
                print("parents", len(parents))
                for parent_node in parents:

                    #   parent_node.print()
                    childlist = generate_child(parent_node, step)
                    empty = True
                    if position != x1 and parent_node.is_head == 1:
                        print("ignore", position, x1, parent_node.is_head)
                        continue
                    # print("childlist", len(childlist))
                    for child in childlist:
                        # if parent_node.max_height == 0:
                        #     child.max_height = round(blocks[child.block][1],2)
                        # else:
                        #     child.max_height = round(parent_node.max_height,2)
                        if parent_node.is_head != 1 and parent_node.max_height != 0:
                            child.max_height = round(parent_node.max_height, 2)
                        else:
                            child.max_height = round(blocks[child.block][1], 2)

                        # initialize current_structure
                        if position == x1:
                            child.current_structure_height = round(parent_node.current_structure_height +
                                                                   parent_node.max_height, 2)
                        else:
                            child.current_structure_height = round(
                                parent_node.current_structure_height, 2)

                        child.position = round(position, 2)
                        # child.point = find_point(
                        #     position, parent_node, child.current_structure_height)
                        child.point = child.current_structure_height
                        print("--------------------test-----------------\n", child.point,
                              child.block, position, child.current_structure_height)
                        print(check_stability(child, parent_node), (position+blocks[child.block][0]) <= x2, (blocks[child.block][1] <= height_limit(
                            position, child.point)-child.point), position == x1, check_overlap(child, parent_node), '\n')
                        if child.point != parent_node.current_structure_height:
                            continue
                        if check_stability(child, parent_node) and (position+blocks[child.block][0]) <= x2 and (blocks[child.block][1] <= height_limit(position, child.point)-child.point) and (position == x1 or check_overlap(child, parent_node)):
                            child.parent = parent_node
                            if position+blocks[child.block][0] > x2-0.30:
                                child.is_head = 1
                                temp_leaf.append(child)
                            temp_parents.append(child)
                            empty = False
                            child.print()
                    # if no child is suitable
                    if empty:
                        child = Node(parent_node)
                        if position == x1:
                            child.current_structure_height = round(parent_node.current_structure_height +
                                                                   parent_node.max_height, 2)
                        else:
                            child.current_structure_height = round(
                                parent_node.current_structure_height, 2)
                        child.block = str(0)
                        if parent_node.is_head != 1 and parent_node.max_height != 0:
                            child.max_height = round(parent_node.max_height, 2)
                        else:
                            child.max_height = round(0, 2)
                        child.position = round(position, 2)
                        child.point = child.current_structure_height
                        if round(position, 2)+gap >= round(x2, 2):
                            child.is_head = 1
                            temp_leaf.append(child)
                        child.print()
                        temp_parents.append(child)
                    # child.print()

                        # if position > x2-0.22:
                        #     temp_leaf.append(child)

                parents.clear()
                parents = copy.copy(temp_parents)
        print("----------------------------------")
        print('temp_leaf: ', len(temp_leaf))
        if len(temp_leaf) > 15:
            # if step == 1:
            temp_leaf = prune(temp_leaf, step)
        if len(temp_leaf) == 0:
            signal = False
            break
        print('temp_leaf: ', len(temp_leaf))
        leaf_node.clear()
        step += 1
    print("finished", len(structures))
    structures = prune(structures, step)
    return structures


def construct(nodes, folder):
    os.makedirs(folder, exist_ok=True)
    structures = []
    i = 4
    for node in nodes:
        with open(folder+'/node'+str(i), 'wb') as filehandler:
            pickle.dump(node, filehandler)
        complete_locations = []
        while node.is_start != 1:
            if node.block != str(0):
                # node.print()
                complete_locations.append(
                    [int(node.block), round(level_width_min+node.position+round(blocks[node.block][0]/2.0, 3), 3), round(absolute_ground+node.point+round(blocks[node.block][1]/2.0, 3), 3)])
            node = node.parent
        complete_locations, final_pig_positions = find_pig_position(
            list(reversed(complete_locations)))
        final_pig_positions, removed_pigs = remove_unnecessary_pigs(
            number_pigs, final_pig_positions)
        final_TNT_positions = add_TNT(removed_pigs)
        write_level_xml(folder, complete_locations,
                        [], final_pig_positions, final_TNT_positions, [], 5, i, [])
        i = i+1
        structures.append(complete_locations.reverse())
        print('\n')


def prune(leaves, step):
    print('\n pruning \n')
    file = Path("export.csv")
    if file.is_file():
        os.remove("export.csv")
    columns = []
    for leaf in leaves:
        column = []
        temp_leaf = copy.copy(leaf)
        # temp_leaf.print()
        column.insert(0, temp_leaf)
        temp_leaf = temp_leaf.parent
        while temp_leaf.is_head != 1:  # might abort at begining
            column.insert(0, temp_leaf)
            # if temp_leaf.is_start == 1:
            #     print("head")
            # temp_leaf.print()
            temp_leaf = temp_leaf.parent
        columns.append(column)
        # print(list([x.block, x.position] for x in column))
    # sort columns according to columns' first node
    # columns_tree = []
    # for x in columns:
    #     if x[0] not in columns_tree:
    #         columns_tree.append([x[0]])
    #     else:
    #         ([y for y in columns_tree if x[0] in y][0]).append(x)

    # for x in range(0, len(columns_tree)):
    #     cosine_simility(columns_tree[x])
    for x in columns:
        start, end = limit_boundary(
            x[0].current_structure_height)
        vectorization(x, round(start, 2), round(end, 2))
    if step == 1:
        copyfile('export.csv', 'export_step_1.csv')
    eng = matlab.engine.start_matlab()
    eng.calculate_k(nargout=0)
    K = int(eng.workspace['I'])
    closestIdx, Idx, centroid = eng.Structure_prune(K, nargout=3)
    # parent_nodes=rebuild_node([list(map(lambda x: round(x, 2), i)) for i in prune_result])
    parent_nodes = []
    for i in closestIdx[0]:
        parent_nodes.append(leaves[i-1])
    eng.quit()
    return parent_nodes


def cosine_simility(columns):
    start, end = limit_boundary(
        columns[0].current_structure_height)
    # print(len(columns))
    for index, val in enumerate(columns[1:]):
        columns[1+index] = vectorization(val, start, end)
        unit_vector = np.zeros(len(columns[1+index]))
        unit_vector[0] = 1
        # calculate cosine value between vector and unit vector
        columns[1+index] = np.dot(columns[1+index],
                                  unit_vector)/np.linalg.norm(columns[1+index])


def vectorization(column, start, end):
    column_vector = np.zeros((len(np.arange(start, end, gap)), 2))
    # print(start, end, np.shape(column_vector),
    #       column_vector, np.arange(start, end, 0.22))
    for block in column:
        # print(block.position, blocks[block.block][0], blocks[block.block][1])
        if block.block != str(0):
            width = blocks[block.block][0]
            height = blocks[block.block][1]
            position = int((block.position-start)/gap)
            # print("vectorization", position, width, height, start, end)
            # print(position)
            # print(column_vector)
            for x in np.arange(0, width, gap):
                if x+gap <= width:
                    column_vector[int(position+x/gap)][0] = round(gap, 2)
                elif x+gap > width:
                    column_vector[int(position+x/gap)][0] = round(width-x, 2)
                column_vector[int(position+x/gap)][1] = round(height, 2)
    column_vector_flatten = column_vector.flatten()
    df = pd.DataFrame([column_vector_flatten])
    # print("vectorization")
    if os.path.isfile("export.csv"):
        with open("export.csv", 'a') as f:
            df.to_csv(f, header=False)
    else:
        df.to_csv("export.csv")

    # return column_vector_flatten


def find_least_f(openlist):
    min_num = 0
    min_f = 0
    for i, val in enurmerate(openlist):
        if val.f < min_f:
            min_num = i
            min_f = val.f
    return min_num


def check_block_type(node):
    nd = copy.deepcopy(node)
    type_list = []
    while nd is None or nd.block is not str(0):
        if nd.block not in type_list:
            type_list.append(nd.block)
        nd = nd.parent
    return len(type_list)

# check if block overlap with other blocks


def check_overlap(node, parent):
    nd = copy.copy(parent)
    if nd.is_head == 1:
        return True
    while nd.is_head != 1:
        if nd.block != str(0):
            if nd.position+blocks[nd.block][0] > node.position:
                return False
            else:
                return True
        nd = nd.parent
    return True


def check_stability(node, parent):
    nd = copy.copy(parent)
    start = round(node.position, 2)
    end = round(node.position+blocks[node.block][0], 2)
    shadow_blocks = []
    contiguous_blocks = []
    if nd.is_start == 1 or node.current_structure_height == 0:
        return True

    signal = 0
    while nd.is_start != 1:
        if nd.is_head == 1:
            signal = signal+1
        if signal == 2:
            break
        if nd.block == str(0):
            nd = nd.parent
            continue
        elif nd.block != str(0) and ((round(nd.position+blocks[nd.block][0], 2) > start and round(nd.position+blocks[nd.block][0], 2) < end) or (round(nd.position, 2) > start and round(nd.position, 2) < end) or (round(nd.position, 2) <= start and round(nd.position+blocks[nd.block][0], 2) >= end)):
            shadow_blocks.append(nd)
            # print((nd.position+blocks[nd.block][0] >
            #        start and nd.position+blocks[nd.block][0] < end))
            # print((nd.position > start and nd.position < end))
            # print((nd.position <= start and nd.position +
            #        blocks[nd.block][0] >= end))
            print("shadow_blocks", nd.block, nd.position, nd.point)
            if (round(nd.position+blocks[nd.block][0], 2) > start and round(nd.position+blocks[nd.block][0], 2) < end and round(nd.position, 2) <= start) or (round(nd.position, 2) <= start and round(nd.position+blocks[nd.block][0], 2) >= end):
                break
        nd = nd.parent

    # shadow_blocks.sort(key=lambda x: x.point +
    #                    blocks[x.block][1], reverse=False)
    shadow_blocks.reverse()

    # max_point = max(shadow_blocks, key=lambda x: x.point+blocks[x.block][1])

    # for block in shadow_blocks:
    #     if block.point+blocks[block.block][1] == max_point:
    #         contiguous_blocks.append(block)

    # contiguous_blocks.sort(key=lambda x: x.position, reverse=False)
    contiguous_blocks = shadow_blocks
    # print("contiguous_blocks", len(contiguous_blocks))
    if len(contiguous_blocks) == 1:
        # if blocks[contiguous_blocks[0].block][0] >= blocks[node.block][0]:
        #     return True
        contiguous_block_left_point = round(contiguous_blocks[0].position, 2)
        contiguous_block_right_point = round(
            contiguous_block_left_point+blocks[contiguous_blocks[0].block][0], 2)
        node_left_point = round(node.position, 2)
        node_right_point = round(node_left_point+blocks[node.block][0], 2)
        if (contiguous_block_left_point <= node_left_point and contiguous_block_right_point >= node_right_point) or round(node_left_point+(blocks[node.block][0])/2.0, 2) == round(contiguous_block_left_point+(blocks[contiguous_blocks[0].block][0])/2.0, 2):
            print(1)
            return True
        else:
            return False
    elif len(contiguous_blocks) >= 2:
        if round(contiguous_blocks[0].position+blocks[contiguous_blocks[0].block][0], 2) > start and contiguous_blocks[-1].position + blocks[contiguous_blocks[-1].block][0]/2.0 >= (3*end+start)/4:
            return True
        else:
            return False
    return False


def find_point(position, node, current_structure_height):
    nd = copy.copy(node)
    while nd.is_start != 1:
        # nd.print()
        if nd.block == "0":
            nd = nd.parent
            continue
        if nd.point == current_structure_height:
            nd = nd.parent
            continue
        if nd.position < position and nd.position+blocks[nd.block][0] > position:
            return round(nd.point+blocks[nd.block][1], 2)
        nd = nd.parent
    return 0


def find_height(node):
    current_structure_height = node.current_structure_height
    nd = copy.deepcopy(node)
    start = nd.position
    end = nd.position+blocks[nd.block][0]
    overlap_blocks = []
    contiguous_blocks = []
    nd = nd.parent
    while nd.is_start == 0:
        if nd.block == "0":
            nd = nd.parent
            continue
        if nd.point == current_structure_height:
            nd = nd.parent
            continue
        if (nd.position+blocks[nd.block][0] > start and nd.position+blocks[nd.block][0] < end) or (nd.position > start and nd.position < end):
            overlap_blocks.append(nd)
        nd = nd.parent

    overlap_blocks.sort(key=lambda x: x.point, reverse=False)

    point = 0

    for block in overlap_blocks:
        if block.point >= point:
            contiguous_blocks.append(block)
        point = block.point

    contiguous_blocks.sort(key=lambda x: x.position, reverse=False)
    return contiguous_blocks[0].point


def find_block(height):
    candidates = []
    for index, size in blocks.items():
        if height == size[1]:
            candidates.append(blocks.get(index))
    return candidates


def generate_child(parent_node, step):
    childlist = []
    # print("2", parent_node.max_height)
    for key, val in blocks.items():
        # step is odd number, vertical alignment
        # if step % 2 == 1:
        #     if val[0] > val[1]:
        #         continue
        #     if parent_node.is_head == 0 and parent_node.max_height!=0 and round(val[1],2) != round(parent_node.max_height,2):
        #         continue
        # # step is even number, horizontal alignment
        # else:
        #     if val[0] < val[1]:
        #         continue
        #     if parent_node.is_head == 0 and parent_node.max_height!=0 and round(val[1],2) != round(parent_node.max_height,2):
        #         continue
        # # print(val)
        if parent_node.is_head == 0 and parent_node.max_height != 0 and round(val[1], 2) != round(parent_node.max_height, 2):
            continue
        child = Node()
        child.block = str(key)
        # child.g = 1-check_block_type(child)/13.0+child.current_volume / \
        # structure_volume
        child.g = 0
        child.f = child.g
        # child.h = max_width*parent_node.max_height-child.g
        # if parent_node.max_height == 0:
        #     child.max_height = blocks[i][0]
        # child.print()
        childlist.append(child)
    return childlist


def find_pig_position(complete_locations):
    # identify all possible pig positions on top of blocks (maximum 2 pigs per
    # block, checks center before sides)
    possible_pig_positions = []
    for block in complete_locations:
        block_width = round(blocks[str(block[0])][0], 10)
        block_height = round(blocks[str(block[0])][1], 10)
        pig_width = pig_size[0]
        pig_height = pig_size[1]

        # dont place block on edge if block too thin
        if blocks[str(block[0])][0] < pig_width:
            test_positions = [[round(block[1], 10), round(
                block[2] + (pig_height/2) + (block_height/2), 10)]]
        else:
            test_positions = [[round(block[1], 10), round(block[2] + (pig_height/2) + (block_height/2), 10)],
                              [round(block[1] + (block_width/3), 10),
                               round(block[2] + (pig_height/2) + (block_height/2), 10)],
                              [round(block[1] - (block_width/3), 10), round(block[2] + (pig_height/2) + (block_height/2), 10)]]  # check above centre of block
        for test_position in test_positions:
            valid_pig = True
            for i in complete_locations:
                if (round((test_position[0] - pig_width/2), 10) < round((i[1] + (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[0] + pig_width/2), 10) > round((i[1] - (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[1] + pig_height/2), 10) > round((i[2] - (blocks[str(i[0])][1])/2), 10) and
                        round((test_position[1] - pig_height/2), 10) < round((i[2] + (blocks[str(i[0])][1])/2), 10)):
                    valid_pig = False
            if valid_pig == True:
                possible_pig_positions.append(test_position)

    # identify all possible pig positions on ground within structure
    print('\ncomplete_locations\n', complete_locations)
    left_bottom = [complete_locations[0][0],
                   complete_locations[0][1]]  # total_tree[-1][0]
    print('right_bottom', list(
        filter(lambda x: x[2] == complete_locations[0][2], complete_locations)))
    right_bottom_block = sorted(list(filter(
        lambda x: x[2] == complete_locations[0][2], complete_locations)), key=lambda y: y[1])[-1]
    right_bottom = [right_bottom_block[0], right_bottom_block[1]]
    test_positions = []
    x_pos = left_bottom[1]

    while x_pos < right_bottom[1]:
        test_positions.append([round(x_pos, 10), round(
            absolute_ground + (pig_height/2), 10)])
        x_pos = x_pos + pig_precision

    for test_position in test_positions:
        valid_pig = True
        for i in complete_locations:
            if (round((test_position[0] - pig_width/2), 10) < round((i[1] + (blocks[str(i[0])][0])/2), 10) and
                    round((test_position[0] + pig_width/2), 10) > round((i[1] - (blocks[str(i[0])][0])/2), 10) and
                    round((test_position[1] + pig_height/2), 10) > round((i[2] - (blocks[str(i[0])][1])/2), 10) and
                    round((test_position[1] - pig_height/2), 10) < round((i[2] + (blocks[str(i[0])][1])/2), 10)):
                valid_pig = False
        if valid_pig == True:
            possible_pig_positions.append(test_position)

    # randomly choose a pig position and remove those that overlap it, repeat
    # until no more valid positions
    final_pig_positions = []
    while len(possible_pig_positions) > 0:
        pig_choice = possible_pig_positions.pop(
            randint(1, len(possible_pig_positions))-1)
        final_pig_positions.append(pig_choice)
        new_pig_positions = []
        for i in possible_pig_positions:
            if (round((pig_choice[0] - pig_width/2), 10) >= round((i[0] + pig_width/2), 10) or
                    round((pig_choice[0] + pig_width/2), 10) <= round((i[0] - pig_width/2), 10) or
                    round((pig_choice[1] + pig_height/2), 10) <= round((i[1] - pig_height/2), 10) or
                    round((pig_choice[1] - pig_height/2), 10) >= round((i[1] + pig_height/2), 10)):
                new_pig_positions.append(i)
        possible_pig_positions = new_pig_positions

    # number of pigs present in the structure
    print("Pig number:", len(final_pig_positions))
    print("")
    return complete_locations, final_pig_positions

# remove random pigs until number equals the desired amount


def remove_unnecessary_pigs(number_pigs, final_pig_positions):
    removed_pigs = []
    while len(final_pig_positions) > number_pigs:
        remove_pos = randint(0, len(final_pig_positions)-1)
        removed_pigs.append(final_pig_positions[remove_pos])
        final_pig_positions.pop(remove_pos)
    return final_pig_positions, removed_pigs


# add pigs on the ground until number equals the desired amount

def add_necessary_pigs(number_pigs):
    while len(final_pig_positions) < number_pigs:
        test_position = [
            uniform(level_width_min, level_width_max), absolute_ground]
        pig_width = pig_size[0]
        pig_height = pig_size[1]
        valid_pig = True
        for i in complete_locations:
            if (round((test_position[0] - pig_width/2), 10) < round((i[1] + (blocks[str(i[0])][0])/2), 10) and
                    round((test_position[0] + pig_width/2), 10) > round((i[1] - (blocks[str(i[0])][0])/2), 10) and
                    round((test_position[1] + pig_height/2), 10) > round((i[2] - (blocks[str(i[0])][1])/2), 10) and
                    round((test_position[1] - pig_height/2), 10) < round((i[2] + (blocks[str(i[0])][1])/2), 10)):
                valid_pig = False
        for i in final_pig_positions:
            if (round((test_position[0] - pig_width/2), 10) < round((i[0] + (pig_width/2)), 10) and
                    round((test_position[0] + pig_width/2), 10) > round((i[0] - (pig_width/2)), 10) and
                    round((test_position[1] + pig_height/2), 10) > round((i[1] - (pig_height/2)), 10) and
                    round((test_position[1] - pig_height/2), 10) < round((i[1] + (pig_height/2)), 10)):
                valid_pig = False
        if valid_pig == True:
            final_pig_positions.append(test_position)
    return final_pig_positions


# choose the number of birds based on the number of pigs and structures
# present within level

def choose_number_birds(final_pig_positions, number_ground_structures, number_platforms):
    number_birds = int(ceil(len(final_pig_positions)/2))
    if (number_ground_structures + number_platforms) >= number_birds:
        number_birds = number_birds + 1
    number_birds = number_birds + 1         # adjust based on desired difficulty
    return number_birds


# identify all possible triangleHole positions on top of blocks

def find_trihole_positions(complete_locations):
    possible_trihole_positions = []
    for block in complete_locations:
        block_width = round(blocks[str(block[0])][0], 10)
        block_height = round(blocks[str(block[0])][1], 10)
        trihole_width = additional_object_sizes['1'][0]
        trihole_height = additional_object_sizes['1'][1]

        # don't place block on edge if block too thin
        if blocks[str(block[0])][0] < trihole_width:
            test_positions = [[round(block[1], 10), round(
                block[2] + (trihole_height/2) + (block_height/2), 10)]]
        else:
            test_positions = [[round(block[1], 10), round(block[2] + (trihole_height/2) + (block_height/2), 10)],
                              [round(block[1] + (block_width/3), 10), round(block[2] +
                                                                            (trihole_height/2) + (block_height/2), 10)],
                              [round(block[1] - (block_width/3), 10), round(block[2] + (trihole_height/2) + (block_height/2), 10)]]

        for test_position in test_positions:
            valid_position = True
            for i in complete_locations:
                if (round((test_position[0] - trihole_width/2), 10) < round((i[1] + (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[0] + trihole_width/2), 10) > round((i[1] - (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[1] + trihole_height/2), 10) > round((i[2] - (blocks[str(i[0])][1])/2), 10) and
                        round((test_position[1] - trihole_height/2), 10) < round((i[2] + (blocks[str(i[0])][1])/2), 10)):
                    valid_position = False
            for j in final_pig_positions:
                if (round((test_position[0] - trihole_width/2), 10) < round((j[0] + (pig_size[0]/2)), 10) and
                        round((test_position[0] + trihole_width/2), 10) > round((j[0] - (pig_size[0]/2)), 10) and
                        round((test_position[1] + trihole_height/2), 10) > round((j[1] - (pig_size[1]/2)), 10) and
                        round((test_position[1] - trihole_height/2), 10) < round((j[1] + (pig_size[1]/2)), 10)):
                    valid_position = False
            for j in final_TNT_positions:
                if (round((test_position[0] - trihole_width/2), 10) < round((j[0] + (pig_size[0]/2)), 10) and
                        round((test_position[0] + trihole_width/2), 10) > round((j[0] - (pig_size[0]/2)), 10) and
                        round((test_position[1] + trihole_height/2), 10) > round((j[1] - (pig_size[1]/2)), 10) and
                        round((test_position[1] - trihole_height/2), 10) < round((j[1] + (pig_size[1]/2)), 10)):
                    valid_position = False
            for i in final_platforms:
                for j in i:
                    if (round((test_position[0] - trihole_width/2), 10) < round((j[0] + (platform_size[0]/2)), 10) and
                            round((test_position[0] + trihole_width/2), 10) > round((j[0] - (platform_size[0]/2)), 10) and
                            round((test_position[1] + platform_distance_buffer + trihole_height/2), 10) > round((j[1] - (platform_size[1]/2)), 10) and
                            round((test_position[1] - platform_distance_buffer - trihole_height/2), 10) < round((j[1] + (platform_size[1]/2)), 10)):
                        valid_position = False
            if valid_position == True:
                possible_trihole_positions.append(test_position)

    return possible_trihole_positions


# identify all possible triangle positions on top of blocks

def find_tri_positions(complete_locations):
    possible_tri_positions = []
    for block in complete_locations:
        block_width = round(blocks[str(block[0])][0], 10)
        block_height = round(blocks[str(block[0])][1], 10)
        tri_width = additional_object_sizes['2'][0]
        tri_height = additional_object_sizes['2'][1]

        # don't place block on edge if block too thin
        if blocks[str(block[0])][0] < tri_width:
            test_positions = [[round(block[1], 10), round(
                block[2] + (tri_height/2) + (block_height/2), 10)]]
        else:
            test_positions = [[round(block[1], 10), round(block[2] + (tri_height/2) + (block_height/2), 10)],
                              [round(block[1] + (block_width/3), 10),
                               round(block[2] + (tri_height/2) + (block_height/2), 10)],
                              [round(block[1] - (block_width/3), 10), round(block[2] + (tri_height/2) + (block_height/2), 10)]]

        for test_position in test_positions:
            valid_position = True
            for i in complete_locations:
                if (round((test_position[0] - tri_width/2), 10) < round((i[1] + (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[0] + tri_width/2), 10) > round((i[1] - (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[1] + tri_height/2), 10) > round((i[2] - (blocks[str(i[0])][1])/2), 10) and
                        round((test_position[1] - tri_height/2), 10) < round((i[2] + (blocks[str(i[0])][1])/2), 10)):
                    valid_position = False
            for j in final_pig_positions:
                if (round((test_position[0] - tri_width/2), 10) < round((j[0] + (pig_size[0]/2)), 10) and
                        round((test_position[0] + tri_width/2), 10) > round((j[0] - (pig_size[0]/2)), 10) and
                        round((test_position[1] + tri_height/2), 10) > round((j[1] - (pig_size[1]/2)), 10) and
                        round((test_position[1] - tri_height/2), 10) < round((j[1] + (pig_size[1]/2)), 10)):
                    valid_position = False
            for j in final_TNT_positions:
                if (round((test_position[0] - tri_width/2), 10) < round((j[0] + (pig_size[0]/2)), 10) and
                        round((test_position[0] + tri_width/2), 10) > round((j[0] - (pig_size[0]/2)), 10) and
                        round((test_position[1] + tri_height/2), 10) > round((j[1] - (pig_size[1]/2)), 10) and
                        round((test_position[1] - tri_height/2), 10) < round((j[1] + (pig_size[1]/2)), 10)):
                    valid_position = False
            for i in final_platforms:
                for j in i:
                    if (round((test_position[0] - tri_width/2), 10) < round((j[0] + (platform_size[0]/2)), 10) and
                            round((test_position[0] + tri_width/2), 10) > round((j[0] - (platform_size[0]/2)), 10) and
                            round((test_position[1] + platform_distance_buffer + tri_height/2), 10) > round((j[1] - (platform_size[1]/2)), 10) and
                            round((test_position[1] - platform_distance_buffer - tri_height/2), 10) < round((j[1] + (platform_size[1]/2)), 10)):
                        valid_position = False

            # as block not symmetrical need to check for support
            if blocks[str(block[0])][0] < tri_width:
                valid_position = False
            if valid_position == True:
                possible_tri_positions.append(test_position)

    return possible_tri_positions


# identify all possible circle positions on top of blocks (can only be
# placed in middle of block)

def find_cir_positions(complete_locations):
    possible_cir_positions = []
    for block in complete_locations:
        block_width = round(blocks[str(block[0])][0], 10)
        block_height = round(blocks[str(block[0])][1], 10)
        cir_width = additional_object_sizes['3'][0]
        cir_height = additional_object_sizes['3'][1]

        # only checks above block's center
        test_positions = [[round(block[1], 10), round(
            block[2] + (cir_height/2) + (block_height/2), 10)]]

        for test_position in test_positions:
            valid_position = True
            for i in complete_locations:
                if (round((test_position[0] - cir_width/2), 10) < round((i[1] + (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[0] + cir_width/2), 10) > round((i[1] - (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[1] + cir_height/2), 10) > round((i[2] - (blocks[str(i[0])][1])/2), 10) and
                        round((test_position[1] - cir_height/2), 10) < round((i[2] + (blocks[str(i[0])][1])/2), 10)):
                    valid_position = False
            for j in final_pig_positions:
                if (round((test_position[0] - cir_width/2), 10) < round((j[0] + (pig_size[0]/2)), 10) and
                        round((test_position[0] + cir_width/2), 10) > round((j[0] - (pig_size[0]/2)), 10) and
                        round((test_position[1] + cir_height/2), 10) > round((j[1] - (pig_size[1]/2)), 10) and
                        round((test_position[1] - cir_height/2), 10) < round((j[1] + (pig_size[1]/2)), 10)):
                    valid_position = False
            for j in final_TNT_positions:
                if (round((test_position[0] - cir_width/2), 10) < round((j[0] + (pig_size[0]/2)), 10) and
                        round((test_position[0] + cir_width/2), 10) > round((j[0] - (pig_size[0]/2)), 10) and
                        round((test_position[1] + cir_height/2), 10) > round((j[1] - (pig_size[1]/2)), 10) and
                        round((test_position[1] - cir_height/2), 10) < round((j[1] + (pig_size[1]/2)), 10)):
                    valid_position = False
            for i in final_platforms:
                for j in i:
                    if (round((test_position[0] - cir_width/2), 10) < round((j[0] + (platform_size[0]/2)), 10) and
                            round((test_position[0] + cir_width/2), 10) > round((j[0] - (platform_size[0]/2)), 10) and
                            round((test_position[1] + platform_distance_buffer + cir_height/2), 10) > round((j[1] - (platform_size[1]/2)), 10) and
                            round((test_position[1] - platform_distance_buffer - cir_height/2), 10) < round((j[1] + (platform_size[1]/2)), 10)):
                        valid_position = False
            if valid_position == True:
                possible_cir_positions.append(test_position)

    return possible_cir_positions


# identify all possible circleSmall positions on top of blocks

def find_cirsmall_positions(complete_locations):
    possible_cirsmall_positions = []
    for block in complete_locations:
        block_width = round(blocks[str(block[0])][0], 10)
        block_height = round(blocks[str(block[0])][1], 10)
        cirsmall_width = additional_object_sizes['4'][0]
        cirsmall_height = additional_object_sizes['4'][1]

        # don't place block on edge if block too thin
        if blocks[str(block[0])][0] < cirsmall_width:
            test_positions = [[round(block[1], 10), round(
                block[2] + (cirsmall_height/2) + (block_height/2), 10)]]
        else:
            test_positions = [[round(block[1], 10), round(block[2] + (cirsmall_height/2) + (block_height/2), 10)],
                              [round(block[1] + (block_width/3), 10), round(block[2] +
                                                                            (cirsmall_height/2) + (block_height/2), 10)],
                              [round(block[1] - (block_width/3), 10), round(block[2] + (cirsmall_height/2) + (block_height/2), 10)]]

        for test_position in test_positions:
            valid_position = True
            for i in complete_locations:
                if (round((test_position[0] - cirsmall_width/2), 10) < round((i[1] + (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[0] + cirsmall_width/2), 10) > round((i[1] - (blocks[str(i[0])][0])/2), 10) and
                        round((test_position[1] + cirsmall_height/2), 10) > round((i[2] - (blocks[str(i[0])][1])/2), 10) and
                        round((test_position[1] - cirsmall_height/2), 10) < round((i[2] + (blocks[str(i[0])][1])/2), 10)):
                    valid_position = False
            for j in final_pig_positions:
                if (round((test_position[0] - cirsmall_width/2), 10) < round((j[0] + (pig_size[0]/2)), 10) and
                        round((test_position[0] + cirsmall_width/2), 10) > round((j[0] - (pig_size[0]/2)), 10) and
                        round((test_position[1] + cirsmall_height/2), 10) > round((j[1] - (pig_size[1]/2)), 10) and
                        round((test_position[1] - cirsmall_height/2), 10) < round((j[1] + (pig_size[1]/2)), 10)):
                    valid_position = False
            for j in final_TNT_positions:
                if (round((test_position[0] - cirsmall_width/2), 10) < round((j[0] + (pig_size[0]/2)), 10) and
                        round((test_position[0] + cirsmall_width/2), 10) > round((j[0] - (pig_size[0]/2)), 10) and
                        round((test_position[1] + cirsmall_height/2), 10) > round((j[1] - (pig_size[1]/2)), 10) and
                        round((test_position[1] - cirsmall_height/2), 10) < round((j[1] + (pig_size[1]/2)), 10)):
                    valid_position = False
            for i in final_platforms:
                for j in i:
                    if (round((test_position[0] - cirsmall_width/2), 10) < round((j[0] + (platform_size[0]/2)), 10) and
                            round((test_position[0] + cirsmall_width/2), 10) > round((j[0] - (platform_size[0]/2)), 10) and
                            round((test_position[1] + platform_distance_buffer + cirsmall_height/2), 10) > round((j[1] - (platform_size[1]/2)), 10) and
                            round((test_position[1] - platform_distance_buffer - cirsmall_height/2), 10) < round((j[1] + (platform_size[1]/2)), 10)):
                        valid_position = False
            if valid_position == True:
                possible_cirsmall_positions.append(test_position)

    return possible_cirsmall_positions


# finds possible positions for valid additional block types

def find_additional_block_positions(complete_locations):
    possible_trihole_positions = []
    possible_tri_positions = []
    possible_cir_positions = []
    possible_cirsmall_positions = []
    if trihole_allowed == True:
        possible_trihole_positions = find_trihole_positions(complete_locations)
    if tri_allowed == True:
        possible_tri_positions = find_tri_positions(complete_locations)
    if cir_allowed == True:
        possible_cir_positions = find_cir_positions(complete_locations)
    if cirsmall_allowed == True:
        possible_cirsmall_positions = find_cirsmall_positions(
            complete_locations)
    return possible_trihole_positions, possible_tri_positions, possible_cir_positions, possible_cirsmall_positions


# combine all possible additonal block positions into one set

def add_additional_blocks(possible_trihole_positions, possible_tri_positions, possible_cir_positions, possible_cirsmall_positions):
    all_other = []
    for i in possible_trihole_positions:
        all_other.append(['1', i[0], i[1]])
    for i in possible_tri_positions:
        all_other.append(['2', i[0], i[1]])
    for i in possible_cir_positions:
        all_other.append(['3', i[0], i[1]])
    for i in possible_cirsmall_positions:
        all_other.append(['4', i[0], i[1]])

    # randomly choose an additional block position and remove those that overlap it
    # repeat untill no more valid position

    selected_other = []
    while (len(all_other) > 0):
        chosen = all_other.pop(randint(0, len(all_other)-1))
        selected_other.append(chosen)
        new_all_other = []
        for i in all_other:
            if (round((chosen[1] - (additional_object_sizes[chosen[0]][0]/2)), 10) >= round((i[1] + (additional_object_sizes[i[0]][0]/2)), 10) or
                    round((chosen[1] + (additional_object_sizes[chosen[0]][0]/2)), 10) <= round((i[1] - (additional_object_sizes[i[0]][0]/2)), 10) or
                    round((chosen[2] + (additional_object_sizes[chosen[0]][1]/2)), 10) <= round((i[2] - (additional_object_sizes[i[0]][1]/2)), 10) or
                    round((chosen[2] - (additional_object_sizes[chosen[0]][1]/2)), 10) >= round((i[2] + (additional_object_sizes[i[0]][1]/2)), 10)):
                new_all_other.append(i)
        all_other = new_all_other

    return selected_other


# remove restricted block types from the available selection

def remove_blocks(restricted_blocks):
    total_prob_removed = 0.0
    new_prob_table = copy.deepcopy(probability_table_blocks)
    for block_name in restricted_blocks:
        for key, value in block_names.items():
            if value == block_name:
                total_prob_removed = total_prob_removed + \
                    probability_table_blocks[key]
                new_prob_table[key] = 0.0
    new_total = 1.0 - total_prob_removed
    for key, value in new_prob_table.items():
        new_prob_table[key] = value/new_total
    return new_prob_table


# add TNT blocks based on removed pig positions

def add_TNT(potential_positions):
    final_TNT_positions = []
    for position in potential_positions:
        if (uniform(0.0, 1.0) < TNT_block_probability):
            final_TNT_positions.append(position)
    return final_TNT_positions


def read_limit(filename):
    with open(filename, "r") as file:
        l = file.readline().strip('\n').split(',')
        function_x = []
        function_y = []
        lx = []
        ly = []
        while (l != ['']):
            print(l)
            function_x.append(l[0])
            lx.append(l[1])
            l = file.readline().strip('\n').split(',')

        l = file.readline().strip('\n').split(',')
        while (l != ['']):
            print(l)
            function_y.append(l[0])
            ly.append(l[1])
            l = file.readline().strip().strip('\n').split(',')

        middle = float(file.readline().strip('\n'))
        return Piecewise(*[(sympify(f), y < float(lx)) if i != len(function_x)-1 else (sympify(f), y <= float(lx))
                           for i, (f, lx)
                           in enumerate(zip(function_x, lx))]), Piecewise(*[(sympify(f), x <= float(ly))
                                                                            for f, ly
                                                                            in zip(function_y, ly)]), float(lx[len(lx)-1]), middle
    return False


def limit_boundary(structure_height):
    startpoint = round(float(px.subs(y, structure_height)), 2)
    current_max_width = (
        middle - startpoint)*2
    print(structure_height, current_max_width)
    # print(current_max_width,m_height)
    return startpoint, round(startpoint+float(current_max_width), 2)


def height_limit(position, point):  # limit problems
    position = middle-Abs(middle-position)
    y_limit = py.subs(x, position)
    # print("y_limit", y_limit)
    if y_limit == 0:
        return min(list(filter(lambda x: x >= point, solve(px-position, y))))
    else:
        return min(list(filter(lambda x: x >= point, solve(px-position, y)+[y_limit])))

# write level out in desired xml format


def write_level_xml(folder, complete_locations, selected_other, final_pig_positions, final_TNT_positions, final_platforms, number_birds, current_level, restricted_combinations):
    f = open(folder+"/level-%02d.xml" % current_level, "w")

    f.write('<?xml version="1.0" encoding="utf-16"?>\n')
    f.write('<Level width ="2">\n')
    f.write('<Camera x="0" y="2" minWidth="20" maxWidth="30">\n')
    f.write('<Birds>\n')
    for i in range(number_birds):   # bird type is chosen using probability table
        f.write('<Bird type="%s"/>\n' %
                bird_names[str(choose_item(bird_probabilities))])
    f.write('</Birds>\n')
    f.write('<Slingshot x="-8" y="-2.5">\n')
    f.write('<GameObjects>\n')

    for i in complete_locations:
        # material is chosen randomly
        material = materials[randint(0, len(materials)-1)]
        # if material if not allowed for block type then pick again
        while [material, block_names[str(i[0])]] in restricted_combinations:
            material = materials[randint(0, len(materials)-1)]
        rotation = 0
        if (i[0] in (3, 7, 9, 11, 13)):
            rotation = 90
        f.write('<Block type="%s" material="%s" x="%s" y="%s" rotation="%s" />\n' %
                (block_names[str(i[0])], material, str(i[1]), str(i[2]), str(rotation)))

    for i in selected_other:
        # material is chosen randomly
        material = materials[randint(0, len(materials)-1)]
        # if material if not allowed for block type then pick again
        while [material, additional_objects[str(i[0])]] in restricted_combinations:
            material = materials[randint(0, len(materials)-1)]
        if i[0] == '2':
            facing = randint(0, 1)
            f.write('<Block type="%s" material="%s" x="%s" y="%s" rotation="%s" />\n' %
                    (additional_objects[i[0]], material, str(i[1]), str(i[2]), str(facing*90.0)))
        else:
            f.write('<Block type="%s" material="%s" x="%s" y="%s" rotation="0" />\n' %
                    (additional_objects[i[0]], material, str(i[1]), str(i[2])))

    for i in final_pig_positions:
        f.write('<Pig type="BasicSmall" material="" x="%s" y="%s" rotation="0" />\n' %
                (str(i[0]), str(i[1])))

    for i in final_platforms:
        for j in i:
            f.write('<Platform type="Platform" material="" x="%s" y="%s" />\n' %
                    (str(j[0]), str(j[1])))

    for i in final_TNT_positions:
        f.write('<TNT type="" material="" x="%s" y="%s" rotation="0" />\n' %
                (str(i[0]), str(i[1])))

    f.write('</GameObjects>\n')
    f.write('</Level>\n')

    f.close()


if __name__ == '__main__':
    px, py, m_height, middle = read_limit("limit_parameter2.txt")
    print(px)
    print(py)
    print(time.ctime())
    structures = generate(m_height)
    construct(structures, sys.argv[1])
