from random import randint
from random import uniform
from math import sqrt, ceil
import copy
from sympy import *
import numpy as np
import itertools
import pandas as pd
import os.path

# blocks number and size
blocks = {'1': [0.84, 0.84], '2': [0.85, 0.43], '3': [0.43, 0.85], '4': [0.43, 0.43],
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
        self.position = 0
        self.point = 0
        self.max_height = 0
        self.is_start = 0

    def print(self):
        # for i in self.__dict__.items():
        #     print(i)
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
    start = Node(None)
    start.g = 0
    start.h = 0
    start.is_start = 1
    # openlist = []
    # closelist = [start]
    # complete_locations = []
    leaf_node = []
    temp_leaf = []
    temp_leaf.append(start)
    leaf_node.append(start)
    step = 1
    height = 0
    # relate to previous child
    x1, x2 = limit_boundary(height)

    # maybe using linspace
    sections = np.arange(x1, x2, 0.22)

    while True:

        if len(temp_leaf) == 0:
            break
        leaf_node = copy.copy(temp_leaf)
        temp_leaf.clear()

        print(step)
        for leaf in leaf_node:

            leaf.print()

            # return structure when cannot add more blocks
            if leaf.current_structure_height+0.22 > structure_height:
                return start
            parents = []
            temp_parents = []
            x1, x2 = limit_boundary(
                leaf.current_structure_height+leaf.max_height)
            sections = np.arange(x1, x2, 0.22)
            parents.append(leaf)
            print(x1, x2)
            # each position
            for position in sections:
                position = round(position, 2)
                print(position)
                temp_parents.clear()
                # blocks in the same position
                print("parents", len(parents))
                for parent_node in parents:

                    # parent_node.print()
                    childlist = generate_child(parent_node, step)
                    empty = True
                    # print("childlist", len(childlist))
                    for child in childlist:
                        if parent_node.max_height == 0:
                            child.max_height = blocks[child.block][1]
                        else:
                            child.max_height = parent_node.max_height
                        child.position = position
                        child.point = find_point(position, parent_node)
                        if child.point < parent_node.current_structure_height:
                            break
                        if check_stablity(child, parent_node) and (position+blocks[child.block][0]) <= x2 and (child.point+blocks[child.block][1] <= height_limit(position)) and (child.max_height >= blocks[child.block][1]):
                            # if position+blocks[child.block][0]+0.22 > x2:
                            #     if blocks[child.block][1] == child.max_height:
                            #         child.parent = parent_node
                            #         if position > x2-0.22:
                            #             temp_leaf.append(child)
                            #             # child.print()
                            # else:
                            child.parent = parent_node
                            if position+blocks[child.block][0] > x2-0.22:
                                temp_leaf.append(child)
                                # child.print()
                            temp_parents.append(child)
                            #empty = False
                            # child.print()
                        # initialize current_structure
                        if position == x1:
                            child.current_structure_height = parent_node.current_structure_height + \
                                parent_node.max_height
                        else:
                            child.current_structure_height = parent_node.current_structure_height
                        # child.print()
                    # if no child is suitable
                    # if empty:
                    child = Node(parent_node)
                    child.block = str(0)
                    child.max_height = parent_node.max_height
                    temp_parents.append(child)
                    # child.print()

                    if position == x1:
                        child.current_structure_height = parent_node.current_structure_height + \
                            parent_node.max_height
                    else:
                        child.current_structure_height = parent_node.current_structure_height

                        # if position > x2-0.22:
                        #     temp_leaf.append(child)

                parents.clear()
                parents = copy.copy(temp_parents)
        if step == 1:
            prune(start, temp_leaf)
        leaf_node.clear()
        step += 1

    construct(start)


def construct(node):
    pass


def prune(parent, leaves):
    columns = []
    for leaf in leaves:
        column = []
        while leaf is not parent:
            column.insert(0, leaf)
            leaf = leaf.parent
        columns.append(column)
    print("len:", len(columns))
    # sort columns according to columns' first node
    columns_tree = []
    for x in columns:
        if x[0] not in columns_tree:
            columns_tree.append([x[0]])
        else:
            ([x for x in columns_tree if x[0] in x][0]).append(x)

    for x in range(0, len(columns_tree)):
        cosine_simility(columns_tree[x])


def cosine_simility(columns):
    print(type(columns))
    start, end = limit_boundary(
        columns[0].current_structure_height+columns[0].max_height)
    for index, val in enumerate(columns[1:]):
        columns[1+index] = vectorization(val, start.end)
        unit_vector = np.zeros(len(columns[1+index]))
        unit_vector[0] = 1
        # calculate cosine value between vector and unit vector
        columns[1+index] = np.dot(columns[1+index],
                                  unit_vector)/np.linalg.norm(columns[1+index])


def vectorization(column, start, end):
    column_vector = np.zeros((np.arange(start, end, 0.22), 2))
    for block in column:
        if block.block != str(0):
            width = blocks[block.block][0]
            height = blocks[block.block][1]
            position = round(block.position/0.22)
            for x in np.arange(0, width, 0.22):
                if x+0.22 < width:
                    column_vector[position+x/0.22][0] = 0.22
                elif x+0.22 >= width:
                    column_vector[position+x/0.22][0] = x+0.22-width
                column_vector[position+x/0.22][1] = height
    column_vector_flatten = column_vector.flatten()
    df = pd.DataFrame([column_vector_flatten])
    if os.path.isfile("export.csv"):
        with open('export.csv', 'a') as f:
            df.to_csv(f, header=False)
    else:
        df.to_csv('export.csv')

    return column_vector_flatten


def cluster(columns):
    pass


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
    nd = copy.deepcopy(parent)
    start = node.position
    end = node.position+blocks[node.block][0]
    shadow_blocks = []
    contiguous_blocks = []
    if nd.is_start == 1:
        return True
    while nd.is_start != 1:
        if (nd.position+blocks[nd.block][0] > start and nd.position+blocks[nd.block][0] < end) or (nd.position > start and nd.position < end):
            shadow_blocks.append(nd)
        nd = nd.parent

    # sort according to height
    shadow_blocks.sort(key=lambda x: x.point +
                       blocks[x.block][1], reverse=False)
    for block in shadow_blocks:
        if (block.point+blocks[block.block][1] > nd.point and block.point < point) and ((block.position > start and block.position < end) or (block.position+blocks[block.block][0] > start and block.position+blocks[block.block][0] < end)):
            return False
    return True


def check_stablity(node, parent):
    nd = copy.deepcopy(parent)
    start = node.position
    end = node.position+blocks[node.block][0]
    shadow_blocks = []
    contiguous_blocks = []
    if nd.is_start == 1 or node.current_structure_height == 0:
        return True
    while nd.is_start != 1:
        if nd.block == "0":
            nd = nd.parent
            continue
        if (nd.position+blocks[nd.block][0] > start and nd.position+blocks[nd.block][0] < end) or (nd.position > start and nd.position < end):
            shadow_blocks.append(nd)
        nd = nd.parent

    shadow_blocks.sort(key=lambda x: x.point +
                       blocks[x.block][1], reverse=False)

    max_point = max(shadow_blocks, key=lambda x: x.point+blocks[x.block])

    for block in shadow_blocks:
        if block.point+blocks[block.block][1] == max_point:
            contiguous_blocks.append(block)

    contiguous_blocks.sort(key=lambda x: x.position, reverse=False)

    if len(contiguous_blocks) == 1:
        if blocks[nd.block][0] >= blocks[contiguous_blocks[0].block][0]:
            if nd.position+(blocks[nd.block][0])/2.0 == contiguous_blocks[0].position+(blocks[contiguous_blocks[0]][0])/2.0:
                return True
            else:
                return False
        else:
            return True
    elif len(contiguous_blocks) >= 2:
        if contiguous_blocks[0].position+blocks[contiguous_blocks[0].block][0] > start and contiguous_blocks[len(contiguous_blocks)].position < end:
            return True
        else:
            return False


def find_point(position, node):
    current_structure_height = node.current_structure_height
    nd = copy.deepcopy(node)
    while nd.is_start != 1:
        if nd.block == "0":
            nd = nd.parent
            continue
        if nd.point == current_structure_height:
            nd = nd.parent
            continue
        if nd.position < position and nd.position+blocks[nd.block][0] > position:
            return nd.point+blocks[nd.block][1]
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


def generate_child(parent_node, step):
    childlist = []
    for key, val in blocks.items():
        # step is odd number, horizontal alignment
        if step % 2 == 1:
            if val[0] > val[1]:
                continue
        # step is even number, vertical alignment
        else:
            if val[0] < val[1]:
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
        childlist.append(child)
    return childlist


# creates the peaks (first row) of the structure


def make_peaks(center_point):

    current_tree_bottom = []        # bottom blocks of structure
    # this is the number of peaks the structure will have
    number_peaks = randint(1, max_peaks)
    # this is the item at top of structure
    top_item = choose_item(probability_table_blocks)

    if number_peaks == 1:
        current_tree_bottom.append([top_item, center_point])

    if number_peaks == 2:
        distance_apart_extra = round(
            randint(min_peak_split, max_peak_split)/100.0, 10)
        current_tree_bottom.append([top_item, round(
            center_point - (blocks[str(top_item)][0]*0.5) - distance_apart_extra, 10)])
        current_tree_bottom.append([top_item, round(
            center_point + (blocks[str(top_item)][0]*0.5) + distance_apart_extra, 10)])

    if number_peaks == 3:
        distance_apart_extra = round(
            randint(min_peak_split, max_peak_split)/100.0, 10)
        current_tree_bottom.append([top_item, round(
            center_point - (blocks[str(top_item)][0]) - distance_apart_extra, 10)])
        current_tree_bottom.append([top_item, round(center_point, 10)])
        current_tree_bottom.append([top_item, round(
            center_point + (blocks[str(top_item)][0]) + distance_apart_extra, 10)])

    if number_peaks == 4:
        distance_apart_extra = round(
            randint(min_peak_split, max_peak_split)/100.0, 10)
        current_tree_bottom.append([top_item, round(
            center_point - (blocks[str(top_item)][0]*1.5) - (distance_apart_extra*2), 10)])
        current_tree_bottom.append([top_item, round(
            center_point - (blocks[str(top_item)][0]*0.5) - distance_apart_extra, 10)])
        current_tree_bottom.append([top_item, round(
            center_point + (blocks[str(top_item)][0]*0.5) + distance_apart_extra, 10)])
        current_tree_bottom.append([top_item, round(
            center_point + (blocks[str(top_item)][0]*1.5) + (distance_apart_extra*2), 10)])

    if number_peaks == 5:
        distance_apart_extra = round(
            randint(min_peak_split, max_peak_split)/100.0, 10)
        current_tree_bottom.append([top_item, round(
            center_point - (blocks[str(top_item)][0]*2.0) - (distance_apart_extra*2), 10)])
        current_tree_bottom.append([top_item, round(
            center_point - (blocks[str(top_item)][0]) - distance_apart_extra, 10)])
        current_tree_bottom.append([top_item, round(center_point, 10)])
        current_tree_bottom.append([top_item, round(
            center_point + (blocks[str(top_item)][0]) + distance_apart_extra, 10)])
        current_tree_bottom.append([top_item, round(
            center_point + (blocks[str(top_item)][0]*2.0) + (distance_apart_extra*2), 10)])
    return current_tree_bottom


# recursively adds rows to base of strucutre until max_width or max_height is passed
# once this happens the last row added is removed and the structure is returned

def make_structure(absolute_ground, center_point, max_width, max_height):

    total_tree = []                 # all blocks of structure (so far)
    # creates the first row (peaks) for the structure, ensuring that max_width
    # restriction is satisfied
    current_tree_bottom = make_peaks(center_point)

    max_width, max_height = limit_boundary(
        max_width, max_height, (blocks[str(current_tree_bottom[0][0])][1])/2)
    if max_width > 0.0:
        while find_structure_width(current_tree_bottom) > max_width:
            current_tree_bottom = make_peaks(center_point)
            max_width, max_height = limit_boundary(
                max_width, max_height, (blocks[str(current_tree_bottom[0][0])][1])/2)

    total_tree.append(current_tree_bottom)

    # recursively add more rows of blocks to the level structure
    structure_width = find_structure_width(current_tree_bottom)
    structure_height = (blocks[str(current_tree_bottom[0][0])][1]/2)
    print('w', structure_width, 'h', structure_height,
          'w', max_width, 'h', max_height)
    # print(current_width(structure_height),1,limit_boundary(max_width,max_height,current_tree_bottom))
    if max_height > 0.0 or max_width > 0.0:
        pre_total_tree = [current_tree_bottom]
        while structure_height < max_height and structure_width < max_width:
            total_tree, current_tree_bottom = add_new_row(
                current_tree_bottom, total_tree)
            complete_locations = []
            ground = absolute_ground
            for row in reversed(total_tree):
                for item in row:
                    complete_locations.append([item[0], item[1], round(
                        (((blocks[str(item[0])][1])/2)+ground), 10)])
                ground = ground + (blocks[str(item[0])][1])
            structure_height = find_structure_height(complete_locations)
            structure_width = find_structure_width(complete_locations)
            # print(structure_width,find_structure_width(current_tree_bottom))
            max_width, max_height = limit_boundary(
                max_width, max_height, structure_height)
            print('w:', structure_width, 'h:', structure_height,
                  'w', max_width, 'h', max_height)

            if structure_height > max_height:
                total_tree = deepcopy(pre_total_tree)
            elif structure_width > max_width:
                total_tree = deepcopy(pre_total_tree)
                complete_locations = []
                ground = absolute_ground
                for row in reversed(total_tree):
                    for item in row:
                        complete_locations.append([item[0], item[1], round(
                            (((blocks[str(item[0])][1])/2)+ground), 10)])
                    ground = ground + (blocks[str(item[0])][1])
                structure_height = find_structure_height(complete_locations)
                structure_width = find_structure_width(complete_locations)
                current_tree_bottom = total_tree[len(total_tree)-1]
                max_width, max_height = limit_boundary(
                    max_width, max_height, structure_height)
            else:
                pre_total_tree = deepcopy(total_tree)
            print('w:', structure_width, 'h:', structure_height,
                  'w', max_width, 'h', max_height)
            print('\n')

    # make structure vertically correct (add y position to blocks)
    complete_locations = []
    ground = absolute_ground
    for row in reversed(total_tree):
        for item in row:
            complete_locations.append([item[0], item[1], round(
                (((blocks[str(item[0])][1])/2)+ground), 10)])
        ground = ground + (blocks[str(item[0])][1])

    print(total_tree)
    print("Width:", find_structure_width(complete_locations))
    print("Height:", find_structure_height(complete_locations))
    # number blocks present in the structure
    print("Block number:", len(complete_locations))

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
    left_bottom = total_tree[-1][0]
    right_bottom = total_tree[-1][-1]
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


# divide the available ground space between the chosen number of ground
# structures

def create_ground_structures():
    valid = False
    while valid == False:
        ground_divides = []
        if number_ground_structures > 0:
            ground_divides = [level_width_min, level_width_max]
        for i in range(number_ground_structures-1):
            ground_divides.insert(
                i+1, uniform(level_width_min, level_width_max))
        valid = True
        print(range(len(ground_divides)-1))
        for j in range(len(ground_divides)-1):
            if (ground_divides[j+1] - ground_divides[j]) < min_ground_width:
                valid = False

    # determine the area available to each ground structure
    ground_positions = []
    ground_widths = []
    for j in range(len(ground_divides)-1):
        ground_positions.append(
            ground_divides[j]+((ground_divides[j+1] - ground_divides[j])/2))
        ground_widths.append(ground_divides[j+1] - ground_divides[j])

    print("number ground structures:", len(ground_positions))
    print("")

    # creates a ground structure for each defined area
    complete_locations = []
    final_pig_positions = []
    for i in range(len(ground_positions)):
        max_width = ground_widths[i]
        max_height = ground_structure_height_limit
        print("max_width", max_width, "max_height", max_height)
        center_point = ground_positions[i]
        complete_locations2, final_pig_positions2 = make_structure(
            absolute_ground, center_point, max_width, max_height)
        complete_locations = complete_locations + complete_locations2
        final_pig_positions = final_pig_positions + final_pig_positions2

    return len(ground_positions), complete_locations, final_pig_positions


# creates a set number of platforms within the level
# automatically reduced if space not found after set number of attempts

def create_platforms(number_platforms, complete_locations, final_pig_positions):

    platform_centers = []
    attempts = 0            # number of attempts so far to find space for platform
    final_platforms = []
    while len(final_platforms) < number_platforms:
        platform_width = randint(4, 7)
        platform_position = [uniform(level_width_min+((platform_width*platform_size[0])/2.0), level_width_max-((platform_width*platform_size[0])/2.0)),
                             uniform(level_height_min, (level_height_max - minimum_height_gap))]
        temp_platform = []

        if platform_width == 1:
            temp_platform.append(platform_position)

        if platform_width == 2:
            temp_platform.append(
                [platform_position[0] - (platform_size[0]*0.5), platform_position[1]])
            temp_platform.append(
                [platform_position[0] + (platform_size[0]*0.5), platform_position[1]])

        if platform_width == 3:
            temp_platform.append(
                [platform_position[0] - (platform_size[0]), platform_position[1]])
            temp_platform.append(platform_position)
            temp_platform.append(
                [platform_position[0] + (platform_size[0]), platform_position[1]])

        if platform_width == 4:
            temp_platform.append(
                [platform_position[0] - (platform_size[0]*1.5), platform_position[1]])
            temp_platform.append(
                [platform_position[0] - (platform_size[0]*0.5), platform_position[1]])
            temp_platform.append(
                [platform_position[0] + (platform_size[0]*0.5), platform_position[1]])
            temp_platform.append(
                [platform_position[0] + (platform_size[0]*1.5), platform_position[1]])

        if platform_width == 5:
            temp_platform.append(
                [platform_position[0] - (platform_size[0]*2.0), platform_position[1]])
            temp_platform.append(
                [platform_position[0] - (platform_size[0]), platform_position[1]])
            temp_platform.append(platform_position)
            temp_platform.append(
                [platform_position[0] + (platform_size[0]), platform_position[1]])
            temp_platform.append(
                [platform_position[0] + (platform_size[0]*2.0), platform_position[1]])

        if platform_width == 6:
            temp_platform.append(
                [platform_position[0] - (platform_size[0]*2.5), platform_position[1]])
            temp_platform.append(
                [platform_position[0] - (platform_size[0]*1.5), platform_position[1]])
            temp_platform.append(
                [platform_position[0] - (platform_size[0]*0.5), platform_position[1]])
            temp_platform.append(
                [platform_position[0] + (platform_size[0]*0.5), platform_position[1]])
            temp_platform.append(
                [platform_position[0] + (platform_size[0]*1.5), platform_position[1]])
            temp_platform.append(
                [platform_position[0] + (platform_size[0]*2.5), platform_position[1]])

        if platform_width == 7:
            temp_platform.append(
                [platform_position[0] - (platform_size[0]*3.0), platform_position[1]])
            temp_platform.append(
                [platform_position[0] - (platform_size[0]*2.0), platform_position[1]])
            temp_platform.append(
                [platform_position[0] - (platform_size[0]), platform_position[1]])
            temp_platform.append(platform_position)
            temp_platform.append(
                [platform_position[0] + (platform_size[0]), platform_position[1]])
            temp_platform.append(
                [platform_position[0] + (platform_size[0]*2.0), platform_position[1]])
            temp_platform.append(
                [platform_position[0] + (platform_size[0]*3.0), platform_position[1]])

        overlap = False
        for platform in temp_platform:

            if (((platform[0]-(platform_size[0]/2)) < level_width_min) or ((platform[0]+(platform_size[0])/2) > level_width_max)):
                overlap = True

            for block in complete_locations:
                if (round((platform[0] - platform_distance_buffer - platform_size[0]/2), 10) <= round((block[1] + blocks[str(block[0])][0]/2), 10) and
                        round((platform[0] + platform_distance_buffer + platform_size[0]/2), 10) >= round((block[1] - blocks[str(block[0])][0]/2), 10) and
                        round((platform[1] + platform_distance_buffer + platform_size[1]/2), 10) >= round((block[2] - blocks[str(block[0])][1]/2), 10) and
                        round((platform[1] - platform_distance_buffer - platform_size[1]/2), 10) <= round((block[2] + blocks[str(block[0])][1]/2), 10)):
                    overlap = True

            for pig in final_pig_positions:
                if (round((platform[0] - platform_distance_buffer - platform_size[0]/2), 10) <= round((pig[0] + pig_size[0]/2), 10) and
                        round((platform[0] + platform_distance_buffer + platform_size[0]/2), 10) >= round((pig[0] - pig_size[0]/2), 10) and
                        round((platform[1] + platform_distance_buffer + platform_size[1]/2), 10) >= round((pig[1] - pig_size[1]/2), 10) and
                        round((platform[1] - platform_distance_buffer - platform_size[1]/2), 10) <= round((pig[1] + pig_size[1]/2), 10)):
                    overlap = True

            for platform_set in final_platforms:
                for platform2 in platform_set:
                    if (round((platform[0] - platform_distance_buffer - platform_size[0]/2), 10) <= round((platform2[0] + platform_size[0]/2), 10) and
                            round((platform[0] + platform_distance_buffer + platform_size[0]/2), 10) >= round((platform2[0] - platform_size[0]/2), 10) and
                            round((platform[1] + platform_distance_buffer + platform_size[1]/2), 10) >= round((platform2[1] - platform_size[1]/2), 10) and
                            round((platform[1] - platform_distance_buffer - platform_size[1]/2), 10) <= round((platform2[1] + platform_size[1]/2), 10)):
                        overlap = True

            for platform_set2 in final_platforms:
                for i in platform_set2:
                    if i[0]+platform_size[0] > platform[0] and i[0]-platform_size[0] < platform[0]:
                        if i[1]+minimum_height_gap > platform[1] and i[1]-minimum_height_gap < platform[1]:
                            overlap = True

        if overlap == False:
            final_platforms.append(temp_platform)
            platform_centers.append(platform_position)

        attempts = attempts + 1
        if attempts > max_attempts:
            attempts = 0
            number_platforms = number_platforms - 1

    print("number platforms:", number_platforms)
    print("")

    return number_platforms, final_platforms, platform_centers


# create sutiable structures for each platform

def create_platform_structures(final_platforms, platform_centers, complete_locations, final_pig_positions):
    current_platform = 0
    for platform_set in final_platforms:
        platform_set_width = len(platform_set)*platform_size[0]

        above_blocks = []
        for platform_set2 in final_platforms:
            if platform_set2 != platform_set:
                for i in platform_set2:
                    if i[0]+platform_size[0] > platform_set[0][0] and i[0]-platform_size[0] < platform_set[-1][0] and i[1] > platform_set[0][1]:
                        above_blocks.append(i)

        min_above = level_height_max
        for j in above_blocks:
            if j[1] < min_above:
                min_above = j[1]

        center_point = platform_centers[current_platform][0]
        absolute_ground = platform_centers[
            current_platform][1] + (platform_size[1]/2)

        max_width = platform_set_width
        max_height = (min_above - absolute_ground) - \
            pig_size[1] - platform_size[1]

        complete_locations2, final_pig_positions2 = make_structure(
            absolute_ground, center_point, max_width, max_height)
        complete_locations = complete_locations + complete_locations2
        final_pig_positions = final_pig_positions + final_pig_positions2

        current_platform = current_platform + 1

    return complete_locations, final_pig_positions


# remove random pigs until number equals the desired amount

def remove_unnecessary_pigs(number_pigs):
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


x = symbols("x")
y = symbols("y")


def read_limit():
    file = open("limit_parameter.txt", "r")
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
    return Piecewise(*[(sympify(f), y <= float(lx))
                       for f, lx
                       in zip(function_x, lx)]), Piecewise(*[(sympify(f), x <= float(ly))
                                                             for f, ly
                                                             in zip(function_y, ly)]), float(lx[len(lx)-1]), middle

px, py, m_height, middle = read_limit()
print(px)
print(py)


def limit_boundary(structure_height):
    current_max_width = (
        middle-px.subs(y, structure_height))*2
    # print(current_max_width,m_height)
    return px.subs(y, structure_height), px.subs(y, structure_height)+current_max_width


def height_limit(position):
    position = middle-Abs(middle-position)
    y_limit = py.subs(x, position)
    #print("y_limit", y_limit)
    if y_limit == 0:
        return min(solve(px-position, y))
    else:
        return min(solve(px-position, y)+[y_limit])

# write level out in desired xml format


def write_level_xml(complete_locations, selected_other, final_pig_positions, final_TNT_positions, final_platforms, number_birds, current_level, restricted_combinations):

    f = open("S:\i.t\AI\WeirdAliens-Windows-2.0\WeirdAliens-Windows\WeirdAliens_Data\StreamingAssets\Levels\level-%s.xml" % current_level, "w")

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


# generate levels using input parameters


generate(m_height)
# backup_probability_table_blocks = copy.deepcopy(probability_table_blocks)
# backup_materials = copy.deepcopy(materials)

# FILE = open("parameters.txt", 'r')
# checker = FILE.readline()
# finished_levels = 0
# while (checker != ""):
#     if checker == "\n":
#         checker = FILE.readline()
#     else:
#         # the number of levels to generate
#         number_levels = int(copy.deepcopy(checker))
#         # block type and material combination that are banned from the level
#         restricted_combinations = FILE.readline().split(',')
#         for i in range(len(restricted_combinations)):
#             # if all materials are baned for a block type then do not use that
#             # block type
#             restricted_combinations[i] = restricted_combinations[i].split()
#         pig_range = FILE.readline().split(',')
#         # time limit to create the levels, shouldn't be an issue for most
#         # generators (approximately an hour for 10 levels)
#         time_limit = int(FILE.readline())
#         checker = FILE.readline()

#         # block types that cannot be used with any materials
#         restricted_blocks = []
#         for key, value in block_names.items():
#             completely_restricted = True
#             for material in materials:
#                 if [material, value] not in restricted_combinations:
#                     completely_restricted = False
#             if completely_restricted == True:
#                 restricted_blocks.append(value)

#         probability_table_blocks = copy.deepcopy(
#             backup_probability_table_blocks)
#         trihole_allowed = True
#         tri_allowed = True
#         cir_allowed = True
#         cirsmall_allowed = True
#         TNT_allowed = True

#         # remove restricted block types from the structure generation process
#         probability_table_blocks = remove_blocks(restricted_blocks)
#         if "TriangleHole" in restricted_blocks:
#             trihole_allowed = False
#         if "Triangle" in restricted_blocks:
#             tri_allowed = False
#         if "Circle" in restricted_blocks:
#             cir_allowed = False
#         if "CircleSmall" in restricted_blocks:
#             cirsmall_allowed = False

#         for current_level in range(number_levels):

#             number_ground_structures = 1                     # number of ground structures
#             # number of platforms (reduced automatically if not enough space)
#             number_platforms = 0
#             # number of pigs (if set too large then can cause program to
#             # infinitely loop)
#             number_pigs = randint(int(pig_range[0]), int(pig_range[1]))

#             if (current_level+finished_levels+4) < 10:
#                 level_name = "0"+str(current_level+finished_levels+4)
#             else:
#                 level_name = str(current_level+finished_levels+4)

#             number_ground_structures, complete_locations, final_pig_positions = create_ground_structures()
#             number_platforms, final_platforms, platform_centers = create_platforms(
#                 number_platforms, complete_locations, final_pig_positions)
#             complete_locations, final_pig_positions = create_platform_structures(
#                 final_platforms, platform_centers, complete_locations, final_pig_positions)
#             final_pig_positions, removed_pigs = remove_unnecessary_pigs(
#                 number_pigs)
#             final_pig_positions = add_necessary_pigs(number_pigs)
#             final_TNT_positions = add_TNT(removed_pigs)
#             number_birds = choose_number_birds(
#                 final_pig_positions, number_ground_structures, number_platforms)
#             possible_trihole_positions, possible_tri_positions, possible_cir_positions, possible_cirsmall_positions = find_additional_block_positions(
#                 complete_locations)
#             selected_other = add_additional_blocks(
#                 possible_trihole_positions, possible_tri_positions, possible_cir_positions, possible_cirsmall_positions)
#             write_level_xml(complete_locations, selected_other, final_pig_positions, final_TNT_positions,
#                             final_platforms, number_birds, level_name, restricted_combinations)
#         finished_levels = finished_levels + number_levels
