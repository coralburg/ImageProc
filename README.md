# ImageProc
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib
import numpy as np
import math
from numpy import linalg
import datetime
import os


margins = 4
w = 1240
h = 1754
boxes = [[116, 175], [271, 341], [421, 481], [534, 590], [647, 708], [763, 821], [875, 934], [988, 1049], [1102, 1162], [1215, 1276], [1330, 1389], [1443, 1504], [1562, 1638]]
MAX_CHAN = 255
RNG_ERROR = 5
WHITE_RGB = [MAX_CHAN, MAX_CHAN, MAX_CHAN]
RNG_MAX = MAX_CHAN - RNG_ERROR
NUM_OF_PIX_RECT = 400
RNG_ERR_RECT = 0.5
EDGE_SIZE = 21
MAX_BLACK_CHAN = 100
FULL_ROUND = 180
MID_OF_IMG = 570
DIF_REG_LINES = 230
RANGE_MID = range(400, MID_OF_IMG)
PRINT_BEG = 13
PRINT_BEG_MAX = PRINT_BEG + 3
PADDING_DOWN_LINE = 30
PADD = 3
avg_letter_size = 20
rectangle_side = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def main_func(name):

    # getting the img, size, and the data:
    data, img = get_image(name)
    # save coordinates of the black rectangles
    cords = save_cords(data)
    # perspective transform on the margins
    aligned = get_perspective(img, cords)
    # rotate 180 deg the img if needed, and load matrix in order to manipulate image
    aligned_data, aligned = check_opp(aligned)
    written_ranges, sides, components = curr_func(aligned_data)
    tree = xml(written_ranges, name, sides, components)
    # aligned_data = paint_lines(written_ranges, aligned_data)
    aligned = Image.fromarray(aligned_data, 'RGB')
    return aligned, tree


def curr_func(data):
    aligned_data = np.copy(data)
    # raise the blue channel in the matrix in order to prepare the matrix to binarization.
    aligned_data = raise_blue(aligned_data)
    # delete the skeleton between boxes
    aligned = Image.fromarray(aligned_data, 'RGB')
    # aligned.show()
    aligned_data = between_lines(aligned_data)
    # we need to render the img after all manipulation
    aligned = Image.fromarray(aligned_data, 'RGB')
    # grayscale and binarization
    gray = aligned.convert('L')
    bw = gray.point(lambda x: 0 if x < 200 else 255, '1')
    # bw.show()
    bin_data = np.array(bw)
    # finding bounds in y axis of each written line of input
    written_ranges = find_written_rows(bin_data)
    # now with each range of input we find connected components in order to analyze each component
    sides, components = connected_components(bin_data, written_ranges)
    # xml(written_ranges)
    return written_ranges, sides, components


# getting by image name the properties of img
def get_image(img_name):
    img = Image.open(img_name)
    img = img.resize((w, h), Image.ANTIALIAS)
    data = np.array(img)
    return data, img


# method that manage the search of alignments exact coordinates
def save_cords(data):
    # where to look at(x,y axis)
    margins_side = math.floor(w / margins)
    margins_vertical = math.floor(h / margins)
    # up-left - no need to flip
    arr_u_l = convert_to_bin_array(data, 0, margins_vertical, 0, margins_side)
    i_u_l, j_u_l = find_edges(arr_u_l, False)
    # up-right - flip the x axis
    arr_u_r = convert_to_bin_array(data, 0, margins_vertical, w - margins_side, w)
    arr_u_r = flip_by_axis(arr_u_r, True, False)
    i_u_r, j_u_r = find_edges(arr_u_r, False)
    # back to original coordinates
    j_u_r = w - j_u_r
    # down-left - flip in y axis
    arr_d_l = convert_to_bin_array(data, h - margins_vertical, h, 0, margins_side)
    arr_d_l = flip_by_axis(arr_d_l, False, True)
    i_d_l, j_d_l = find_edges(arr_d_l, True)
    # back to original coordinates
    i_d_l = h - i_d_l
    # down_right - flip in both axises
    arr_d_r = convert_to_bin_array(data,  h - margins_vertical, h, w - margins_side, w)
    arr_d_r = flip_by_axis(arr_d_r, True, True)
    i_d_r, j_d_r = find_edges(arr_d_r, True)
    # back to original in both axises
    j_d_r = w - j_d_r
    i_d_r = h - i_d_r
    # put all the data in the array
    arr = [(j_u_l, i_u_l), (j_u_r, i_u_r), (j_d_r, i_d_r), (j_d_l, i_d_l)]
    return arr


# its more convenient to look in 2d binary array than in 3d rgb
def convert_to_bin_array(data, from_i, to_i, from_j, to_j):
    arr = []
    # just in specific range in margins
    for i in range(from_i, to_i):
        line = []
        for j in range(from_j, to_j):
            col_to_app = 0
            if is_black(data[i][j]):
                col_to_app = 1
            line.append(col_to_app)
        arr.append(line)
    return arr


# check by definition of black color(lowest channel), can be updated by need
def is_black(pix):
    for ind in range(3):
        if pix[ind] > MAX_BLACK_CHAN:
            return False
    return True


# flip binaric array in order to look from [0,0] in the bin array in each margin
def flip_by_axis(arr, right, down):
    if right:
        arr = np.flip(arr, 1)
    if down:
        arr = np.flip(arr, 0)
    return arr


# find the first sharp "change" - the edge
def find_edges(arr, b):
    range_i, range_j, range_delta = len(arr), len(arr[0]), 1
    rng_matrix = EDGE_SIZE - range_delta
    if b:
        tmp = range_i
        range_i, range_j = range_j, tmp
    for i in range(range_delta, range_i-rng_matrix):
        for j in range(range_delta, range_j-rng_matrix):
            if edge_matrix(arr, i, j, b):
                delta_i, counter = 1, 1
                while delta_i < rng_matrix:
                    if b:
                        is_line = edge_matrix(arr, i, j+delta_i, b)
                    else:
                        is_line = edge_matrix(arr, i+delta_i, j, b)
                    if is_line:
                        counter = counter + 1
                    delta_i = delta_i + 1
                # enough tests are pass, its not a noise - probably rectangle!
                if counter >= rng_matrix*RNG_ERR_RECT:
                    # verify rectangle
                    if is_rect(i, j, arr, b):
                        # back to original coordinates
                        if b:
                            return j, i+1
                        return i, j+1


# the tests for being edge(each i, j)
def edge_matrix(arr, i, j, b):
    if b:
        tmp = i
        i, j = j, tmp
    if arr[i][j] == 0:
        if arr[i][j + 1] == 1:
            return True
    return False


# rectangle side is defined up, 21 pixels for each edge and 0's wrap in each direction
def is_rect(i, j, arr, b):
    if b:
        tmp = i
        i, j = j, tmp
    counter = 0
    for x in range(EDGE_SIZE):
        for y in range(EDGE_SIZE):
            if arr[i+x][j+y] == rectangle_side[x][y]:
                counter = counter+1
                # enough witnesses for rectangle
                if counter > NUM_OF_PIX_RECT*RNG_ERR_RECT:
                    return True
    return False


# defining the align coordinates and transform  with perspective projection
def get_perspective(img, cords):
    bases = create_base([[(0, 0), cords[0]], [(w, 0), cords[1]], [(w, h), cords[2]], [(0, h), cords[3]]])
    img = img.transform((w, h), method=Image.PERSPECTIVE, data=bases)
    return img


# create base co-effitions for the transform
def create_base(coordinates):
    arr_to_a, arr_to_b = [], []
    for pair in coordinates:
        first, sec = return_arrays(pair[0], pair[1])
        arr_to_a.append(first)
        arr_to_a.append(sec)
        for ind in range(2):
            arr_to_b.append(pair[1][ind])
    a = np.array(arr_to_a, dtype=np.float32)
    b = np.array(arr_to_b, dtype=np.float32)
    return linalg.solve(a, b)


# manipulation for convenience
def return_arrays(a, b):
    a0, a1, b0, b1 = a[0], a[1], b[0], b[1]
    first = [a0, a1, 1, 0, 0, 0, -b0*a0, -b0*a1]
    second = [0, 0, 0, a0, a1, 1, -b1*a0, -b1*a1]
    return first, second


# if the image was scan in the opposed direction
def check_opp(aligned):
    aligned_data = np.array(aligned)
    b = opp(aligned_data)
    if not b:
        aligned = aligned.rotate(FULL_ROUND)
        aligned_data = np.array(aligned)
    return aligned_data, aligned


# the differentiation between opposed scan and straight one
def opp(data):
    i = PRINT_BEG
    while i < PRINT_BEG_MAX:
        for j in RANGE_MID:
            if is_black(data[i][j]):
                return True
        i = i+1
    return False


# running over all img, and raising the blue channel in each i,j: in order to "ignore" yellow lines
def raise_blue(data):
    for i in range(h):
        for j in range(w):
            data[i][j][2] = MAX_CHAN
    return data


def ignore_large_noises(data):
    for i in range(h):
        for j in range(w):
            if not np.array_equal(data[i][j], WHITE_RGB):
                if data[i][j][0]>200:
                    if data[i][j][1]>200:
                        data[i][j] = WHITE_RGB

    return data


# delete printed (skeleton)
def between_lines(data):
    # delete up the yellows
    data = delete_colors(data, range(boxes[0][0]), range(w), [0], [False], [RNG_MAX])
    for x in range(2):
        # delete red(pink) skeleton
        data = delete_colors(data, range(boxes[x][1], boxes[x+1][0]), range(MID_OF_IMG, w), [1, 0], [False, True],
                             [240, 50])
        # delete black skeleton
        data = delete_colors(data, range(boxes[x][1] + PADDING_DOWN_LINE, boxes[x+1][0]),  range(w), [1], [False],
                             [RNG_MAX])
    # delete only the printed(from the mid and right)
    for x in range(11):
        data = delete_colors(data, range(boxes[x][1] + PADDING_DOWN_LINE, boxes[x+1][0] + RNG_ERROR), range(MID_OF_IMG + DIF_REG_LINES,
                                                                                                            w),
                             [0], [False], [RNG_MAX])
    # last line - printed thrugh hall width
    data = delete_colors(data, range(boxes[11][1] + PADDING_DOWN_LINE, boxes[12][0]), range(w), [0], [False], [RNG_MAX])
    return data


# method that actually paint the pixel in white if needed
def delete_colors(data, rng_i, rng_j, ind_rgb, is_bigger_than, lhs):
    for i in rng_i:
        for j in rng_j:
            cont = True
            for rgb in range(len(ind_rgb)):
                ind_to_check = data[i][j][ind_rgb[rgb]]
                if is_bigger_than[rgb]:
                    that_channel = (ind_to_check > lhs[rgb])
                else:
                    that_channel = (ind_to_check < lhs[rgb])
                cont = cont and that_channel
            if cont:
                data[i][j] = WHITE_RGB
    return data


# find the range of written rows
def find_written_rows(bin_data):
    written_rows = []
    # not in down black rectangles
    for i in range(h - 30):
        count_black = 0
        for j in range(w):
            if not bin_data[i][j]:
                count_black = count_black + 1
        # if pixels are written in this rows: add the row to written rows data structure
        if count_black > 0:
            written_rows.append(i)
    return bound_written(written_rows)


def bound_written(written_rows):
    ranges = []
    i = 0
    while i < len(written_rows) - 1:
        first = written_rows[i]
        i = i+1
        while i < len(written_rows) - 1 and (written_rows[i] - written_rows[i-1]) < 2:
            i = i+1
        last = written_rows[i - 1]
        ranges.append([first, last])
    rng_no_noise = []
    # delete rows that marked, but are just noise:
    for rng in ranges:
        if rng[1] - rng[0] > avg_letter_size:
            rng_no_noise.append(rng)
    # padding:
    for ind in range(len(rng_no_noise)):
        rng_no_noise[ind][0] = rng_no_noise[ind][0] - PADD
        rng_no_noise[ind][1] = rng_no_noise[ind][1] + PADD
    return rng_no_noise


# side effect method
def paint_lines(ranges, data):
    for rng in ranges:
        for j in range(w):
            data[rng[0]][j] = [200,0,15]
            data[rng[1]][j] = [0,200,15]
    return data


# this method send each row in iterative way to the component search
def connected_components(bin_data, rows):
    sided = []
    components = []
    for row in rows:
        side, components_row = find_connected_components(row, bin_data)
        sided.append(side)
        components.append(components_row)
    return sided, components


def find_connected_components(row, bin_data):
    connected_components_in_curr_row = []
    sze = row[1] - row[0]
    # maintaining array of components, not used yet
    zrs = np.zeros((sze, w), dtype=int)
    j = w - 1
    count_comp = 1
    # j decreases from convenience reasons - hebrew is written from right to left
    while j > 0:
        i = row[0]
        while i < row[1]:
            if not bin_data[i][j]:
                visited, zrs = dfs(i, j, bin_data, zrs, [[i, j]], count_comp, row[0])
                if visited is not None:
                    next_j = find_min_j(visited, j)
                    # if its not noise: add the bound coordinates to the data structure
                    if (j - next_j) > 3:
                        curr_comp = [[find_max_j(visited, j), next_j],find_min_max_i(visited, i, i)]
                        connected_components_in_curr_row.append(curr_comp)
                        # next component more possibly to start after this, but this is for not getting into infinite
                        # loop or find the same component several times.
                        # anyway, find_max_j find the start of next component even if it start into current component
                        # interval
                        i = curr_comp[1][1]
                        while i < row[1]:
                            for m in range(next_j, j):
                                if not bin_data[i][m]:
                                    if not visited_pixel(i,m,visited):
                                        n_visites, zrs = dfs(i, m, bin_data, zrs, [[i, m]], count_comp, row[0])
                                        if len(n_visites) > 2:
                                            connected_components_in_curr_row.append([[find_max_j(n_visites, m),
                                                                                      find_min_j(n_visites, j)],
                                                                                     find_min_max_i(n_visites, i, i)])
                                            i = row[1]
                                            break
                            i = i + 1
                        i = row[0]
                        j = next_j
                        count_comp = count_comp+1
            i = i+1
        j = j - 1
    co = connected_components_in_curr_row
    row_side = [co[0][0][0] + PADD, co[len(co)-1][0][1]-PADD]
    return row_side, co


# in order to pass the hall component, using the dfs recursive familiar algorithm
def dfs(i, j, data, zrs, visited, counter, start_i):
    zrs[i-start_i][j] = counter
    # where to look for continuation
    neibors = [[i+1, j], [i-1, j], [i, j+1], [i, j-1], [i+1, j+1], [i+1, j-1], [i-1, j+1], [i-1, j-1]]
    for neib in neibors:
        if 0 < (neib[0] - start_i) < len(zrs) and 0 < neib[1] < w and not data[neib[0]][neib[1]]:
            if not visited_pixel(neib[0], neib[1], visited):
                # memoization data structure
                visited.append([neib[0], neib[1]])
                # recursive call
                visited, zrs = dfs(neib[0], neib[1], data, zrs, visited, counter, start_i)
    return visited, zrs


# memoization
def visited_pixel(i, j, visited):
    for pix in visited:
        if i == pix[0] and j == pix[1]:
            return True
    return False


def find_min_j(visited, min_j):
    for pix in visited:
        if min_j > pix[1]:
            min_j = pix[1]
    return min_j


def find_max_j(visited, max_j):
    for pix in visited:
        if max_j < pix[1]:
            max_j = pix[1]
    return max_j


def find_min_max_i(visited, min_i, max_i):
    for pix in visited:
        if min_i > pix[0]:
            min_i = pix[0]
        if max_i < pix[0]:
            max_i = pix[0]
    return [min_i, max_i]


def xml(rows, name, sides, components):
    root = ET.Element("PcGts")
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Coral Burg"
    ET.SubElement(metadata, "Created").text = str(datetime.datetime.now())
    page = ET.SubElement(root, "Page", imageFilename=name, imageWidth=str(w), imageHeight=str(h),
                         readingDirection="right-to-left", primaryLanguage="Hebrew")
    for counter_lines in range(0,len(rows)):
        txt_reg = ET.SubElement(page, "TextLine", id="r"+str(counter_lines), type="textline")
        up, down = str(rows[counter_lines][0]), str(rows[counter_lines][1])
        right, left = str(sides[counter_lines][0]), str(sides[counter_lines][1])
        ET.SubElement(txt_reg, "Coords", mostRight=right, mostLeft=left, highest=up, lowest=down)
        txt_eq = ET.SubElement(txt_reg, "TextEquiv")
        ET.SubElement(txt_eq, "Unicode").text = "The PAGE format for a textline"
        for count_components in range(0, len(components[counter_lines])):
            component = ET.SubElement(txt_reg, "Component", id="c"+str(count_components), type="component")
            curr_comp = components[counter_lines][count_components]
            up_c, down_c = str(curr_comp[1][0]), str(curr_comp[1][1])
            right_c,left_c = str(curr_comp[0][0]), str(curr_comp[0][1])
            ET.SubElement(component, "Coords", mostRight=right_c, mostLeft=left_c, highest=up_c, lowest=down_c)
    tree = ET.ElementTree(root)
    return tree


def main():
    os.mkdir("input_output")
    curr_dir = os.getcwd()
    new_dir = curr_dir+"/input_output"
    os.chdir(new_dir)
    os.mkdir("input")
    os.mkdir("output")
    os.chdir(curr_dir)
    for file in os.listdir("."):
        if file.endswith(".jpg"):
            inputs, output = main_func(file)
            os.chdir(new_dir+"/input")
            inputs.save(file)
            os.chdir(new_dir+"/output")
            str_name = file[:-3]
            output.write(str_name + "xml")
            os.chdir(curr_dir)


if __name__ == '__main__':
        main()
