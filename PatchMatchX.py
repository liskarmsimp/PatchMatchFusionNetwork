import numpy as np
from PIL import Image
import time
import os

def getValue(A_padding, p_size, y, x):#returns color, position, and variation
    p = p_size // 2
    variation = np.empty((3, 3, 3))
    for a in range(3):
        for b in range(3):
            variation[a, b] = A_padding[y, x] - A_padding[y - 1 + a, x - 1 + b]
    return A_padding[y - p:y + p + 1, x - p:x + p + 1, :], [y, x], variation

def cal_distance(a, b, A_padding, p_size, box):#calculates how different the patches are
    p = p_size // 2
    patch_a = A_padding[a[0] - p:a[0] + p + 1, a[1] - p:a[1] + p + 1, :].copy()
    patch_b = A_padding[b[0] - p:b[0] + p + 1, b[1] - p:b[1] + p + 1, :].copy()#maybe check math?
    temp = patch_a - patch_b
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    if np.isnan(dist):
        return 100000
    return dist

def cal_distance2(a, b, A_padding, p_size, box):#calculates how different the patches are
    p = p_size // 2
    patch_a = A_padding[a[0] - p:a[0] + p + 1, a[1] - p:a[1] + p + 1, :].copy()
    patch_b = A_padding[b[0] - p:b[0] + p + 1, b[1] - p:b[1] + p + 1, :].copy()#maybe check math?
    onEdge = False
    for y in range(a[0] - p, a[0] + p + 1):
        for x in range(a[1] - p, a[1] + p + 1):
            if (y, x) not in box:
                onEdge = True
                break
        if onEdge:
            break
    if onEdge:
        for y in range(a[0] - p, a[0] + p + 1):
            for x in range(a[1] - p, a[1] + p + 1):
                if (y, x) in box:
                    patch_a[y - (a[0] - p), x - (a[1] - p)] = np.array([np.nan, np.nan, np.nan])
    #print(onEdge)
    #Image.fromarray(np.array(patch_a).astype(np.uint8), mode = "RGB").show()
    #Image.fromarray(np.array(patch_b).astype(np.uint8), mode = "RGB").show()
    # print(patch_a)
    # print(patch_b)
    temp = patch_a - patch_b
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    if np.isnan(dist):
        return 100000
    #print(temp)
    #print(num)
    #print(dist)
    # temp = A_padding.copy()
    # temp[a[0], a[1]] = [255, 255, 255]
    # Image.fromarray(np.array(temp).astype(np.uint8), mode = "RGB").show()
    return dist

def resizeHalf(A_padding, p_size):#makes every other pixel the average of it, right, down, and down right
    h = np.size(A_padding, 0)
    w = np.size(A_padding, 1)
    p = p_size // 2
    temp = np.zeros([h // 2 + p, w // 2 + p, 3])
    for x in range(0, h - 2 * p, 2):
        for y in range(0, w - 2 * p, 2):
            sum0 = np.zeros([2, 2])
            sum1 = np.zeros([2, 2])
            sum2 = np.zeros([2, 2])
            nan = False
            for a in range(0, 2):
                for b in range(0, 2):
                    if np.isnan(A_padding[x + a + p, y + b + p, 0]) or np.isnan(A_padding[x + a + p, y + b + p, 1]) or np.isnan(A_padding[x + a + p, y + b + p, 2]):
                        nan = True
                        break
                    sum0[a, b] = A_padding[x + a + p, y + b + p][0]
                    sum1[a, b] = A_padding[x + a + p, y + b + p][1]
                    sum2[a, b] = A_padding[x + a + p, y + b + p][2]
                if nan:
                    break
            if not nan:
                temp[x // 2 + p, y // 2 + p] = np.array([np.average(sum0), np.average(sum1), np.average(sum2)])
            else:
                temp[x // 2 + p, y // 2 + p] = np.array([np.nan, np.nan, np.nan])
    #Image.fromarray(np.array(temp).astype(np.uint8), mode = "RGB").show()
    return temp

def scaleUp(a, b, dist, dist2, p_size, A_padding, box, f):#a is smaller f but algorithmed, b is larger but random, scale a onto b
    p = p_size // 2
    A_h = np.size(a, 0) - 2 * p#5, 8, 14, 26, 50
    A_w = np.size(a, 1) - 2 * p#3, 6, 12, 24, 48
    B_h = np.size(b, 0) - 2 * p
    B_w = np.size(b, 1) - 2 * p
    for y in range(B_h):#go through indices of b
        for x in range(B_w):
            aValue = np.array([a[y // 2 + p, x // 2 + p][0], a[y // 2 + p, x // 2 + p][1]])
            b[y + p, x + p] = np.array([(aValue[0] - p) * 2 + p, aValue[1] * 2 - p])
    for y in range(B_h):
        for x in range(B_w):
            aValue = np.array([a[y // 2 + p, x // 2 + p][0], a[y // 2 + p, x // 2 + p][1]])
            if y >= p and x >= p and B_h - y >= p and B_w - x >= p:
                dist2[y + p, x + p] = 100000
            else:
                dist2[y + p, x + p] = cal_distance2(aValue, b[y + p, x + p], A_padding, p_size, box)


def reconstruction(f, A, p_size, x, y):#makes and display the image
    p = p_size // 2
    B_h = np.size(f, 0) - 2 * p
    B_w = np.size(f, 1) - 2 * p
    for i in range(B_h):
        for j in range(B_w):
            A[i + y, j + x, :] = A[f[i + p, j + p][0] - p, f[i + p, j + p][1] - p, :]
    Image.fromarray(np.array(A).astype(np.uint8), mode = "RGB").show()

def reconstruction2(f, A, p_size, x, y):#makes and displays the image with blending
    B_h = np.size(f, 0)
    B_w = np.size(f, 1)
    p = p_size // 2
    for a in range(p, B_h - p):#1 -> 48 (not inclusive)
        for b in range(p, B_w - p):
            temp0 = np.ones([3, 3]) * np.nan#find average
            temp1 = np.ones([3, 3]) * np.nan
            temp2 = np.ones([3, 3]) * np.nan
            ymin = xmin = 0
            ymax = xmax = 3
            if a == p:
                ymin = 1
            if a + p == B_h:
                ymax = 2
            if b == p:
                xmin = 1
            if b + p == B_w:
                xmax = 2
            for m in range(ymin, ymax):#0, 1, 2
                for n in range(xmin, xmax):
                    temp0[m, n] = A[f[a - 1 + m, b - 1 + n][0], f[a - 1 + m, b - 1 + n][1], 0]
                    temp1[m, n] = A[f[a - 1 + m, b - 1 + n][0], f[a - 1 + m, b - 1 + n][1], 1]
                    temp2[m, n] = A[f[a - 1 + m, b - 1 + n][0], f[a - 1 + m, b - 1 + n][1], 2]
            A[a + y - p, b + x - p, :] = np.array([np.nansum(temp0) / ((xmax - xmin) * (ymax - ymin)), np.nansum(temp1) / ((xmax - xmin) * (ymax - ymin)), np.nansum(temp2) / ((xmax - xmin) * (ymax - ymin))])
    Image.fromarray(np.array(A).astype(np.uint8), mode = "RGB").show()

def reconstruction3(f, A, p_size, x, y):#makes and displays the image with blending
    B_h = np.size(f, 0)
    B_w = np.size(f, 1)
    p = p_size // 2
    for a in range(3):#1 -> 48 (not inclusive)
        for b in range(3):
            temp0 = np.ones([3, 3]) * np.nan#find average
            temp1 = np.ones([3, 3]) * np.nan
            temp2 = np.ones([3, 3]) * np.nan
            ymin = xmin = 0
            ymax = xmax = 3
            if a == p:
                ymin = 1
            if a + p == B_h:
                ymax = 2
            if b == p:
                xmin = 1
            if b + p == B_w:
                xmax = 2
            for m in range(ymin, ymax):#0, 1, 2
                for n in range(xmin, xmax):
                    if m == 1 and n == 1:
                        temp0[m, n] = A[f[a - 1 + m, b - 1 + n][0], f[a - 1 + m, b - 1 + n][1], 0]
                        temp1[m, n] = A[f[a - 1 + m, b - 1 + n][0], f[a - 1 + m, b - 1 + n][1], 1]
                        temp2[m, n] = A[f[a - 1 + m, b - 1 + n][0], f[a - 1 + m, b - 1 + n][1], 2]
            A[a + y - p, b + x - p, :] = np.array([(np.nansum(temp0) / ((xmax - xmin) * (ymax - ymin) - 1) + A[f[a, b][0], f[a][b][1], 0]) / 2, (np.nansum(temp1) / ((xmax - xmin) * (ymax - ymin) - 1) + A[f[a, b][0], f[a][b][1], 1]) / 2, (np.nansum(temp2) / ((xmax - xmin) * (ymax - ymin) - 1) + A[f[a, b][0], f[a][b][1], 2]) / 2])
    Image.fromarray(np.array(A).astype(np.uint8), mode = "RGB").show()

def fReconstruction(f, A):
    temp = np.zeros([50, 50, 3])
    for a in range(50):
        for b in range(50):
            temp[a, b, :] = A[f[a,b][0], f[a,b][1]]
    Image.fromarray(np.array(temp).astype(np.uint8), mode = "RGB").show()

def blackRectangle(A, x, y, w, h):#makes black rectangle at position
    for a in range(h):
        for b in range(w):
            A[y + a, x + b] = np.array([0, 0, 0]) * np.nan
    return A
    
def initialization(A, p_size):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    B_w = B_h = 0
    x = y = False
    for h in range(A_h):#find black rectangle
        for w in range(A_w):
            if sum(A[h, w]) == 0 or np.isnan(A[h, w, 0]):
                x = w
                y = h
                break
        if x:
            break
    while x < A_w and (sum(A[y, x]) == 0 or np.isnan(A[y, x, 0])):
        B_w += 1
        x += 1
    x -= 1
    while y < A_h and (sum(A[y, x]) == 0 or np.isnan(A[y, x, 0])):
        B_h += 1
        y += 1
    x -= B_w - 1
    y -= B_h
    p = p_size // 2
    random_B_r = np.random.randint(p, A_h + p, [B_h + p * 2, B_w + p * 2])#assigns a random pixel from the whole image to a x/y arrays of black box size
    random_B_c = np.random.randint(p, A_w + p, [B_h + p * 2, B_w + p * 2])
    box = set()
    for a in range(B_h):#make sure f doesn't use any of the nian pixels
        for b in range(B_w):
            while random_B_r[a, b] > y and random_B_r[a, b] < y + B_h and random_B_c[a, b] > x and random_B_c[a, b] < x + B_h:
                random_B_r[a, b] = np.random.randint(p, A_h + p - 1)
                random_B_c[a, b] = np.random.randint(p, A_w + p - 1)
            box.add((a + y + p, b + x + p))
    A_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
    A_padding[p:A_h+p, p:A_w+p, :] = A#A_padding is A surrounded by a bunch of NaN's?    
    f = np.zeros([B_h + p * 2, B_w + p * 2], dtype=object)
    dist = np.zeros([B_h + p * 2, B_w + p * 2])#dist will have an extra p box around it to accomodate for prop
    for i in range(B_h + p * 2):
        for j in range(B_w + p * 2):
            a = np.array([i + y, j + x])
            if i >= p and j >= p and i < B_h + p and j < B_w + p:
                A_padding[i + y, j + x] = np.array([np.nan, np.nan, np.nan])
            b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)
            f[i, j] = b#assigns the random thing to f
            if i < p or j < p or i >= B_h + p or j >= B_w + p:
                dist[i, j] = 100000
            else:
                dist[i, j] = cal_distance(a, b, A_padding, p_size, box)
    return f, dist, A_padding, B_h, B_w, x, y, box#f matches a pixel on A with its random B pixel, dist is presumably the distance, A padding is A with padding around it

def propagation(f, a, xpos, ypos, dist, A_padding, p_size, is_odd, box):
    A_h = np.size(A_padding, 0) - p_size
    A_w = np.size(A_padding, 1) - p_size
    x = a[0]
    y = a[1]
    a = (x + ypos, y + xpos)
    p = p_size // 2
    if is_odd:
        d_current = dist[x, y]#distance of current pixel
        d_up = dist[max(x-1, p), y]#distance of pixel on top?
        d_left = dist[x, max(y-1, p)]#distance of pixel on left?
        idx = np.argmin(np.array([d_current, d_up, d_left]))#idx = index of smolest distance
        if idx == 1 and (max(x-1, p), y) not in box:
            dist2 = cal_distance2([max(x-1, p), y], np.array([max(f[max(x - 1, p), y][0], p), f[max(x - 1, p), y][1]]), A_padding, p_size, box)
            if dist2 < d_current:
                f[x, y] = np.array([max(f[max(x - 1, p), y][0], p), f[max(x - 1, p), y][1]])#if left has smaller distance, change current to left
                dist[x, y] = dist2
        if idx == 2 and (x, max(y-1, p)) not in box:
            dist2 = cal_distance2([x, max(y-1, p)], np.array([f[x, max(y - 1, p)][0], max(f[x, max(y - 1, p)][1], p)]), A_padding, p_size, box)
            if dist2 < d_current:
                f[x, y] = np.array([f[x, max(y - 1, p)][0], max(f[x, max(y - 1, p)][1], p)])#same but for top.
                dist[x, y] = dist2
    else:
        d_current = dist[x, y]
        d_down = dist[min(x + 1, A_h - p - 2), y]#dist of pixel beneath
        d_right = dist[x, min(y + 1, A_w - p)]#dist of pixel on right
        idx = np.argmin(np.array([d_current, d_down, d_right]))
        if idx == 1 and (min(x + 1, A_h - p), y) not in box:
            dist2 = cal_distance2([min(x + 1, A_h - p - 2), y], np.array([min(f[min(x + 1, A_h - p), y][0], A_h - p - 2), f[min(x + 1, A_h - p - 2), y][1]]), A_padding, p_size, box)
            if dist2 < d_current:
                f[x, y] = np.array([min(f[min(x + 1, A_h - p), y][0], A_h - p - 2), f[min(x + 1, A_h - p - 2), y][1]])
                dist[x, y] = dist2
        if idx == 2 and (x, min(y + 1, A_w - p - 2)) not in box:
            dist2 = cal_distance2([x, min(y + 1, A_w - p)], np.array([f[x, min(y + 1, A_w - p - 2)][0], min(f[x, min(y + 1, A_w - p - 2)][1], A_w - p - 2)]), A_padding, p_size, box)
            if dist2 < d_current:
                f[x, y] = np.array([f[x, min(y + 1, A_w - p - 2)][0], min(f[x, min(y + 1, A_w - p - 2)][1], A_w - p - 2)])
                dist[x, y] = dist2

def random_search(f, a, xpos, ypos, dist, A_padding, p_size, box, alpha=0.5):
    x = a[0]
    y = a[1]
    a = (x + ypos, y + xpos)
    B_h = np.size(A_padding, 0)
    B_w = np.size(A_padding, 1)
    p = p_size // 2
    i = 1
    search_h = B_h * alpha ** i#restricting size of search(?)
    search_w = B_w * alpha ** i
    b_x = f[x, y][0]
    b_y = f[x, y][1]
    while search_h > 1 and search_w > 1:
        search_min_r = max(b_x - search_h, 2 * p)
        search_max_r = min(b_x + search_h - 1, B_h - p)
        random_b_x = np.random.randint(search_min_r, max(search_max_r, search_min_r + 1))
        search_min_c = max(b_y - search_w, 2 * p)
        search_max_c = min(b_y + search_w - 1, B_w - p)
        random_b_y = np.random.randint(search_min_c, max(search_max_c, search_min_c + 1))#choose a random pixel in the search area
        search_h = B_h * alpha ** i
        search_w = B_w * alpha ** i
        b = np.array([random_b_x - p, random_b_y - p])
        d = cal_distance2(a, b, A_padding, p_size, box)
        if d < dist[x, y] and (random_b_x - p, random_b_y - p) not in box:#if the random pixel is better, change to that one
            dist[x, y] = d
            f[x, y] = b
        i += 1#reduce next search area

def NNS(img, p_size, itr):
    f, dist, img_padding, B_h, B_w, x, y, box = initialization(img, p_size)#B_w and B_h are only 48x48; do not take into account padding from initialization
    p = p_size // 2
    for itr in range(1, itr+1):#alternate between these 2 things
        reconstruction(f, img, p_size, x, y)
        if itr % 2 == 0:#go right to left, bottom to top, propagate based on right and bottom
            for i in range(B_h - 1 + p, -1 + p, -1):
                for j in range(B_w - 1 + p, -1 + p, -1):
                    a = np.array([i, j])
                    propagation(f, a, x, y, dist, img_padding, p_size, False, box)
                    random_search(f, a, x, y, dist, img_padding, p_size, box)
        else:#go left to right, top to bottom, propagate based on left and top
            for i in range(p, B_h + p):
                for j in range(p, B_w + p):
                    a = np.array([i, j])
                    propagation(f, a, x, y, dist, img_padding, p_size, True, box)
                    random_search(f, a, x, y, dist, img_padding, p_size, box)
        print("iteration: %d"%(itr))
    return f, x, y

def MSNNS(img, p_size, itr):#multiscale
    start = time.time()
    f = [0, 0, 0, 0, 0, 0, 0, 0]
    dist = [0, 0, 0, 0, 0, 0, 0, 0]
    img_padding = [0, 0, 0, 0, 0, 0, 0, 0]
    B_h = [0, 0, 0, 0, 0, 0, 0, 0]
    B_w = [0, 0, 0, 0, 0, 0, 0, 0]
    x = [0, 0, 0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0, 0, 0]
    box = [0, 0, 0, 0, 0, 0, 0, 0]
    f[0], dist[0], img_padding[0], B_h[0], B_w[0], x[0], y[0], box[0] = initialization(img, p_size)#B_w and B_h are only 48x48; do not take into account padding from initialization
    p = p_size // 2
    ret = img_padding[0]        
    for k in range(1, 7):
        ret = resizeHalf(ret, p_size)        
        ret = ret[p:np.size(ret, 0) - p, p:np.size(ret, 1) - p, :]
        f[k], dist[k], img_padding[k], B_h[k], B_w[k], x[k], y[k], box[k] = initialization(ret, p_size)
        ret = img_padding[k]
    end = time.time()
    print("Initialization: " + str(end - start))
    start = time.time()
    for k in range(6, -1, -1):#MAKE SMALLER Fs IMPACT NEXT LARGEST F
        if k == 2:
            end = time.time()
            print("To gen 2: " + str(end - start))
            start = time.time()
        for itr in range(1, itr+1):#alternate between these 2 things
            print("current iteration: " + str((k, itr)))
            # reconstruction(f[k], img_padding[k][p:np.size(img_padding[k], 0) - p, p:np.size(img_padding[k], 1) - p, :], p_size, x[k], y[k])
            if itr % 2 == 0:#go right to left, bottom to top, propagate based on right and bottom
                for i in range(B_h[k] + p - 1, p - 1, -1):
                    for j in range(B_w[k] + p - 1, p - 1, -1):
                        a = np.array([i, j])
                        dist[k][i][j] = cal_distance2([a[0] + y[k], a[1] + x[k]], f[k][a[0]][a[1]], img_padding[k], p_size, box[k])
                        propagation(f[k], a, x[k], y[k], dist[k], img_padding[k], p_size, False, box[k])
                        random_search(f[k], a, x[k], y[k], dist[k], img_padding[k], p_size, box[k])
            else:#go left to right, top to bottom, propagate based on left and top
                for i in range(p, B_h[k] + p):
                    for j in range(p, B_w[k] + p):
                        a = np.array([i, j])
                        dist[k][i][j] = cal_distance2([a[0] + y[k], a[1] + x[k]], f[k][a[0]][a[1]], img_padding[k], p_size, box[k])
                        propagation(f[k], a, x[k], y[k], dist[k], img_padding[k], p_size, True, box[k])
                        random_search(f[k], a, x[k], y[k], dist[k], img_padding[k], p_size, box[k])
        if k > 0:
            scaleUp(f[k], f[k - 1], dist[k], dist[k - 1], p_size, img_padding[k - 1], box[k - 1], f[k])
    end = time.time()
    print("The rest: " + str(end - start))
    reconstruction(f[0], img_padding[0][p:np.size(img_padding[0], 0) - p, p:np.size(img_padding[0], 1) - p, :], p_size, x[0], y[0])
    return f, x, y

def makeMask(box):
    image = np.array(Image.new(mode="RGB", size=(256, 256), color = "white"))
    for a in box:
        image[a[0], a[1]] = (0, 0, 0)
    Image.fromarray(image).save('./samples/test/mask/mask.png')

def bruteNNS(a, f, dist, w, h, p_size, box, x2, y2):
    p = p_size // 2
    a_h = np.size(a, 0)
    a_w = np.size(a, 1)
    for y in range(p, h - p):
        for x in range(p, w - p):
            f[y, x] = [0, 0]
            dist[y, x] = 100000
            for i in range(p, a_h - p, 20):
                for j in range(p, a_w - p, 20):
                    if (i + y2, j + x2) not in box:
                        d = cal_distance2([i, j], [y + y2, x + x2], a, p_size, box)
                        if d < dist[y, x]:
                            dist[y, x] = d
                            f[y, x] = [i, j]
    return a, f, dist

def MSNNNNS(img, p_size, itr):#multiscale neural network nearest neighbor search
    start = time.time()
    f = [0, 0, 0]
    dist = [0, 0, 0]
    img_padding = [0, 0, 0]
    B_h = [0, 0, 0]
    B_w = [0, 0, 0]
    x = [0, 0, 0]
    y = [0, 0, 0]
    box = [0, 0, 0]
    f[0], dist[0], img_padding[0], B_h[0], B_w[0], x[0], y[0], box[0] = initialization(img, p_size)#B_w and B_h are only 48x48; do not take into account padding from initialization
    p = p_size // 2
    ret = img_padding[0]        
    for k in range(1, 3):
        ret = resizeHalf(ret, p_size)        
        ret = ret[p:np.size(ret, 0) - p, p:np.size(ret, 1) - p, :]
        f[k], dist[k], img_padding[k], B_h[k], B_w[k], x[k], y[k], box[k] = initialization(ret, p_size)
        ret = img_padding[k]
    end = time.time()
    print("Initialization: " + str(end - start))
    start = time.time()
    makeMask(box[2])
    os.system('python test.py --model model/model_places2.pth --img samples/test/img --mask samples/test/mask --output output/test --merge')
    image = np.array(Image.open('./output/test/result/result-place-mask.png'))
    image = resizeHalf(image, 0)
    img_padding[2][p:p + np.size(image, 0), p:p + np.size(image, 1), :] = image
    dist[2][p:p + np.size(image, 0), p:p + np.size(image, 1)] = 100000
    img_padding[2], f[2], dist[2] = bruteNNS(img_padding[2], f[2], dist[2], B_h[2], B_w[2], p_size, box[2], x[2], y[2])
    # os.remove('./samples/test/img/place.png')
    # os.remove('./samples/test/mask/mask.png')
    # os.remove('./output/test/result/result-place-mask.png')
    end = time.time()
    print("NN: " + str(end - start))
    start = time.time()
    for k in range(2, -1, -1):#MAKE SMALLER Fs IMPACT NEXT LARGEST F
        for itr in range(1, itr+1):#alternate between these 2 things
            print("current iteration: " + str((k, itr)))
            # reconstruction(f[k], img_padding[k][p:np.size(img_padding[k], 0) - p, p:np.size(img_padding[k], 1) - p, :], p_size, x[k], y[k])
            if itr % 2 == 0:#go right to left, bottom to top, propagate based on right and bottom
                for i in range(B_h[k] + p - 1, p - 1, -1):
                    for j in range(B_w[k] + p - 1, p - 1, -1):
                        a = np.array([i, j])
                        dist[k][a[0]][a[1]] = cal_distance2([a[0] + y[k], a[1] + x[k]], f[k][a[0]][a[1]], img_padding[k], p_size, box[k])
                        propagation(f[k], a, x[k], y[k], dist[k], img_padding[k], p_size, False, box[k])
                        random_search(f[k], a, x[k], y[k], dist[k], img_padding[k], p_size, box[k])
            else:#go left to right, top to bottom, propagate based on left and top
                for i in range(p, B_h[k] + p):
                    for j in range(p, B_w[k] + p):
                        a = np.array([i, j])
                        dist[k][a[0]][a[1]] = cal_distance2([a[0] + y[k], a[1] + x[k]], f[k][a[0]][a[1]], img_padding[k], p_size, box[k])
                        propagation(f[k], a, x[k], y[k], dist[k], img_padding[k], p_size, True, box[k])
                        random_search(f[k], a, x[k], y[k], dist[k], img_padding[k], p_size, box[k])
        if k > 0:
            scaleUp(f[k], f[k - 1], dist[k], dist[k - 1], p_size, img_padding[k - 1], box[k - 1], f[k])
    end = time.time()
    print("The rest: " + str(end - start))
    reconstruction(f[0], img_padding[0][p:np.size(img_padding[0], 0) - p, p:np.size(img_padding[0], 1) - p, :], p_size, x[0], y[0])
    return f, x, y

if __name__ == "__main__":
    img = blackRectangle(np.array(Image.open("./samples/test/place3.jpg")), 860, 850, 64, 64)
    temp = np.array(Image.open("./samples/test/place3.jpg"))
    temp = resizeHalf(temp, 0)
    temp = resizeHalf(temp, 0)
    Image.fromarray(np.array(temp).astype(np.uint8), mode = "RGB").save('./samples/test/img/place.png')
    p_size = 3#patch size(?)
    itr = 10
    start = time.time()
    # f, x, y = NNS(img, p_size, itr)
    # f, x, y = MSNNS(img, p_size, itr)
    f, x, y = MSNNNNS(img, p_size, itr)
    end = time.time()
    print(end - start)