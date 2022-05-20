import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from scipy.ndimage import gaussian_filter
from skimage import filters, transform, io
from annoy import AnnoyIndex
from disjoint_set import disjoint_set
#findiing 10 neighbor of each pixel and adding them to similar set 
def _preprocessing(img, sigma=0.3):
    return gaussian_filter(img, sigma)

def segmentation(graph, rows, cols, k, graph_type='grid'):
    def sizet(size, k):
        return k/size
    def mindiff(INT_C1, size_C1, INT_C2, size_C2, k):
        return min(INT_C1 + sizet(size_C1, k), INT_C2 + sizet(size_C2, k))
    Segs_chs = []
    for ch in range(len(graph)):               
        seg = disjoint_set(rows*cols)
        for i in range(graph[ch].shape[0]):
            x = graph[ch][i][0]
            y = graph[ch][i][1]
            w = graph[ch][i][2]
            xp = seg.find(int(x))
            yp = seg.find(int(y))
            if xp != yp and w <= mindiff(seg.INT[xp], seg.arr[xp][2], seg.INT[yp], seg.arr[yp][2], k):
                seg.union(xp, yp)
                seg.update_INT(xp, yp, w)
        Segs_chs.append(seg)
    return Segs_chs
def draw_img(segs, rows, cols, graph_type='grid'):
    coloured_img = np.empty((rows, cols, 3), dtype=np.float64)
    con = None   
    if graph_type == 'grid':
        con = disjoint_set(rows * cols)     
        for i in range(rows):
            for j in range(cols-1):
                x = i * cols + j
                y = i * cols + j + 1
                if all([segs[ch].is_same_parent(x, y) for ch in range(len(segs))]):
                    con.union(x, y)
        for j in range(cols):
            for i in range(rows-1):
                x = i * cols + j
                y = (i+1) * cols + j
                if all([segs[ch].is_same_parent(x, y) for ch in range(len(segs))]):
                    con.union(x, y)
    elif graph_type == 'nn':
        con = segs[0]        
    else:
        print("Graph err")
    for i in range(rows):
        for j in range(cols):
            p = con.find(i*cols + j)
            np.random.seed(p)
            colour = np.random.randint(256, size=3)
            coloured_img[i][j] = colour
    return coloured_img/255
    
def generate_graph(img, graph_type='grid', d=10, n_tree=10, search_k=-1):    
    img = _preprocessing(img)    
    graphs = []
    rows = img.shape[0]
    cols = img.shape[1]
    num_vertices = rows * cols
    num_edges = (rows-1) * cols + (cols-1) * rows
    if graph_type == 'grid':
        for c in range(img.shape[2]):           
            edges = np.empty((num_edges, 3), dtype=np.float64)
            index = 0
            for i in range(rows):
                for j in range(cols):
                    if j < cols-1:
                        edges[index][0] = i*cols+j             
                        edges[index][1] = i*cols+j+1             
                        edges[index][2] = abs(img[i][j][c] - img[i][j+1][c])
                        index += 1  
                    if i < rows-1:
                        edges[index][0] = i*cols+j             
                        edges[index][1] = (i+1)*cols+j              
                        edges[index][2] = abs(img[i][j][c] - img[i+1][j][c])   
                        index += 1  
            edges = edges[edges[:,2].argsort()]
            graphs.append(edges)
    elif graph_type == 'nn':    
        f = 5
        t = AnnoyIndex(5, 'euclidean')
        nn_graph = []
        rows = img.shape[0]
        cols = img.shape[1]
        for i in range(rows):
            for j in range(cols):
                v = [img[i, j, 0], img[i, j, 1], img[i, j, 2] , i, j]
                t.add_item(i*cols+j, v)
        t.build(n_tree)
        for i in range(rows*cols):
            for neighbor in t.get_nns_by_item(i, d):
                if neighbor > i:
                    nn_graph.append([i, neighbor, t.get_distance(i, neighbor)])
                elif neighbor < i:
                    nn_graph.append([neighbor, i, t.get_distance(i, neighbor)])                    
        nn_graph = np.array(nn_graph)
        nn_graph = nn_graph[np.unique(nn_graph[:, :2], axis=0, return_index=True)[1]]
        graphs.append(nn_graph[nn_graph[:,2].argsort()])    
    else:
        print("cannot create graph")
    return graphs