import sys
import pandas as pd
import numpy as np
import symnmfmodule as sm
import symnmf as smpy

def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensionality")
    distance = 0.0
    for i in range(len(vector1)):
        distance += (vector1[i] - vector2[i]) ** 2
    
    return np.sqrt(distance)

def find_vec_index(data_mat, vec):
    k = len(vec)
    for index in range(0, len(data_mat), k):
        if data_mat[index:index+k] == vec:
            return index // k
    return -1

def compute_a(vec, data_mat ,clusters_mat, cols):
    index = find_vec_index(data_mat, vec)
    vec_cluster = clusters_mat[index]
    sum1 = 0
    i = 0
    clus_size = 0
    while(i<len(data_mat)):
        i_index = find_vec_index(data_mat, data_mat[i:i+cols])
        if (i_index != index and clusters_mat[i_index] == vec_cluster):
            clus_size += 1
            sum1 += euclidean_distance(data_mat[i:i+cols], vec)
        i += cols
    if clus_size != 0:
        return sum1/clus_size
    else:
        return 0

def mean_vec_clus_dist(vec, data_mat, clusters_mat, clus_index, cols):
    sum1 = 0 
    clus_size = 0
    i = 0
    while(i<len(data_mat)):
        if clusters_mat[find_vec_index(data_mat ,data_mat[i:i+cols])] == clus_index:
            sum1 += euclidean_distance(data_mat[i:i+cols], vec)
            clus_size += 1
        i += cols
    if clus_size != 0:
        return sum1/clus_size
    else:
        return 0

def compute_b(vec, data_mat ,clusters_mat, cols, k):
    clus_dist = []
    index = find_vec_index(data_mat, vec)
    vec_cluster = clusters_mat[index]
    for clus in range(k):
        if vec_cluster != clus:
            clus_dist.append(mean_vec_clus_dist(vec, data_mat, clusters_mat, clus, cols)) 
    return min(clus_dist)
    
    
def clusters_mat(data_mat, cols):
    result = []
    for i in range(0, len(data_mat), cols):
        vector = data_mat[i:i+cols]
        max_index = vector.index(max(vector))
        result.append(max_index)
    return result

def silhouette_coefficient(vec, data_mat, clusters_mat, k, cols):
    a = compute_a(vec, data_mat, clusters_mat, cols)
    b = compute_b(vec, data_mat ,clusters_mat, cols, k)
    coe = (b - a) / max(b,a)
    return coe

def silhouette_score(data_mat, clusters_mat, k, cols):
    i = 0
    sum1 = 0
    while(i<len(data_mat)):
        sum1 += silhouette_coefficient(data_mat[i:i+cols], data_mat, clusters_mat, k, cols)
        i += cols
    return sum1/(len(data_mat)//cols)

def main():
    np.random.seed(0)
    if (len(sys.argv) != 3):
        print("An Error Has Occurred")
        return 1
    else:
        k = int(sys.argv[1])
        filename = sys.argv[2]

    data_mat, vec_num, vec_size = smpy.read_matrix_from_file(filename)

    results = sm.norm(data_mat, vec_num, vec_size)
    norm_mat = results[0]
    normal_mat_avg = smpy.calculate_average(norm_mat)

    start_h = np.random.uniform(0, (np.sqrt(normal_mat_avg/k) * 2), (k*vec_num))
    final_h = sm.symnmf(list(start_h), vec_num, k, norm_mat, vec_num, vec_num, 0.5, 300, 0.0001)
    
    clusters_matrix = clusters_mat(final_h[0], final_h[1])

    smpy.print_matrix_with_precision(final_h[0], final_h[1])
    print("\n",clusters_matrix,"\n")

    sil_score = silhouette_score(data_mat, clusters_matrix, k, vec_size)
    print("nmf: ", round(sil_score, 4))
    return 0

if __name__ == "__main__":
    main()
    