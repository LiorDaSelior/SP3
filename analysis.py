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

def find_vec_index(matrix, target_vector):
    k = len(target_vector)
    for index in range(0, len(matrix), k):
        if matrix[index:index+k] == target_vector:
            return index // k
    return -1

def compute_a(vec, data_mat ,clusters_mat, cols):
    index = find_vec_index(data_mat, vec)
    vec_cluster = clusters_mat[index]
    sum1 = 0
    a = 0
    clus_size = 0
    for x in clusters_mat:
        x_index = find_vec_index(data_mat ,data_mat[clusters_mat.index(x):clusters_mat.index(x)+cols])
        x_cluster = clusters_mat[x_index]
        if (x_index != index and x_cluster == vec_cluster):
            clus_size += 1
            sum1 += euclidean_distance(data_mat[clusters_mat.index(x)*cols:clusters_mat.index(x)+cols], vec)
    if clus_size != 0:
        a = sum1/clus_size
    else:
        a = 0
    return a

def mean_vec_clus_dist(vec, clusters_mat, clus_index, data_mat):
    sum1 = 0 
    clus_size = 0
    for x in clusters_mat:
        if clus_index == x:
            sum1 += euclidean_distance(data_mat[clusters_mat.index(x)*len(vec):clusters_mat.index(x)+len(vec)], vec)
            clus_size += 1
    return sum1/clus_size

def compute_b(vec, data_mat ,clusters_mat, cols, k):
    clus_dist = []
    index = find_vec_index(data_mat, vec)
    vec_cluster = clusters_mat[index]
    for clus in range(k):
        if vec_cluster != clus:
            clus_dist.append(mean_vec_clus_dist(vec, clusters_mat, clus, data_mat)) 
    return min(clus_dist)
    
    
def clusters_mat(matrix, num_columns):
    result = []
    for i in range(0, len(matrix), num_columns):
        vector = matrix[i:i+num_columns]
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
    if (len(sys.argv) != 3):
        print("An Error Has Occurred")
        return 1
    else:
        k = int(sys.argv[1])
        filename = sys.argv[2]

    matrix, vec_num, vec_size = smpy.read_matrix_from_file(filename)

    results = sm.norm(matrix, vec_num, vec_size)
    norm_mat = results[0]
    normal_mat_avg = smpy.calculate_average(norm_mat)

    start_h = np.random.uniform(0, (np.sqrt(normal_mat_avg/k) * 2), (k*vec_num))

    final_h = sm.symnmf(list(start_h), vec_num, k, norm_mat, vec_num, vec_num, 0.5, 300, 0.0001)
    clusters_matrix = clusters_mat(final_h[0], final_h[1])

    sil_score = silhouette_score(matrix, clusters_matrix, k, vec_size)
    
    print(sil_score)
    return 0

if __name__ == "__main__":
    main()
    