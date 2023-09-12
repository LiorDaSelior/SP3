import sys
import pandas as pd
import numpy as np
import symnmf as smpy
import kmeans as km
from sklearn.metrics import silhouette_score

def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensionality")
    distance = 0.0
    for i in range(len(vector1)):
        distance += (vector1[i] - vector2[i]) ** 2
    
    return np.sqrt(distance)

def compute_a(index, vec, data_mat ,clusters_mat, cols):
    vec_cluster = clusters_mat[index]
    sum1 = 0
    i = 0
    j = 0
    clus_size = 0
    while(i<len(data_mat)):
        if (index != j and clusters_mat[j] == vec_cluster):
            clus_size += 1
            sum1 += euclidean_distance(data_mat[i:i+cols], vec)
        i += cols
        j += 1
    if clus_size != 0:
        return sum1/clus_size
    else:
        return 0

def mean_vec_clus_dist(vec, data_mat, clusters_mat, clus_index, cols):
    sum1 = 0 
    clus_size = 0
    i = 0
    j = 0
    while(i<len(data_mat)):
        if clusters_mat[j] == clus_index:
            sum1 += euclidean_distance(data_mat[i:i+cols], vec)
            clus_size += 1
        i += cols
        j += 1
    if clus_size != 0:
        return sum1/clus_size
    else:
        return 0

def compute_b(index, vec, data_mat ,clusters_mat, cols, k):
    clus_dist = []
    vec_cluster = clusters_mat[index]
    for clus in range(k):
        if vec_cluster != clus:
            clus_dist.append(mean_vec_clus_dist(vec, data_mat, clusters_mat, clus, cols)) 
    return min(clus_dist)
        
def clusters_mat(h_mat, cols):
    result = []
    for i in range(0, len(h_mat), cols):
        vector = h_mat[i:i+cols]
        max_index = vector.index(max(vector))
        result.append(max_index)
    return result

def silhouette_coefficient(i, vec, data_mat, clusters_mat, k, cols):
    a = compute_a(i, vec, data_mat, clusters_mat, cols)
    b = compute_b(i, vec, data_mat ,clusters_mat, cols, k)
    coe = (b - a) / max(b,a)
    return coe

def silhouette_score_custom(data_mat, clusters_mat, k, cols):
    i = 0
    j = 0
    sum1 = 0
    while(i<len(data_mat)):
        sum1 += silhouette_coefficient(j, data_mat[i:i+cols], data_mat, clusters_mat, k, cols)
        i += cols
        j += 1
    return sum1/(len(data_mat)//cols)

def main():
    np.random.seed(0)
    if (len(sys.argv) != 3):
        print("An Error Has Occurred")
        return 1
    else:
        k = int(sys.argv[1])
        filename = sys.argv[2]

    temp = smpy.read_matrix_from_file(filename)
    if temp is None:
        print("An Error Has Occurred")
        return 1
    data_mat, vec_num, vec_size = smpy.read_matrix_from_file(filename)

    results = smpy.sm.norm(data_mat, vec_num, vec_size)
    norm_mat = results[0]
    normal_mat_avg = smpy.calculate_average(norm_mat)

    start_h = np.random.uniform(0, (np.sqrt(normal_mat_avg/k) * 2), (k*vec_num))
    final_h = smpy.sm.symnmf(list(start_h), vec_num, k, norm_mat, vec_num, vec_num, 0.5, 300, 0.0001)
    
    clusters_matrix = clusters_mat(final_h[0], final_h[1])

    nmf_sil_score = silhouette_score_custom(data_mat, clusters_matrix, k, vec_size)
    print("nmf:", round(nmf_sil_score, 4))
    
    vector_list = km.file_to_vector_list(filename)
    data_mat = km.vector_list_to_vector_data(vector_list)
    centroid_list = km.Centroid.create_k_len_centroid_list(vector_list, k)
    clusters_matrix = km.algo(vector_list, centroid_list, 300)
    
    kmenas_sil_score = silhouette_score_custom(data_mat, clusters_matrix, k, vec_size)
    print("kmeans:", round(kmenas_sil_score, 4))
    return 0

def main_sklearn():
    np.random.seed(0)
    if (len(sys.argv) != 3):
        print("An Error Has Occurred")
        return 1
    else:
        k = int(sys.argv[1])
        filename = sys.argv[2]

    temp = smpy.read_matrix_from_file(filename)
    if temp is None:
        print("An Error Has Occurred")
        return 1
    data_mat, vec_num, vec_size = smpy.read_matrix_from_file(filename)

    results = smpy.sm.norm(data_mat, vec_num, vec_size)
    norm_mat = results[0]
    normal_mat_avg = smpy.calculate_average(norm_mat)

    start_h = np.random.uniform(0, (np.sqrt(normal_mat_avg/k) * 2), (k*vec_num))
    final_h = smpy.sm.symnmf(list(start_h), vec_num, k, norm_mat, vec_num, vec_num, 0.5, 300, 0.0001)
    
    clusters_matrix = clusters_mat(final_h[0], final_h[1])
    
    sublists = [data_mat[x:x+vec_size] for x in range(0, len(data_mat), vec_size)]
    matrix = np.array(sublists)
    
    nmf_sil_score = silhouette_score(matrix, clusters_matrix)
    print("sk nmf:", round(nmf_sil_score, 4))
    
    centroid_list = km.Centroid.create_k_len_centroid_list(sublists, k)
    clusters_matrix = km.algo(sublists, centroid_list, 300)
    
    kmenas_sil_score = silhouette_score(matrix, clusters_matrix)
    print("sk kmeans:", round(kmenas_sil_score, 4))
    
    return 0

if __name__ == "__main__":
    main()
    main_sklearn()
    