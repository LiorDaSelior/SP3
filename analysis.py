import sys
import numpy as np
import symnmf as smpy
import kmeans as km
from sklearn.metrics import silhouette_score


def clusters_mat(h_mat, cols):
    result = []
    for i in range(0, len(h_mat), cols):
        vector = h_mat[i:i+cols]
        max_index = vector.index(max(vector))
        result.append(max_index)
    return result

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
    
    sublists = [data_mat[x:x+vec_size] for x in range(0, len(data_mat), vec_size)]
    matrix = np.array(sublists)
    
    try:
        nmf_sil_score = silhouette_score(matrix, clusters_matrix)
        print("nmf:", round(nmf_sil_score, 4))
    except ValueError as e:
        print("An Error Has Occurred")
        return 1
    
    centroid_list = km.Centroid.create_k_len_centroid_list(sublists, k)
    clusters_matrix = km.algo(sublists, centroid_list, 300)
    
    try:
        kmenas_sil_score = silhouette_score(matrix, clusters_matrix)
        print("kmeans:", round(kmenas_sil_score, 4))
    except ValueError as e:
        print("An Error Has Occurred")
        return 1
    
    return 0

if __name__ == "__main__":
    main()