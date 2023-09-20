import sys
import numpy as np
import symnmfmodule as sm

def read_matrix_from_file(file_name):
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
        matrix_entries = []
        for line in lines:
            entries = [float(x.strip()) for x in line.strip().split(',')]
            matrix_entries.extend(list(entries))
        num_rows = len(lines)
        if num_rows > 0:
            num_columns = len(matrix_entries) // num_rows
        else:
            num_columns = 0
        return matrix_entries, num_rows, num_columns

    except Exception as e:
        return None


def print_matrix_with_precision(matrix_list, num_columns):
    for i in range(0, len(matrix_list), num_columns):
        row = matrix_list[i:i + num_columns]
        formatted_row = ["{:.4f}".format(x) for x in row]
        row_str = ','.join(formatted_row)
        print(row_str)


def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    total = sum(numbers)
    average = total / len(numbers)
    return average


def main():
    np.random.seed(0)
    eps = 0.0001
    beta = 0.5
    max_iter = 300

    if (len(sys.argv) != 4):
        print("An Error Has Occurred")
        return 1
    else:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        filename = sys.argv[3]
    
    temp = read_matrix_from_file(filename)
    if temp is None:
        print("An Error Has Occurred")
        return 1
    
    matrix, vec_num, vec_size = read_matrix_from_file(filename)
    
    if (k  <= 1 or k >= vec_num):
        print("An Error Has Occurred")
        return 1

    if (goal == "sym"):
        results = sm.sym(vec_num, vec_size, matrix)
        print_matrix_with_precision(results[0], results[1])

    elif (goal == "ddg"):
        results = sm.ddg(matrix, vec_num, vec_size)
        print_matrix_with_precision(results[0], results[1])

    elif (goal == "norm"):
        results = sm.norm(matrix, vec_num, vec_size)
        print_matrix_with_precision(results[0], results[1])

    elif (goal == "symnmf"):
        results = sm.norm(matrix, vec_num, vec_size)
        norm_mat = results[0]

        normal_mat_avg = calculate_average(norm_mat)

        h = np.random.uniform(0, (np.sqrt(normal_mat_avg/k) * 2), (k*vec_num))

        resultsB = sm.symnmf(list(h), vec_num, k, norm_mat, vec_num, vec_num, beta, max_iter, eps)
        print_matrix_with_precision(resultsB[0], resultsB[1])
    
    else:
        print("An Error Has Occurred")
        return 1

    return 0


if __name__ == "__main__":
    main()