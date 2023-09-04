import sys
import pandas as pd
import numpy as np
import symnmfmodule as sm


def read_matrix_from_file(file_name):
    try:
        # Read the file and split lines
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # Initialize an empty list to store matrix entries
        matrix_entries = []

        # Iterate through the lines, split on commas, and convert to floats
        for line in lines:
            entries = [float(x.strip()) for x in line.strip().split(',')]
            matrix_entries.extend(list(entries))

        # Calculate the number of rows and columns
        num_rows = len(lines)
        if num_rows > 0:
            num_columns = len(matrix_entries) // num_rows
        else:
            num_columns = 0

        # Return a tuple containing matrix entries, number of rows, and number of columns
        return matrix_entries, num_rows, num_columns

    except Exception as e:
        print("An Error Has Occurred")
        return 1


def print_matrix_with_precision(matrix_list, num_columns):
    for i in range(0, len(matrix_list), num_columns):
        row = matrix_list[i:i + num_columns]
        formatted_row = [f'{x:.4f}' for x in row]
        row_str = ','.join(formatted_row)
        print(row_str)

def calculate_average(numbers):
    if len(numbers) == 0:
        return 0  # Handle the case where the list is empty to avoid division by zero.
    
    total = sum(numbers)
    average = total / len(numbers)
    return average

def round_to_4_decimals(arr):
    # Check if the input is a NumPy array
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError("Input must be a one-dimensional NumPy array of floats")
    # Use list comprehension to round each element to 4 decimal places
    rounded_list = [round(x, 4) for x in arr]
    return rounded_list

def round_floats_to_4_decimal_places(input_list):
    rounded_list = [round(num, 4) for num in input_list]
    return rounded_list


def main():
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
    
    matrix, vec_num, vec_size = read_matrix_from_file(filename)
    
    if (k  <= 1 or k >= vec_num):
        print("Invalid number of clusters! \ An Error Has Occurred")
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
        np.random.seed(0)
        results = sm.norm(matrix, vec_num, vec_size)
        norm_mat = results[0]
        norm_mat_round = round_floats_to_4_decimal_places(norm_mat)

        normal_mat_avg = calculate_average(norm_mat_round)

        h = np.random.uniform(0, ((np.sqrt(normal_mat_avg/k) * 2)+1), (k*vec_num))
        
        h_round = round_to_4_decimals(h)
        resultsB = 0, 0
        resultsB = sm.symnmf(h_round, vec_num, k, norm_mat_round, vec_num, vec_size, beta, max_iter, eps)
        print_matrix_with_precision("kkk", resultsB[0], resultsB[1])
    
    else:
        print("An Error Has Occurred")
        return 1

    return 0


if __name__ == "__main__":
    main()