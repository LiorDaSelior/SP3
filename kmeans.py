import sys

def string_to_vector(string):
    return tuple(float(value) for value in string.split(','))

def file_to_string_list(filename):
    with open(filename, 'r') as file:
        tmp = [line.replace('/n','') for line in file.readlines()]
    return tmp

def file_to_vector_list(filename):
    return [string_to_vector(string) for string in file_to_string_list(filename)]

def vector_list_to_vector_data(vector_list):
    return [coor for vector in vector_list for coor in vector]

class Centroid:
    @staticmethod
    def create_k_len_centroid_list(vector_list, k):
        return [Centroid(vector_list[i], vector_list) for i in range(k)]

    def __init__(self, vector, vector_list):
        self.curr_loc = vector
        self.vector_list = vector_list
        self.assigned_vector_list = []

    def update(self):
        self.curr_loc = self.assigned_vector_list[0]
        i = 1
        while i < len(self.assigned_vector_list):
            self.curr_loc = add(self.curr_loc, self.assigned_vector_list[i])
            i+=1
        self.curr_loc = multiply(
            (1/len(self.assigned_vector_list)),
            self.curr_loc)
        self.assigned_vector_list = []

    def __str__(self):
        return f'{self.curr_loc} -- {self.assigned_vector_list}'

def add(vector1, vector2):
    return tuple((vector1[i] + vector2[i]) for i in range(len(vector1)))

def multiply(a, vector):
    return tuple(a * vector[i] for i in range(len(vector)))

def calc_dist(vector1, vector2):
    sum = 0
    for i in range(len(vector1)):
        sum += (vector1[i] - vector2[i]) ** 2
    return sum ** (1/2) 
            
def assign(vector_list, centroid_list):
    centroid_ass_list = []
    for vector in vector_list:
        min_centroid_index = None
        min_dist = None
        for i in range(len(centroid_list)):
            temp = calc_dist(centroid_list[i].curr_loc, vector)
            if min_dist is None or temp <= min_dist:
                min_dist = temp
                min_centroid_index = i
        centroid_list[min_centroid_index].assigned_vector_list.append(vector)
        centroid_ass_list.append(min_centroid_index)
    return centroid_ass_list

def igt_epsilon(vector1, vector2):
    e = 0.0001
    if calc_dist(vector1, vector2) >= e:
        return True
    return False

def algo(vector_list, centroid_list, iter):
    iter_count = 0
    check1 = True
    check2 = True
    old_values = [item.curr_loc for item in centroid_list]
    while check1 and check2:
        centroid_ass_list = assign(vector_list, centroid_list)
        for item in centroid_list:
            item.update()
        iter_count += 1
        check1 = True if iter_count < iter else False
        new_values = [item.curr_loc for item in centroid_list]
        check2 = False
        for i in range(len(new_values)):
            if igt_epsilon(old_values[i],new_values[i]) == True:
                check2 = True
    return centroid_ass_list

def vector_str(vector):
    i = 0
    string = ""
    while (i < len(vector)):
        string += f"{vector[i]:.4f}" 
        if i < len(vector) - 1:
            string += ','
        i += 1
    return string


def main():
    iter = 200
    if (len(sys.argv) == 3):
        k = int(sys.argv[1])
        filename = sys.argv[2]
    elif (len(sys.argv) == 4):
        k = int(sys.argv[1])
        iter = int(sys.argv[2])
        filename = sys.argv[3]
    else:
        print("An Error Has Occurred")
        return 1
    if (iter  <= 1 or iter >= 1000):
        print("Invalid maximum iteration!")
        return 1
    vector_list = file_to_vector_list(filename)
    if (k  <= 1 or k >= len(vector_list)):
        print("Invalid number of clusters!")
        return 1
    centroid_list = Centroid.create_k_len_centroid_list(vector_list, k)
    algo(vector_list, centroid_list, iter)
    for item in centroid_list:
        temp = item.curr_loc
        print(vector_str(temp))
    return 0
    
if __name__ == "__main__":
    main()



        


