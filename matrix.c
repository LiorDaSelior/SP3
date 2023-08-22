#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define GET(mat,i,j) (mat)->data[(i) * ((mat)->col) + (j)]
#define SET(mat,i,j,value) GET((mat),(i),(j)) = (value)

typedef struct {
int row;
int col;
double* data;
} Matrix;

double calc_sq_dist(Matrix*, int, int);

int free_matrix(Matrix* matrix) {
    free(matrix->data);
    free(matrix);
    return 0;
}

int set_matrix_data(Matrix* matrix, int row, int col) {
    matrix -> row = row;
    matrix -> col = col;
    matrix -> data = (double*)calloc(row * col, sizeof(double));
    assert(matrix -> data != NULL);
    return 0;
}

int print_matrix(Matrix* matrix) {
    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++) {
            printf("%.4f,",GET(matrix, i, j));
        }
        printf("\n");
    }
    return 0;
}

Matrix* matrix_mult(Matrix* mat1, Matrix* mat2) {
    assert(mat1->col == mat2->row);
    double calc;
    Matrix* temp = (Matrix*)malloc(sizeof(Matrix));
    assert(temp != NULL);
    set_matrix_data(temp, mat1 -> row, mat2 -> col);
    for (int i = 0; i < temp->row; i++) {
        for (int j = 0; j < temp->col; j++) {
            calc = 0;
            for (int k = 0; k < mat1->col; k++) {
                calc = calc + (GET(mat1,i,k) * GET(mat2,k,j));
            }
            SET(temp, i, j, calc);
        }
    }
    return temp;
}

// Matrix* matrix_trans(Matrix* mat) {
//     double calc;
//     Matrix* temp = (Matrix*)malloc(sizeof(Matrix));
//     assert(temp != NULL);
//     set_matrix_data(temp, mat1 -> row, mat2 -> col);
//     for (int i = 0; i < temp->row; i++) {
//         for (int j = 0; j < temp->col; j++) {
//             calc = 0;
//             for (int k = 0; k < mat1->col; k++) {
//                 calc = calc + (GET(mat1,i,k) * GET(mat2,k,j));
//             }
//             SET(temp, i, j, calc);
//         }
//     }
//     return temp;
// }

Matrix* create_sym_matrix(Matrix* vec_matrix) {
    double calc;
    Matrix* temp = (Matrix*)malloc(sizeof(Matrix));
    assert(temp != NULL);
    set_matrix_data(temp, vec_matrix -> row, vec_matrix -> row);
    for (int i = 0; i < temp->row; i++) {
        for (int j = 0; j < temp->col; j++) {
            calc = (i == j) ? 0 : exp(-1.0 * 0.5 * calc_sq_dist(vec_matrix, i, j));
            SET(temp, i, j, calc);
        }
    }
    return temp;
}

Matrix* create_ddg_matrix(Matrix* sym_matrix) {
    double calc;
    Matrix* temp = (Matrix*)malloc(sizeof(Matrix));
    assert(temp != NULL);
    set_matrix_data(temp, sym_matrix -> row, sym_matrix -> col);
    for (int i = 0; i < temp->row; i++) {
        calc = 0;
        for (int j = 0; j < temp->col; j++) {
            calc = calc + GET(sym_matrix, i, j);
        }
        SET(temp, i, i, calc);
    }
    return temp;
}

Matrix* create_norm_matrix(Matrix* sym_matrix, Matrix* ddg_matrix) {
    double calc;
    for (int i = 0; i < ddg_matrix->row; i++) {
        for (int j = 0; j < ddg_matrix->col; j++) {
            calc = (GET(ddg_matrix, i, j)==0) ? 0 : pow((GET(ddg_matrix, i, j)), -0.5f);
            SET(ddg_matrix, i, j, calc);
        }
    }
    Matrix* temp = matrix_mult(ddg_matrix, sym_matrix);
    temp = matrix_mult(temp, ddg_matrix);
    return temp;
}

double calc_sq_dist(Matrix* vec_matrix, int vec1_index, int vec2_index) {
    if (vec1_index == vec2_index) {return 0;}
    double total = 0;
    for (int i = 0; i < vec_matrix->col; i++) {
        total = total + pow(GET(vec_matrix, vec1_index, i) - GET(vec_matrix, vec2_index, i), 2);
    }
    return total;
}

// Matrix* update_dec_matrix(Matrix* h_matrix, Matrix* norm_matrix, int beta) {
//     Matrix* m1 = matrix_mult(norm_matrix, h_matrix);
//     Matrix* m2 = matrix_mult(h_matrix, h_matrix_trans);
//     m2 = matrix_mult(m2, h_matrix);

//     double calc;
//     for (int i = 0; i < ddg_matrix->row; i++) {
//         for (int j = 0; j < ddg_matrix->col; j++) {
//             calc = (GET(ddg_matrix, i, j)==0) ? 0 : pow((GET(ddg_matrix, i, j)), -0.5f);
//             SET(ddg_matrix, i, j, calc);
//         }
//     }
//     Matrix* m1 = matrix_mult(ddg_matrix, sym_matrix);
//     temp = matrix_mult(temp, ddg_matrix);
//     return temp;
// }

int main(int argc, char const *argv[])
{
    Matrix* vec_matrix = (Matrix*)malloc(sizeof(Matrix));
    set_matrix_data(vec_matrix, 4, 3);
    vec_matrix -> data[0] = 0.1;
    vec_matrix -> data[1] = 0.2;
    vec_matrix -> data[2] = 0.3;
    vec_matrix -> data[3] = 0.5;
    vec_matrix -> data[4] = -0.5;
    vec_matrix -> data[5] = 1;
    vec_matrix -> data[6] = 0.1;
    vec_matrix -> data[7] = 0.1;
    vec_matrix -> data[8] = 0.2;
    vec_matrix -> data[9] = 0.3;
    vec_matrix -> data[10] = 0.6;
    vec_matrix -> data[11] = 0.9;
    print_matrix(vec_matrix);
    printf("\n");
    Matrix* sym_matrix = create_sym_matrix(vec_matrix);
    print_matrix(sym_matrix);
    printf("\n");
    Matrix* ddg_matrix = create_ddg_matrix(sym_matrix);
    print_matrix(ddg_matrix);
    printf("\n");
    Matrix* norm_matrix = create_norm_matrix(sym_matrix, ddg_matrix);
    print_matrix(norm_matrix);
    return 0;
}
