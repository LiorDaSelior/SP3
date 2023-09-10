#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "symnmf.h"

#define GET(mat, i, j) (mat)->data[(i) * ((mat)->col) + (j)]
#define SET(mat, i, j, value) GET((mat), (i), (j)) = (value)

struct cord
{
    double value;
    struct cord *next;
};

struct vector
{
    struct vector *next;
    struct cord *cords;
};

int string_to_vector(struct vector *, int, double *);
int free_matrix(Matrix *);
int set_matrix_data(Matrix *matrix, int row, int col);
double calc_sq_dist(Matrix *, int, int);
Matrix *matrix_mult(Matrix *, Matrix *);
Matrix *matrix_trans(Matrix *);
double calc_sq_dist(Matrix *, int, int);
Matrix *create_updated_h_matrix(Matrix *, Matrix *, double);
double calc_frobenius(Matrix *, Matrix *);

int string_to_vector(struct vector *head_vec, int i, double *coor_arr)
{
    int l = i;
    struct vector *run_vec, *prev_vec;
    struct cord *run_cord, *prev_cord;
    run_vec = head_vec;
    while (run_vec != NULL)
    {
        run_cord = run_vec->cords;
        while (run_cord != NULL)
        {
            coor_arr[l] = (double)run_cord->value;
            l++;
            prev_cord = run_cord;
            run_cord = run_cord->next;
            free(prev_cord);
        }
        prev_vec = run_vec;
        run_vec = run_vec->next;
        free(prev_vec);
    }
    return 0;
}

int free_matrix(Matrix *matrix)
{
    free(matrix->data);
    free(matrix);
    return 0;
}

int set_matrix_data(Matrix *matrix, int row, int col)
{
    matrix->row = row;
    matrix->col = col;
    matrix->data = (double *)calloc(row * col, sizeof(double));
    assert(matrix->data != NULL);
    return 0;
}

int print_matrix(Matrix *matrix)
{
    int i=0, j=0;
    for (i = 0; i < matrix->row; i++)
    {
        for (j = 0; j < matrix->col; j++)
        {
            if (j == matrix->col - 1)
            {
                printf("%.4f", GET(matrix, i, j));
            }
            else
            {
                printf("%.4f,", GET(matrix, i, j));
            }
        }
        printf("\n"); /*check if newline needed*/
    }
    return 0;
}

Matrix *matrix_mult(Matrix *mat1, Matrix *mat2)
{
    int i=0, j=0, k=0;
    double calc;
    Matrix *temp = (Matrix *)malloc(sizeof(Matrix));
    assert(mat1->col == mat2->row);
    assert(temp != NULL);
    set_matrix_data(temp, mat1->row, mat2->col);
    for (i = 0; i < temp->row; i++)
    {
        for (j = 0; j < temp->col; j++)
        {
            calc = 0;
            for (k = 0; k < mat1->col; k++)
            {
                calc = calc + (GET(mat1, i, k) * GET(mat2, k, j));
            }
            SET(temp, i, j, calc);
        }
    }
    return temp;
}

Matrix *matrix_trans(Matrix *mat)
{
    int i=0, j=0;
    Matrix *temp = (Matrix *)malloc(sizeof(Matrix));
    assert(temp != NULL);
    set_matrix_data(temp, mat->col, mat->row);
    for (i = 0; i < temp->row; i++)
    {
        for (j = 0; j < temp->col; j++)
        {
            SET(temp, i, j, GET(mat, j, i));
        }
    }
    return temp;
}

Matrix *create_sym_matrix(Matrix *vec_matrix)
{
    int i=0, j=0;
    double calc;
    Matrix *temp = (Matrix *)malloc(sizeof(Matrix));
    assert(temp != NULL);
    set_matrix_data(temp, vec_matrix->row, vec_matrix->row);
    for (i = 0; i < temp->row; i++)
    {
        for (j = 0; j < temp->col; j++)
        {
            calc = (i == j) ? 0 : exp(-1.0 * 0.5 * calc_sq_dist(vec_matrix, i, j));
            SET(temp, i, j, calc);
        }
    }
    return temp;
}

Matrix *create_ddg_matrix(Matrix *sym_matrix)
{
    int i=0, j=0;
    double calc;
    Matrix *temp = (Matrix *)malloc(sizeof(Matrix));
    assert(temp != NULL);
    set_matrix_data(temp, sym_matrix->row, sym_matrix->col);
    for (i = 0; i < temp->row; i++)
    {
        calc = 0;
        for (j = 0; j < temp->col; j++)
        {
            calc = calc + GET(sym_matrix, i, j);
        }
        SET(temp, i, i, calc);
    }
    return temp;
}

Matrix *create_norm_matrix(Matrix *sym_matrix, Matrix *ddg_matrix)
{
    Matrix *m1, *temp;
    int i=0, j=0;
    double calc;
    for (i = 0; i < ddg_matrix->row; i++)
    {
        for (j = 0; j < ddg_matrix->col; j++)
        {
            calc = (GET(ddg_matrix, i, j) == 0) ? 0 : pow((GET(ddg_matrix, i, j)), -0.5f);
            SET(ddg_matrix, i, j, calc);
        }
    }
    m1 = matrix_mult(ddg_matrix, sym_matrix);
    temp = matrix_mult(m1, ddg_matrix);
    free_matrix(m1);
    return temp;
}

double calc_sq_dist(Matrix *vec_matrix, int vec1_index, int vec2_index)
{
    int i = 0;
    double total = 0;
    if (vec1_index == vec2_index)
    {
        return 0;
    }
    for (i = 0; i < vec_matrix->col; i++)
    {
        total = total + pow(GET(vec_matrix, vec1_index, i) - GET(vec_matrix, vec2_index, i), 2);
    }
    return total;
}

Matrix *create_updated_h_matrix(Matrix *h_matrix, Matrix *norm_matrix, double beta)
{
    int i=0, j=0;
    Matrix *temp = matrix_mult(norm_matrix, h_matrix);
    Matrix *h_matrix_trans = matrix_trans(h_matrix);
    Matrix *m1 = matrix_mult(norm_matrix, h_matrix);
    Matrix *m2_temp = matrix_mult(h_matrix, h_matrix_trans);
    Matrix *m2 = matrix_mult(m2_temp, h_matrix);
    double calc;
    for (i = 0; i < temp->row; i++)
    {
        for (j = 0; j < temp->col; j++)
        {
            calc = GET(h_matrix, i, j) * (1 - beta + beta * (GET(m1, i, j) / GET(m2, i, j)));
            SET(temp, i, j, calc);
        }
    }
    free_matrix(h_matrix_trans);
    free_matrix(m1);
    free_matrix(m2_temp);
    free_matrix(m2);
    return temp;
}

double calc_frobenius(Matrix *mat1, Matrix *mat2)
{
    int i=0, j=0;
    double total = 0;
    for (i = 0; i < mat1->row; i++)
    {
        for (j = 0; j < mat1->col; j++)
        {
            total = total + pow((GET(mat1, i, j) - GET(mat2, i, j)), 2);
        }
    }
    return total;
}

Matrix *create_ass_matrix(Matrix *h_matrix, Matrix *norm_matrix, double beta, int iter, double eps)
{
    int i = 0;
    int check = 0;
    Matrix *prev_mat = h_matrix;
    Matrix *next_mat = h_matrix;
    while ((i < iter) && check == 0)
    {
        next_mat = create_updated_h_matrix(prev_mat, norm_matrix, beta);
        if (calc_frobenius(next_mat, prev_mat) < eps)
        {
            check = 1;
        }
        free_matrix(prev_mat);
        prev_mat = next_mat;
        i++;
    }
    return next_mat;
}

int main(int argc, char const *argv[])
{
    struct vector *head_vec, *curr_vec, *prev_vec;
    struct cord *head_cord, *curr_cord;
    int check = 0, vec_size = 1, vec_num = 0, i = 0, arr_size;
    double n;
    double *coor_arr;
    char c;
    char const *mode, *filename;
    Matrix *vector_matrix, *sym_matrix, *ddg_matrix, *norm_matrix;
    FILE *fp;

    if (argc != 3)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }

    mode = argv[1];
    filename = argv[2];

    head_cord = malloc(sizeof(struct cord));
    curr_cord = head_cord;
    curr_cord->next = NULL;

    head_vec = malloc(sizeof(struct vector));
    curr_vec = head_vec;
    curr_vec->next = NULL;
    prev_vec = curr_vec;

    fp = fopen(filename, "r");
    assert(fp != NULL);

    while (fscanf(fp, "%lf%c", &n, &c) == 2)
    {
        if (check == 0) {
            check++;
        }
        if (c == '\n')
        {
            curr_cord->value = n;
            curr_vec->cords = head_cord;
            curr_vec->next = malloc(sizeof(struct vector));
            if (curr_vec->next == NULL)
            {
                printf("An Error Has Occurred\n");
                return 1;
            }
            vec_num++;
            prev_vec = curr_vec;
            curr_vec = curr_vec->next;
            curr_vec->next = NULL;
            head_cord = malloc(sizeof(struct cord));
            if (head_cord == NULL)
            {
                printf("An Error Has Occurred\n");
                return 1;
            }
            curr_cord = head_cord;
            curr_cord->next = NULL;
            continue;
        }
        curr_cord->value = n;
        curr_cord->next = malloc(sizeof(struct cord));
        if (curr_cord->next == NULL)
        {
            printf("An Error Has Occurred\n");
            return 1;
        }
        if (vec_num == 1)
        {
            vec_size++;
        }
        curr_cord = curr_cord->next;
        curr_cord->next = NULL;
    }
    fclose(fp);
    
    if (check == 1) {
        free(curr_cord);
        free(curr_vec);
        curr_vec = prev_vec;
        curr_vec->next = NULL;
    }
    else {
        free(curr_cord);
    }

    arr_size = vec_num * vec_size;
    coor_arr = (double *)malloc(arr_size * sizeof(double));

    if (coor_arr == NULL)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }

    string_to_vector(head_vec, i, coor_arr);

    vector_matrix = (Matrix *)malloc(sizeof(Matrix));
    vector_matrix->row = vec_num;
    vector_matrix->col = vec_size;
    vector_matrix->data = coor_arr;

    if (strcmp(mode, "sym") == 0)
    {
        sym_matrix = create_sym_matrix(vector_matrix);
        print_matrix(sym_matrix);
        free_matrix(sym_matrix);
    }
    else if (strcmp(mode, "ddg") == 0)
    {
        sym_matrix = create_sym_matrix(vector_matrix);
        ddg_matrix = create_ddg_matrix(sym_matrix);
        print_matrix(ddg_matrix);
        free_matrix(sym_matrix);
        free_matrix(ddg_matrix);
    }
    else if (strcmp(mode, "norm") == 0)
    {
        sym_matrix = create_sym_matrix(vector_matrix);
        ddg_matrix = create_ddg_matrix(sym_matrix);
        norm_matrix = create_norm_matrix(sym_matrix, ddg_matrix);
        print_matrix(norm_matrix);
        free_matrix(sym_matrix);
        free_matrix(ddg_matrix);
        free_matrix(norm_matrix);
    }
    else
    {
        printf("An Error Has Occurred\n");
        free_matrix(vector_matrix);
        return 1;
    }
    free_matrix(vector_matrix);
    return 0;
}
