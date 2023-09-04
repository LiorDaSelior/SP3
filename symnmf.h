typedef struct
{
    int row;
    int col;
    double *data;
} Matrix;

Matrix *create_sym_matrix(Matrix *);
Matrix *create_ddg_matrix(Matrix *);
Matrix *create_norm_matrix(Matrix *, Matrix *);
Matrix *create_ass_matrix(Matrix *, Matrix *, double, int, double);