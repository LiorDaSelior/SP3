#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

static PyObject *sym(PyObject *self, PyObject *args) // function called from Python file
{
    double num;
    int mat_size, i;
    int rows, cols;
    PyObject *item, *py_mat;
    Matrix *Mat, *sym_mat_c;
    if (!PyArg_ParseTuple(args, "iiO", &rows, &cols, &py_mat))
    {
        return NULL; /*In the CPython API, a NULL value is never valid for a PyObject so it is used to signal that an error has occurred. */
    }

    Mat = (Matrix *)malloc(sizeof(Matrix));
    sym_mat_c = (Matrix *)malloc(sizeof(Matrix));

    mat_size = rows * cols;
    Mat->row = rows;
    Mat->col = cols;
    Mat->data = (double *)malloc(mat_size * sizeof(double));

    if (Mat->data == NULL)
    {
        printf("An Error Has Occurred");
        exit(1);
    }

    for (i = 0; i < mat_size; i++)
    {
        item = PyList_GetItem(py_mat, i);
        num = PyFloat_AsDouble(item);
        (Mat->data)[i] = num;
    }

    sym_mat_c = create_sym_matrix(Mat);
    mat_size = sym_mat_c->col * sym_mat_c->row;

    PyObject *sym_mat_py = PyList_New(0);
    for (i = 0; i < mat_size; i++)
    {
        PyList_Append(sym_mat_py, PyFloat_FromDouble((sym_mat_c->data)[i]));
    }

    PyObject *val = PyTuple_New(2);
    PyTuple_SetItem(val, 0, sym_mat_py);
    PyTuple_SetItem(val, 1, PyLong_FromLong(sym_mat_c->col));

    free(Mat->data);
    free(sym_mat_c->data);
    free(Mat);
    free(sym_mat_c);

    return val;
}

static PyObject *ddg(PyObject *self, PyObject *args) // function called from Python file
{
    double num;
    int mat_size, i;
    int rows, cols;
    PyObject *item, *py_mat;
    Matrix *Mat, *ddg_mat_c, *sym_mat;
    if (!PyArg_ParseTuple(args, "Oii", &py_mat, &rows, &cols))
    {
        return NULL; /*In the CPython API, a NULL value is never valid for a PyObject so it is used to signal that an error has occurred. */
    }

    Mat = (Matrix *)malloc(sizeof(Matrix));
    sym_mat = (Matrix *)malloc(sizeof(Matrix));
    ddg_mat_c = (Matrix *)malloc(sizeof(Matrix));

    mat_size = rows * cols;
    Mat->row = rows;
    Mat->col = cols;
    Mat->data = (double *)malloc(mat_size * sizeof(double));

    if (Mat->data == NULL)
    {
        printf("An Error Has Occurred");
        exit(1);
    }

    for (i = 0; i < mat_size; i++)
    {
        item = PyList_GetItem(py_mat, i);
        num = PyFloat_AsDouble(item);
        (Mat->data)[i] = num;
    }

    sym_mat = create_sym_matrix(Mat);
    ddg_mat_c = create_ddg_matrix(sym_mat);
    mat_size = ddg_mat_c->col * ddg_mat_c->row;

    PyObject *ddg_mat_py = PyList_New(0);
    for (i = 0; i < mat_size; i++)
    {
        PyList_Append(ddg_mat_py, PyFloat_FromDouble((ddg_mat_c->data)[i]));
    }

    PyObject *val = PyTuple_New(2);
    PyTuple_SetItem(val, 0, ddg_mat_py);
    PyTuple_SetItem(val, 1, PyLong_FromLong(ddg_mat_c->col));

    free(Mat->data);
    free(ddg_mat_c->data);
    free(sym_mat->data);
    free(sym_mat);
    free(Mat);
    free(ddg_mat_c);

    return val;
}

static PyObject *norm(PyObject *self, PyObject *args) // function called from Python file
{
    double num;
    int mat_size, i;
    int rows, cols;
    PyObject *item, *py_mat;
    Matrix *Mat, *sym_mat, *ddg_mat, *norm_mat_c;
    if (!PyArg_ParseTuple(args, "Oii", &py_mat, &rows, &cols))
    {
        return NULL; /*In the CPython API, a NULL value is never valid for a PyObject so it is used to signal that an error has occurred. */
    }

    Mat = (Matrix *)malloc(sizeof(Matrix));
    sym_mat = (Matrix *)malloc(sizeof(Matrix));
    ddg_mat = (Matrix *)malloc(sizeof(Matrix));
    norm_mat_c = (Matrix *)malloc(sizeof(Matrix));

    mat_size = rows * cols;
    Mat->row = rows;
    Mat->col = cols;
    Mat->data = (double *)malloc(mat_size * sizeof(double));

    if (Mat->data == NULL)
    {
        printf("An Error Has Occurred");
        exit(1);
    }

    for (i = 0; i < mat_size; i++)
    {
        item = PyList_GetItem(py_mat, i);
        num = PyFloat_AsDouble(item);
        (Mat->data)[i] = num;
    }

    sym_mat = create_sym_matrix(Mat);
    ddg_mat = create_ddg_matrix(sym_mat);
    norm_mat_c = create_norm_matrix(sym_mat, ddg_mat);

    mat_size = (norm_mat_c->col) * (norm_mat_c->row);

    PyObject *norm_mat_py = PyList_New(0);
    for (i = 0; i < mat_size; i++)
    {
        PyList_Append(norm_mat_py, PyFloat_FromDouble((norm_mat_c->data)[i]));
    }

    PyObject *val = PyTuple_New(2);
    PyTuple_SetItem(val, 0, norm_mat_py);
    PyTuple_SetItem(val, 1, PyLong_FromLong(norm_mat_c->col));

    free(Mat->data);
    free(ddg_mat->data);
    free(sym_mat->data);
    free(norm_mat_c->data);
    free(Mat);
    free(sym_mat);
    free(ddg_mat);
    free(norm_mat_c);

    return val;
}

static PyObject *symnmf(PyObject *self, PyObject *args) // function called from Python file
{
    double num, beta, eps;
    int mat_size, iter, i;
    int h_rows, h_cols, norm_rows, norm_cols;
    PyObject *item, *py_mat_h, *py_mat_norm;
    Matrix *h_mat, *norm_mat, *symnmf_mat_c;
    if (!PyArg_ParseTuple(args, "OiiOiidid", &py_mat_h, &h_rows, &h_cols, &py_mat_norm, &norm_rows, &norm_cols, &beta, &iter, &eps))
    {
        return NULL; /*In the CPython API, a NULL value is never valid for a PyObject so it is used to signal that an error has occurred. */
    }

    h_mat = (Matrix *)malloc(sizeof(Matrix));
    norm_mat = (Matrix *)malloc(sizeof(Matrix));
    symnmf_mat_c = (Matrix *)malloc(sizeof(Matrix));

    mat_size = h_rows * h_cols;
    h_mat->row = h_rows;
    h_mat->col = h_cols;
    h_mat->data = (double *)malloc(mat_size * sizeof(double));

    if (h_mat->data == NULL)
    {
        printf("An Error Has Occurred");
        exit(1);
    }

    for (i = 0; i < mat_size; i++)
    {
        item = PyList_GetItem(py_mat_h, i);
        num = PyFloat_AsDouble(item);
        (h_mat->data)[i] = num;
    }

    mat_size = norm_rows * norm_cols;
    norm_mat->row = norm_rows;
    norm_mat->col = norm_cols;
    norm_mat->data = (double *)malloc(mat_size * sizeof(double));

    if (norm_mat->data == NULL)
    {
        printf("An Error Has Occurred");
        exit(1);
    }

    for (i = 0; i < mat_size; i++)
    {
        item = PyList_GetItem(py_mat_norm, i);
        num = PyFloat_AsDouble(item);
        (norm_mat->data)[i] = num;
    }

    symnmf_mat_c = create_ass_matrix(h_mat, norm_mat, beta, iter, eps);
    mat_size = (symnmf_mat_c->col) * (symnmf_mat_c->row);
    printf("\ncol %d, row, %d", symnmf_mat_c->col, symnmf_mat_c->row);
    printf("\nmat size %d\n", mat_size);

    printf("\nbegin\n");
    print_matrix(symnmf_mat_c);
    printf("end\n");

    PyObject *symnmf_mat_py = PyList_New(0);
    for (i = 0; i < mat_size; i++)
    {
        PyList_Append(symnmf_mat_py, PyFloat_FromDouble((symnmf_mat_c->data)[i]));
    }

    PyObject *val = PyTuple_New(2);
    assert(val != NULL);
    PyTuple_SetItem(val, 0, symnmf_mat_py);
    PyTuple_SetItem(val, 1, PyLong_FromLong(symnmf_mat_c->col));

    // free(h_mat->data);
    free(norm_mat->data);
    // free(h_mat);
    free(norm_mat);

    return val;
}

static PyMethodDef symnmfMethods[] = {
    // declaring the functions in C to the Python file

    {"sym",                            /* the Python method name that will be used */
     (PyCFunction)sym,                 /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,                     /* flags indicating parameters accepted for this function */
     PyDoc_STR("sym matrix creator")}, /*  The docstring for the function */

    {"ddg",                            /* the Python method name that will be used */
     (PyCFunction)ddg,                 /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,                     /* flags indicating parameters accepted for this function */
     PyDoc_STR("ddg matrix creator")}, /*  The docstring for the function */

    {"norm",                            /* the Python method name that will be used */
     (PyCFunction)norm,                 /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,                      /* flags indicating parameters accepted for this function */
     PyDoc_STR("norm matrix creator")}, /*  The docstring for the function */

    {"symnmf",                            /* the Python method name that will be used */
     (PyCFunction)symnmf,                 /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,                        /* flags indicating parameters accepted for this function */
     PyDoc_STR("symnmf matrix creator")}, /*  The docstring for the function */

    {NULL, NULL, 0, NULL} /* The last entry must be all NULL as shown to act as a
                             sentinel. Python looks for this entry to know that all
                             of the functions for the module have been defined. */
};

static struct PyModuleDef symnmf_module = {
    // initiating the module

    PyModuleDef_HEAD_INIT,
    "symnmfmodule", /* name of module */
    NULL,           /* module documentation, may be NULL */
    -1,             /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    symnmfMethods   /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
// the function that will be called when the module is imported in python

{
    PyObject *m;
    m = PyModule_Create(&symnmf_module);
    if (!m)
    {
        return NULL;
    }
    return m;
}