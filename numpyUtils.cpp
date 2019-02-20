#include <numpyUtils.h>

void npArrayToStdVectorI(std::vector<int> &y, PyArrayObject *array)
{
	int *data = (int *)array->data;
	int size = (int)array->dimensions[0];

	y.resize(size);
	for (int i = 0; i < size; i++)
	{
		y[i] = data[i];
	}
}

void npArrayToEigenVectorD(Eigen::VectorXd &y, PyArrayObject *array)
{
	double *data = (double *)array->data;
	int size = (int)array->dimensions[0];

	y.resize(size);
	for (int i = 0; i < size; i++)
		y[i] = data[i];
}

void npArrayToEigenVectorF(Eigen::VectorXf &y, PyArrayObject *array)
{
	float *data = (float *)array->data;
	int size = (int)array->dimensions[0];

	y.resize(size);
	for (int i = 0; i < size; i++)
		y[i] = data[i];
}

void npArrayToEigenMatrixF(Eigen::MatrixXf &y, PyArrayObject *array)
{
	float *data = (float *)array->data;
	int rows = (int)array->dimensions[0];
	int cols = (int)array->dimensions[1];

	y.resize(rows, cols);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			y(i, j) = data[i * cols + j];
}

void npArrayToEigenMatrixD(Eigen::MatrixXd &y, PyArrayObject *array)
{
	double *data = (double *)array->data;
	int rows = (int)array->dimensions[0];
	int cols = (int)array->dimensions[1];

	y.resize(rows, cols);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			y(i, j) = data[i * cols + j];
}

void npArrayToEigenMatrixFTransposed(Eigen::MatrixXf &y, PyArrayObject *array)
{
	float *data = (float *)array->data;
	int rows = (int)array->dimensions[0];
	int cols = (int)array->dimensions[1];

	y.resize(cols, rows);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			y(j, i) = data[i * cols + j];
}

void npArrayToEigenMatrixDTransposed(Eigen::MatrixXd &y, PyArrayObject *array)
{
	double *data = (double *)array->data;
	int rows = (int)array->dimensions[0];
	int cols = (int)array->dimensions[1];

	y.resize(cols, rows);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			y(j, i) = data[i * cols + j];
}

PyObject *getStdVectorIParam(std::vector<int> &y, PyObject *args)
{
	PyObject *arg1 = NULL;
	if (!PyArg_ParseTuple(args, "O", &arg1))
		return NULL;

	PyArrayObject *arr1 = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_INT32, NPY_IN_ARRAY);
	if (arr1 == NULL)
		return NULL;
	if (arr1->nd != 1)
		return NULL;

	npArrayToStdVectorI(y, arr1);

	Py_DECREF(arr1);
	Py_INCREF(Py_None);
	return Py_None;
}
