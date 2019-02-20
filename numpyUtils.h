#ifndef NUMPY_UTILS_H
#define NUMPY_UTILS_H

#include <vector>
#include <Eigen/Core>
#include <Python.h>
#include <numpy/arrayobject.h>


void npArrayToStdVectorI(std::vector<int> &y, PyArrayObject *array);
void npArrayToEigenVectorF(Eigen::VectorXf &y, PyArrayObject *array);
void npArrayToEigenVectorD(Eigen::VectorXd &y, PyArrayObject *array);
void npArrayToEigenMatrixF(Eigen::MatrixXf &y, PyArrayObject *array);
void npArrayToEigenMatrixD(Eigen::MatrixXd &y, PyArrayObject *array);
void npArrayToEigenMatrixFTransposed(Eigen::MatrixXf &y, PyArrayObject *array);
void npArrayToEigenMatrixDTransposed(Eigen::MatrixXd &y, PyArrayObject *array);
PyObject *getStdVectorIParam(std::vector<int> &y, PyObject *args);

#endif
