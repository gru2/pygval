
#ifdef _DEBUG
	#undef _DEBUG
		#include <Python.h>
	#define _DEBUG
#else
	#include <Python.h>
#endif

#include <numpy/arrayobject.h>
#include "structmember.h"
#include <numpyUtils.h>
#include <iostream>
#include <GVal.h>



/// Class GVal Wrapper.

typedef struct
{
	PyObject_HEAD
	GVal data;
} GValObject;

static void
GVal_dealloc_(GValObject *self)
{
	std::cout << "calling GVal destructor...\n";
	self->data.~GVal();
	Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
GVal_new_(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	GValObject *self;
	self = (GValObject *) type->tp_alloc(type, 0);
	if (self != NULL) {
		std::cout << "calling GVal constructor...\n";
		new (&self->data) GVal();
	}
	return (PyObject *) self;
}

static int GVal_init_(GValObject *self, PyObject *args, PyObject *kwds)
{
	return 0;
}

static PyMemberDef GVal_members[] = {

	{NULL, 0, 0, 0, NULL}  /* Sentinel */
};

static PyObject *GVal_setString(GValObject *self, PyObject *args)
{
	char *x;
	if (!PyArg_ParseTuple(args, "s", &x))
		return NULL;

	self->data.setString(x);

	Py_RETURN_NONE;
}

static PyMethodDef GVal_methods[] =
{
	{"setString", (PyCFunction) GVal_setString, METH_VARARGS, "setString..."},
	{NULL}  /* Sentinel */
};

static void initGValType(PyTypeObject &ct)
{
	ct.tp_name = "pygval.GVal";
	ct.tp_doc = "GVal objects";
	ct.tp_basicsize = sizeof(GValObject);
	ct.tp_itemsize = 0;
	ct.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	ct.tp_new = GVal_new_;
	ct.tp_init = (initproc) GVal_init_;
	ct.tp_dealloc = (destructor) GVal_dealloc_;
	ct.tp_members = GVal_members;
	ct.tp_methods = GVal_methods;
}

static PyTypeObject GValType = {
	PyVarObject_HEAD_INIT(NULL, 0)
};


// pygval Module

// PYTHON2
#if PY_MAJOR_VERSION == 2
static PyMethodDef pygval_methods[] = {
	{NULL}  /* Sentinel */
};

PyMODINIT_FUNC initpygval(void)
{
	initGValType(GValType);
	if (PyType_Ready(&GValType) < 0)
		return;

	import_array();
	PyObject *m = Py_InitModule3("pygval", pygval_methods,
		"pygval module...");
	if (m == NULL)
		return;

	Py_INCREF(&GValType);
	PyModule_AddObject(m, "GVal", (PyObject *) &GValType);
}
#endif


// PYTHON3
#if PY_MAJOR_VERSION == 3
static PyModuleDef pygvalmodule = {
	PyModuleDef_HEAD_INIT,
	.m_name = "pygval",
	.m_doc = "Module implements gval library wrappers..",
	.m_size = -1,
};

PyMODINIT_FUNC
PyInit_pygval(void)
{
	initGValType(GValType);
	if (PyType_Ready(&GValType) < 0)
		return NULL;

	import_array();
	PyObject *m = PyModule_Create(&pygvalmodule);
	if (m == NULL)
		return NULL;

	Py_INCREF(&GValType);
	PyModule_AddObject(m, "GVal", (PyObject *) &GValType);

	return m;
}
#endif

