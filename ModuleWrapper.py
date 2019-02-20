import sys

moduleProto = """
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
$HEADERS

$CLASSES

// $MODULE Module

// PYTHON2
#if PY_MAJOR_VERSION == 2
static PyMethodDef $MODULE_methods[] = {
	{NULL}  /* Sentinel */
};

PyMODINIT_FUNC init$MODULE(void)
{
$INIT_TYPES_PY2
	import_array();
	PyObject *m = Py_InitModule3("$MODULE", $MODULE_methods,
		"$MODULE module...");
	if (m == NULL)
		return;
$ADD_OBJECTS
}
#endif


// PYTHON3
#if PY_MAJOR_VERSION == 3
static PyModuleDef $MODULEmodule = {
	PyModuleDef_HEAD_INIT,
	.m_name = "$MODULE",
	.m_doc = "$MODULE_DOC$.",
	.m_size = -1,
};

PyMODINIT_FUNC
PyInit_$MODULE(void)
{
$INIT_TYPES_PY3
	import_array();
	PyObject *m = PyModule_Create(&$MODULEmodule);
	if (m == NULL)
		return NULL;
$ADD_OBJECTS

	return m;
}
#endif
"""

classProto = """
/// Class $CLASS Wrapper.

typedef struct
{
	PyObject_HEAD
	$CLASS data;
} $CLASSObject;

static void
$CLASS_dealloc_($CLASSObject *self)
{
	std::cout << "calling $CLASS destructor...\\n";
	self->data.~$CLASS();
	Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
$CLASS_new_(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	$CLASSObject *self;
	self = ($CLASSObject *) type->tp_alloc(type, 0);
	if (self != NULL) {
		std::cout << "calling $CLASS constructor...\\n";
		new (&self->data) $CLASS();
	}
	return (PyObject *) self;
}

static int $CLASS_init_($CLASSObject *self, PyObject *args, PyObject *kwds)
{
	return 0;
}

static PyMemberDef $CLASS_members[] = {
$MEMBERS
	{NULL, 0, 0, 0, NULL}  /* Sentinel */
};
$METHODS
static PyMethodDef $CLASS_methods[] =
{
$DEFINITIONS
	{NULL}  /* Sentinel */
};

static void init$CLASSType(PyTypeObject &ct)
{
	ct.tp_name = "$MODULE.$CLASS";
	ct.tp_doc = "$CLASS objects";
	ct.tp_basicsize = sizeof($CLASSObject);
	ct.tp_itemsize = 0;
	ct.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	ct.tp_new = $CLASS_new_;
	ct.tp_init = (initproc) $CLASS_init_;
	ct.tp_dealloc = (destructor) $CLASS_dealloc_;
	ct.tp_members = $CLASS_members;
	ct.tp_methods = $CLASS_methods;
}

static PyTypeObject $CLASSType = {
	PyVarObject_HEAD_INIT(NULL, 0)
};
"""
methodDefProto = "	{\"$METHOD\", (PyCFunction) $CLASS_$METHOD, $METH_FLAG, \"$METHOD...\"},"
addTypeProtoPy2 = """	init$CLASSType($CLASSType);
	if (PyType_Ready(&$CLASSType) < 0)
		return;
"""
addTypeProtoPy3 = """	init$CLASSType($CLASSType);
	if (PyType_Ready(&$CLASSType) < 0)
		return NULL;
"""
addObjectProto = """
	Py_INCREF(&$CLASSType);
	PyModule_AddObject(m, "$CLASS", (PyObject *) &$CLASSType);"""
memberProto = """
	{"$NAME", $TYPE, offsetof($CLASSObject, data) + $OFFSET, 0, "$DOC"},"""
methodProto_Void_VectorI = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	PyObject *arg1 = NULL;
	if (!PyArg_ParseTuple(args, "O", &arg1))
		return NULL;

	PyArrayObject *arr1 = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_INT32, NPY_IN_ARRAY);
	if (arr1 == NULL)
		return NULL;
	if (arr1->nd != 1)
		return NULL;

	std::vector<int> x;
	npArrayToStdVectorI(x, arr1);
	self->data.$CODE;

	Py_DECREF(arr1);
	Py_RETURN_NONE;
}
"""
methodProto_Void_VectorF = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	PyObject *arg1 = NULL;
	if (!PyArg_ParseTuple(args, "O", &arg1))
		return NULL;

	PyArrayObject *arr1 = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_FLOAT32, NPY_IN_ARRAY);
	if (arr1 == NULL)
		return NULL;
	if (arr1->nd != 1)
		return NULL;

	Eigen::VectorXf x;
	npArrayToEigenVectorF(x, arr1);
	self->data.$CODE;

	Py_DECREF(arr1);
	Py_RETURN_NONE;
}
"""
methodProto_Void_VectorD = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	PyObject *arg1 = NULL;
	if (!PyArg_ParseTuple(args, "O", &arg1))
		return NULL;

	PyArrayObject *arr1 = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
	if (arr1 == NULL)
		return NULL;
	if (arr1->nd != 1)
		return NULL;

	Eigen::VectorXd x;
	npArrayToEigenVectorD(x, arr1);
	self->data.$CODE;

	Py_DECREF(arr1);
	Py_RETURN_NONE;
}
"""
methodProto_Void_MatrixF = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	PyObject *arg1 = NULL;
	if (!PyArg_ParseTuple(args, "O", &arg1))
		return NULL;

	PyArrayObject *arr1 = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_FLOAT32, NPY_IN_ARRAY);
	if (arr1 == NULL)
		return NULL;
	if (arr1->nd != 2)
		return NULL;

	Eigen::MatrixXf x;
	npArrayToEigenMatrixF(x, arr1);
	self->data.$CODE;

	Py_DECREF(arr1);
	Py_RETURN_NONE;
}
"""
methodProto_Void_MatrixD = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	PyObject *arg1 = NULL;
	if (!PyArg_ParseTuple(args, "O", &arg1))
		return NULL;

	PyArrayObject *arr1 = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
	if (arr1 == NULL)
		return NULL;
	if (arr1->nd != 2)
		return NULL;

	Eigen::MatrixXd x;
	npArrayToEigenMatrixD(x, arr1);
	self->data.$CODE;

	Py_DECREF(arr1);
	Py_RETURN_NONE;
}
"""
methodProto_Void_Type = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	PyObject *arg1 = NULL;
	if (!PyArg_ParseTuple(args, "O!", &$TYPEType, &arg1))
		return NULL;
	$TYPEObject *x = ($TYPEObject *)arg1;

	std::cout << "calling $METHOD...\\n";
	self->data.$CODE;
	std::cout << "$METHOD called...\\n";

	Py_RETURN_NONE;
}
"""
methodProto_Void_Int = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	int x;
	if (!PyArg_ParseTuple(args, "i", &x))
		return NULL;

	self->data.$CODE;

	Py_RETURN_NONE;
}
"""
methodProto_Void_Float = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	float x;
	if (!PyArg_ParseTuple(args, "f", &x))
		return NULL;

	self->data.$CODE;

	Py_RETURN_NONE;
}
"""
methodProto_Void_Double = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	double x;
	if (!PyArg_ParseTuple(args, "d", &x))
		return NULL;

	self->data.$CODE;

	Py_RETURN_NONE;
}
"""
methodProto_Void_String = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
	char *x;
	if (!PyArg_ParseTuple(args, "s", &x))
		return NULL;

	self->data.$CODE;

	Py_RETURN_NONE;
}
"""
methodProto_VectorI_Void = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *args)
{
npy_intp shape[1];
shape[0] = (npy_intp)self->data.$CODE.size();

PyArrayObject *r = (PyArrayObject *)PyArray_SimpleNewFromData(1, shape, NPY_INT32, &(self->data.$CODE[0]));

return (PyObject *)r;
}
"""
methodProto_VectorF_Void = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *ignored)
{
	npy_intp shape[1];
	shape[0] = (npy_intp)self->data.$CODE.size();

	PyArrayObject *r = (PyArrayObject *)PyArray_SimpleNewFromData(1, shape, NPY_FLOAT32, self->data.$CODE.data());

	return (PyObject *)r;
}
"""
methodProto_VectorD_Void = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *ignored)
{
	npy_intp shape[1];
	shape[0] = (npy_intp)self->data.$CODE.size();

	PyArrayObject *r = (PyArrayObject *)PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE, self->data.$CODE.data());

	return (PyObject *)r;
}
"""
methodProto_Void_Void = """
static PyObject *$CLASS_$METHOD($CLASSObject *self, PyObject *ignored)
{
	self->data.$CODE;
	Py_RETURN_NONE;
}
"""


class MethodWrapper:
    def __init__(self):
        self.name = "someMethod"
        self.signature = "void(void)"
        self.code = "*"
        self.argType = ""
        self.classWrapper = None
        self.meth = "METH_VARARGS"
        self.proto = ""

    def getBody(self, classWrapper):
        self.meth = "METH_VARARGS"
        self.classWrapper = classWrapper
        self.proto = ""
        if self.signature == "void(VectorI)":
            self.proto = methodProto_Void_VectorI
        if self.signature == "void(VectorF)":
            self.proto = methodProto_Void_VectorF
        if self.signature == "void(VectorD)":
            self.proto = methodProto_Void_VectorD
        if self.signature == "void(MatrixF)":
            self.proto = methodProto_Void_MatrixF
        if self.signature == "void(MatrixD)":
            self.proto = methodProto_Void_MatrixD
        if self.signature == "void(Type)":
            self.proto = methodProto_Void_Type
        if self.signature == "void(Int)":
            self.proto = methodProto_Void_Int
        if self.signature == "void(Float)":
            self.proto = methodProto_Void_Float
        if self.signature == "void(Double)":
            self.proto = methodProto_Void_Double
        if self.signature == "void(String)":
            self.proto = methodProto_Void_String
        if self.signature == "VectorI(void)":
            self.proto = methodProto_VectorI_Void
            self.meth = "METH_NOARGS"
        if self.signature == "VectorF(void)":
            self.proto = methodProto_VectorF_Void
            self.meth = "METH_NOARGS"
        if self.signature == "VectorD(void)":
            self.proto = methodProto_VectorD_Void
            self.meth = "METH_NOARGS"
        if self.signature == "void(void)":
            self.proto = methodProto_Void_Void
            self.meth = "METH_NOARGS"
        if self.proto == "":
            print("ERROR: unable to find proto for " + self.signature)
            sys.exit(1)
        r = self.prepareMethod()
        return r

    def getDefinition(self, classWrapper):
        s = methodDefProto
        s = s.replace("$CLASS", classWrapper.name)
        s = s.replace("$METHOD", self.name)
        s = s.replace("$METH_FLAG", self.meth)
        return s

    def prepareMethod(self):
        r = self.proto
        r = r.replace("$CLASS", self.classWrapper.name)
        r = r.replace("$CODE", self.code)
        r = r.replace("$METHOD", self.name)
        r = r.replace("$TYPE", self.argType)
        return r


class MemberWrapper:
    def __init__(self):
        self.name = "noname"
        self.type = "T_INT"
        self.offset = 0
        self.doc = "nodoc"

    def eval(self, classWrapper):
        s = memberProto
        s = s.replace("$CLASS", classWrapper.name)
        s = s.replace("$NAME", self.name)
        s = s.replace("$TYPE", self.type)
        s = s.replace("$OFFSET", self.offset)
        s = s.replace("$DOC", self.doc)
        return s


class ClassWrapper:
    def __init__(self):
        self.name = "SomeClass"
        self.methods = []
        self.members = []

    def addMethod(self, name, signature, code, argType = ""):
        mw = MethodWrapper()
        mw.name = name
        mw.signature = signature
        mw.code = code
        mw.argType = argType
        self.methods.append(mw)
        return mw

    def addMember(self, name, type_, offset, doc = "no doc"):
        mw = MemberWrapper()
        mw.name = name
        mw.type = type_
        mw.offset = offset
        mw.doc = doc
        self.members.append(mw)

    def getClasses(self):
        s = classProto
        s = s.replace("$CLASS", self.name)
        methods = ""
        definitions = ""
        for mw in self.methods:
            methods += mw.getBody(self)
            definitions += mw.getDefinition(self)
            lastMethod = False
            if mw == self.methods[-1]:
                lastMethod = True
            if not lastMethod:
                definitions += "\n"
        members = ""
        for mw in self.members:
            members += mw.eval(self)
        s = s.replace("$METHODS", methods)
        s = s.replace("$DEFINITIONS", definitions)
        s = s.replace("$MEMBERS", members)
        return s

    def getAddTypePy2(self):
        s = addTypeProtoPy2
        s = s.replace("$CLASS", self.name)
        return s

    def getAddTypePy3(self):
        s = addTypeProtoPy3
        s = s.replace("$CLASS", self.name)
        return s

    def getInitObjects(self):
        s = addObjectProto
        s = s.replace("$CLASS", self.name)
        return s


class ModuleWrapper:
    def __init__(self, name):
        self.name = name
        self.classes = []
        self.headers = []
        self.doc = "Modlule " + name + " doc."

    def addClass(self, name):
        cw = ClassWrapper()
        cw.name = name
        self.classes.append(cw)
        return cw

    def addHeader(self, header):
        self.headers.append(header)

    def emit(self):
        s = moduleProto
        classes = self.getClasses()
        typesPy2 = self.getAddTypesPy2()
        typesPy3 = self.getAddTypesPy3()
        objects = self.getInitObjects()
        headers = self.getHeaders()
        s = s.replace("$CLASSES", classes)
        s = s.replace("$INIT_TYPES_PY2", typesPy2)
        s = s.replace("$INIT_TYPES_PY3", typesPy3)
        s = s.replace("$ADD_OBJECTS", objects)
        s = s.replace("$MODULE_DOC$", self.doc)
        s = s.replace("$MODULE", self.name)
        s = s.replace("$HEADERS", headers)
        return s

    def getClasses(self):
        s = ""
        for cw in self.classes:
            s += cw.getClasses()
        return s

    def getAddTypesPy2(self):
        s = ""
        for cw in self.classes:
            s += cw.getAddTypePy2()
        return s

    def getAddTypesPy3(self):
        s = ""
        for cw in self.classes:
            s += cw.getAddTypePy3()
        return s

    def getInitObjects(self):
        s = ""
        for cw in self.classes:
            s += cw.getInitObjects()
        return s

    def getHeaders(self):
        s = ""
        for header in self.headers:
            s += header + "\n"
        return s
