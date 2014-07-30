/*
 * kaldi-io.cpp
 *
 *  Created on: Jul 29, 2014
 *      Author: chorows
 */

extern "C" {
#include "Python.h"
#include "numpy/arrayobject.h"
}

#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>

#include <boost/python.hpp>
#include <boost/python/operators.hpp>

#include <util/kaldi-io.h>
#include <util/kaldi-table.h>
#include <matrix/kaldi-matrix.h>
#include <matrix/kaldi-vector.h>

using namespace std;

namespace bp = boost::python;

//keep a copy of the cPickle module in cache
struct PickleWrapper {
  PickleWrapper() {
    bp::object pickle = bp::import("cPickle");
    loads = pickle.attr("loads");
    dumps = pickle.attr("dumps");
  }

  bp::object loads, dumps;
};

//
// Holder for Python objects.
//
// In binary model uses Pickle to dump, the object is written as dump_length, pickled_string
// In text mode uses repr/eval (only single line), which works OK for simple types - lists, tuples, ints, but may fail for large arrays (as repr skips elemets for ndarray).
//
class PyObjectHolder {
 public:
  typedef bp::object T;

  PyObjectHolder() {
  }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    kaldi::InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      if (binary) {  //pickle the object
        bp::object py_string = PW()->dumps(t,-1);
        int len = bp::extract<int>(py_string.attr("__len__")());
        const char* string = bp::extract<const char*>(py_string);
        kaldi::WriteBasicType(os, true, len);
        os.write(string, len);
      } else {  //use repr
        PyObject* repr = PyObject_Repr(t.ptr());
        os << PyString_AsString(repr) << '\n';
        Py_DECREF(repr);
      }
      return os.good();
    } catch (const std::exception &e) {
      KALDI_WARN<< "Exception caught writing Table object: " << e.what();
      if (!kaldi::IsKaldiError(e.what())) {std::cerr << e.what();}
      return false;  // Write failure.
    }
  }

  bool Read(std::istream &is) {
    bool is_binary;
    if (!kaldi::InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary header\n";
      return false;
    }
    try {
      if (is_binary) {
        int len;
        kaldi::ReadBasicType(is, true, &len);
        std::auto_ptr<char> buf(new char[len]);
        is.read(buf.get(), len);
        bp::str py_string(buf.get(), len);
        t_ = PW()->loads(py_string);
      } else {
        std::string line;
        std::getline(is, line);
        bp::str repr(line);
        t_ = bp::eval(repr);
      }
      return true;
    } catch (std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object";
      if (!kaldi::IsKaldiError(e.what())) {std::cerr << e.what();}
      return false;
    }
  }

  static bool IsReadInBinary() {return true;}

  const T &Value() const {return t_;}  // if t is a pointer, would return *t_;

  void Clear() {}

  ~PyObjectHolder() {}

private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(PyObjectHolder);
  T t_;  // t_ may alternatively be of type T*.
  static PickleWrapper *PW_;
  static PickleWrapper * PW() {
    if (!PW_) {
      PW_ = new PickleWrapper();
    }
    return PW_;
  }
};

PickleWrapper * PyObjectHolder::PW_ = 0;

////template<class Real>
////class DataStealingMatrix: public kaldi::Matrix<Real> {
////
//// public:
////  DataStealingMatrix() : kaldi::Matrix<Real>() {}
////
////  Real* steal_data() {
////    Real* ret = this->data_;
////    this->data_= NULL;
////    return ret;
////  }
////};
//
////
//// Helper python object holding a Kaldi matrix.
//// Will free the matrix upon destruction
////
//
//template<class Real>
//struct _MatrixDeallocator {
//  PyObject_HEAD
//  std::auto_ptr<kaldi::Matrix<Real>> mat;
//};

//Helper to get proper np type
template <class Real>
int get_np_type() {
  //BOOST_STATIC_ASSERT_MSG(false, "Call one of the explicitly instantiated templates for float or double.");
  KALDI_ERR << "Call one of the explicitly instantiated templates for float or double.";
  return -1;
}

template <>
int get_np_type<double>() {
  return NPY_DOUBLE;
}

template <>
int get_np_type<float>() {
  return NPY_FLOAT;
}

template<typename Real>
class NpWrapperMatrix : public kaldi::MatrixBase<Real> {
 public:
  NpWrapperMatrix(PyArrayObject* arr)
      : kaldi::MatrixBase<Real>(),
        arr_(arr) {
    if (!PyArray_NDIM(arr_)==2) {
      KALDI_ERR << "Can wrap only matrices (2D arrays)";
    }
    if (!PyArray_TYPE(arr)==get_np_type<Real>()) {
      KALDI_ERR << "Wrong array dtype";
    }
    npy_intp* dims = PyArray_DIMS(arr_);
    npy_intp* strides = PyArray_STRIDES(arr_);
    if (strides[1]!=sizeof(Real)) {
      KALDI_ERR << "Wrong array column stride";
    }
    Py_INCREF(arr_);
    //why do we have to use this-> in here??
    this->data_ = (Real*)PyArray_DATA(arr);
    this->num_rows_ = dims[0];
    this->num_cols_ = dims[1];
    this->stride_ = strides[0]/sizeof(Real);
  }

  ~NpWrapperMatrix() {
    Py_DECREF(arr_);
  }

 protected:
  PyArrayObject* arr_;
};

template<typename Real>
class NpWrapperVector : public kaldi::VectorBase<Real> {
 public:
  NpWrapperVector(PyArrayObject* arr)
      : kaldi::VectorBase<Real>(),
        arr_(arr) {
    if (!PyArray_NDIM(arr_)==1) {
      KALDI_ERR << "Can wrap only vectors (1D arrays)";
    }
    if (!PyArray_TYPE(arr)==get_np_type<Real>()) {
      KALDI_ERR << "Wrong array dtype";
    }
    npy_intp* dims = PyArray_DIMS(arr_);
    npy_intp* strides = PyArray_STRIDES(arr_);
    if (strides[0]!=sizeof(Real)) {
      KALDI_ERR << "Wrong array column stride";
    }
    Py_INCREF(arr_);
    //why do we have to use this-> in here??
    this->data_ = (Real*)PyArray_DATA(arr);
    this->dim_ = dims[0];
  }

  ~NpWrapperVector() {
    Py_DECREF(arr_);
  }

 protected:
  PyArrayObject* arr_;
};

//
// Read kaldi matrices as NDArrays and store NDArrays ads Kaldi matrices
//
//
template<class Real>
class NdArrayAsMatrixCopyingHolder {
 public:
  typedef bp::object T;

  NdArrayAsMatrixCopyingHolder() {
  }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    kaldi::InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      NpWrapperMatrix<Real> arr_wrap((PyArrayObject*)t.ptr());
      arr_wrap.Write(os,binary);
      return os.good();
    } catch (const std::exception &e) {
      KALDI_WARN<< "Exception caught writing Table object: " << e.what();
      if (!kaldi::IsKaldiError(e.what())) {std::cerr << e.what();}
      return false;  // Write failure.
    }
  }

  bool Read(std::istream &is) {
    bool is_binary;
    if (!kaldi::InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary header\n";
      return false;
    }
    try {
      kaldi::Matrix<Real> mat;
      mat.Read(is, is_binary);
      npy_intp dims[2];
      dims[0] = mat.NumRows();
      dims[1] = mat.NumCols();
      int nd = 2;
      int arr_type = get_np_type<Real>();
      PyObject* ao = PyArray_SimpleNew(nd, dims, arr_type);
      bp::object arr=bp::object(bp::handle<>(
          ao
          ));
      NpWrapperMatrix<Real> arr_wrap((PyArrayObject*)arr.ptr());
      arr_wrap.CopyFromMat(mat);
      t_ = arr;
      return true;
    } catch (std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object";
      if (!kaldi::IsKaldiError(e.what())) {std::cerr << e.what();}
      return false;
    }
  }

  static bool IsReadInBinary() {return true;}

  const T &Value() const {return t_;}  // if t is a pointer, would return *t_;

  void Clear() {}

  ~NdArrayAsMatrixCopyingHolder() {}

private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NdArrayAsMatrixCopyingHolder);
  T t_;  // t_ may alternatively be of type T*.
};

//
// Read kaldi vectors as NDArrays and store NDArrays ads Kaldi vectors
//
//
template<class Real>
class NdArrayAsVectorCopyingHolder {
 public:
  typedef bp::object T;

  NdArrayAsVectorCopyingHolder() {
  }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    kaldi::InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      NpWrapperVector<Real> arr_wrap((PyArrayObject*)t.ptr());
      arr_wrap.Write(os,binary);
      return os.good();
    } catch (const std::exception &e) {
      KALDI_WARN<< "Exception caught writing Table object: " << e.what();
      if (!kaldi::IsKaldiError(e.what())) {std::cerr << e.what();}
      return false;  // Write failure.
    }
  }

  bool Read(std::istream &is) {
    bool is_binary;
    if (!kaldi::InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary header\n";
      return false;
    }
    try {
      kaldi::Vector<Real> vec;
      vec.Read(is, is_binary);
      npy_intp dims[1];
      dims[0] = vec.Dim();
      int nd = 1;
      int arr_type = get_np_type<Real>();
      PyObject* ao = PyArray_SimpleNew(nd, dims, arr_type);
      bp::object arr=bp::object(bp::handle<>(
          ao
          ));
      NpWrapperVector<Real> vec_wrap((PyArrayObject*)arr.ptr());
      vec_wrap.CopyFromVec(vec);
      t_ = arr;
      return true;
    } catch (std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object";
      if (!kaldi::IsKaldiError(e.what())) {std::cerr << e.what();}
      return false;
    }
  }

  static bool IsReadInBinary() {return true;}

  const T &Value() const {return t_;}  // if t is a pointer, would return *t_;

  void Clear() {}

  ~NdArrayAsVectorCopyingHolder() {}

private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NdArrayAsVectorCopyingHolder);
  T t_;  // t_ may alternatively be of type T*.
};


template<class T>
const T& get_self_ref(const T& t) {
  return t;
}

template<class T>
void exit(T& t, const bp::object& type,
              const bp::object& value, const bp::object& traceback) {
  t.Close();
}

template<class T>
bp::object sequential_reader_next(T& reader) {
  if (reader.Done()) {
    PyErr_SetString(PyExc_StopIteration, "No more data.");
    bp::throw_error_already_set();
  }
  //if not done, extract the contents
  bp::str key(reader.Key());
  bp::object val(reader.Value());
  //move the reading head, the contents will be read with the next call to next!
  reader.Next();
  return bp::make_tuple(key,val);
}

template <class Reader>
class RandomAccessWrapper: public bp::class_<Reader> {
public:
  template <class DerivedT>
  inline RandomAccessWrapper(char const* name, bp::init_base<DerivedT> const& i)
          : bp::class_<Reader>(name, i) {
          (*this)
          .def("close", &Reader::Close)
          .def("is_open", &Reader::IsOpen)
          .def("__contains__", &Reader::HasKey)
          .def("has_key", &Reader::HasKey)
          .def("__getitem__", &Reader::Value,
               bp::return_value_policy<bp::copy_const_reference>())
          .def("value", &Reader::Value,
               bp::return_value_policy<bp::copy_const_reference>())
          .def("__enter__", &get_self_ref<Reader>,
               bp::return_internal_reference<1>())
          .def("__exit__", &exit<Reader>)
          ;
      }
};

template <class Reader>
class SequentialReaderWrapper: public bp::class_<Reader> {
public:
  template <class DerivedT>
  inline SequentialReaderWrapper(char const* name, bp::init_base<DerivedT> const& i)
          : bp::class_<Reader>(name, i) {
          (*this)
          .def("close", &Reader::Close)
          .def("is_open", &Reader::IsOpen)
          .def("__enter__", &get_self_ref<Reader>,
               bp::return_internal_reference<1>())
          .def("__iter__", &get_self_ref<Reader>,
               bp::return_internal_reference<1>())
          .def("next", sequential_reader_next<Reader>)
          .def("__exit__", &exit<Reader>)
          .def("done", &Reader::Done)
          .def("_kaldi_value", &Reader::Value,
                     bp::return_value_policy<bp::copy_const_reference>())
          .def("_kaldi_next", &Reader::Next)
          .def("_kaldi_key", &Reader::Key)
          ;
      }
};

template <class Writer>
class WriterWrapper: public bp::class_<Writer> {
public:
  template <class DerivedT>
  inline WriterWrapper(char const* name, bp::init_base<DerivedT> const& i)
          : bp::class_<Writer>(name, i) {
          (*this)
          .def("close", &Writer::Close)
          .def("is_open", &Writer::IsOpen)
          .def("flush", &Writer::Flush)
          .def("write", &Writer::Write)
          .def("__setitem__", &Writer::Write)
          .def("__enter__", &get_self_ref<Writer>,
               bp::return_internal_reference<1>())
          .def("__exit__",&exit<Writer>)
          ;
      }
};

BOOST_PYTHON_MODULE(kaldi_io)
{
  import_array();
  RandomAccessWrapper<kaldi::RandomAccessTableReader<PyObjectHolder> >("RandomAccessTableReader", bp::init<std::string>())
      ;

  RandomAccessWrapper<kaldi::RandomAccessTableReaderMapped<PyObjectHolder> >("RandomAccessTableReaderMapped",
                                                                             bp::init<std::string, std::string>())
      ;

  SequentialReaderWrapper<kaldi::SequentialTableReader<PyObjectHolder> >("SequentialTableReader",bp::init<std::string>())
      ;

  WriterWrapper<kaldi::TableWriter<PyObjectHolder> >("TableWriter", bp::init<std::string>())
      ;

  RandomAccessWrapper<kaldi::RandomAccessTableReader<NdArrayAsMatrixCopyingHolder<kaldi::BaseFloat> > >("NpMatrixRandomAccessTableReader", bp::init<std::string>())
      ;

  RandomAccessWrapper<kaldi::RandomAccessTableReaderMapped<NdArrayAsMatrixCopyingHolder<kaldi::BaseFloat> > >("NpMatrixRandomAccessTableReaderMapped",
                                                                             bp::init<std::string, std::string>())
      ;

  SequentialReaderWrapper<kaldi::SequentialTableReader<NdArrayAsMatrixCopyingHolder<kaldi::BaseFloat> > >("NpMatrixSequentialTableReader",bp::init<std::string>())
      ;

  WriterWrapper<kaldi::TableWriter<NdArrayAsMatrixCopyingHolder<kaldi::BaseFloat> > >("NpMatrixTableWriter", bp::init<std::string>())
      ;

  RandomAccessWrapper<kaldi::RandomAccessTableReader<NdArrayAsVectorCopyingHolder<kaldi::BaseFloat> > >("NpVectorRandomAccessTableReader", bp::init<std::string>())
      ;

  RandomAccessWrapper<kaldi::RandomAccessTableReaderMapped<NdArrayAsVectorCopyingHolder<kaldi::BaseFloat> > >("NpVectorRandomAccessTableReaderMapped",
                                                                             bp::init<std::string, std::string>())
      ;

  SequentialReaderWrapper<kaldi::SequentialTableReader<NdArrayAsVectorCopyingHolder<kaldi::BaseFloat> > >("NpVectorSequentialTableReader",bp::init<std::string>())
      ;

  WriterWrapper<kaldi::TableWriter<NdArrayAsVectorCopyingHolder<kaldi::BaseFloat> > >("NpVectorTableWriter", bp::init<std::string>())
      ;
}
