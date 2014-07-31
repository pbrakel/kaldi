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
#include <boost/python/stl_iterator.hpp>

#include <util/kaldi-io.h>
#include <util/kaldi-table.h>
#include <matrix/kaldi-matrix.h>
#include <matrix/kaldi-vector.h>
#include <util/table-types.h>

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
    if (PyArray_NDIM(arr_)!=2) {
      PyErr_SetString(PyExc_TypeError, "Can wrap only matrices (2D arrays)");
      bp::throw_error_already_set();
    }
    if (PyArray_TYPE(arr)!=get_np_type<Real>()) {
      PyErr_SetString(PyExc_TypeError, "Wrong array dtype");
      bp::throw_error_already_set();
    }
    npy_intp* dims = PyArray_DIMS(arr_);
    npy_intp* strides = PyArray_STRIDES(arr_);
    if (strides[1]!=sizeof(Real)) {
      PyErr_SetString(PyExc_TypeError, "Wrong array column stride");
      bp::throw_error_already_set();
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
    if (PyArray_NDIM(arr_)!=1) {
      PyErr_SetString(PyExc_TypeError, "Can wrap only vectors (1D arrays)");
      bp::throw_error_already_set();
    }
    if (PyArray_TYPE(arr)!=get_np_type<Real>()) {
      PyErr_SetString(PyExc_TypeError, "Wrong array dtype");
            bp::throw_error_already_set();
    }
    npy_intp* dims = PyArray_DIMS(arr_);
    npy_intp* strides = PyArray_STRIDES(arr_);
    if (strides[0]!=sizeof(Real)) {
      PyErr_SetString(PyExc_TypeError, "Wrong array column stride");
      bp::throw_error_already_set();
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

template<class Real>
struct MatrixToNdArrayConverter {
  typedef kaldi::KaldiObjectHolder<kaldi::Matrix<Real> > HR;
  typedef kaldi::KaldiObjectHolder<NpWrapperMatrix<Real> > HW;

  static inline bp::object kaldi_to_python(const kaldi::Matrix<Real>& mat) {
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
    return arr;
  }

  static inline NpWrapperMatrix<Real>* python_to_kaldi(bp::object o) {
    PyObject* raw_arr = PyArray_FromAny(o.ptr(),PyArray_DescrFromType(get_np_type<Real>()), 2, 2, NPY_C_CONTIGUOUS | NPY_FORCECAST, NULL);
    //why does this fail: bp::object arr(bp::handle<>(raw_arr));
    bp::object arr=bp::object(bp::handle<>(raw_arr));
    return new NpWrapperMatrix<Real>((PyArrayObject*)arr.ptr());
  }
};

template<class Real>
struct VectorToNdArrayConverter {
  typedef kaldi::KaldiObjectHolder<kaldi::Vector<Real> > HR;
  typedef kaldi::KaldiObjectHolder<NpWrapperVector<Real> > HW;

  static inline bp::object kaldi_to_python(const kaldi::Vector<Real>& vec) {
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
    return arr;
  }

  static inline NpWrapperVector<Real>* python_to_kaldi(bp::object o) {
    PyObject* raw_arr = PyArray_FromAny(o.ptr(),PyArray_DescrFromType(get_np_type<Real>()), 1, 1, NPY_C_CONTIGUOUS | NPY_FORCECAST, NULL);
    //why does this fail: bp::object arr(bp::handle<>(raw_arr));
    bp::object arr=bp::object(bp::handle<>(raw_arr));
    return new NpWrapperVector<Real>((PyArrayObject*)arr.ptr());
  }
};

template<class T>
struct VectorToListConverter {
  typedef kaldi::BasicVectorHolder<T> HW,HR;

  static inline bp::object kaldi_to_python(const std::vector<T>& vec) {
    boost::python::list l;
    for (size_t i = 0; i < vec.size(); i++)
      l.append(vec[i]);
    return l;
  }

  static inline std::vector<T> * python_to_kaldi(bp::object o) {
    bp::stl_input_iterator<T> begin(o);
    bp::stl_input_iterator<T> end;
    std::vector<T> *v = new std::vector<T>();
    v->insert(v->end(), begin, end);
    return v;
  }
};

//template<class T>
//struct VectorVectorToListListConverter {
//  typedef kaldi::BasicVectorVectorHolder<T> HW,HR;
//
//  static inline bp::object kaldi_to_python(const std::vector<std::vector<T> >& vec) {
//    boost::python::list l;
//    for (size_t i = 0; i < vec.size(); i++)
//      l.append(vec[i]);
//    return l;
//  }
//
//  static inline std::vector<std::vector<T> > * python_to_kaldi(bp::object o) {
//    bp::stl_input_iterator<T> begin(o);
//    bp::stl_input_iterator<T> end;
//    std::vector<T> v;
//    v.insert(v.end(), begin, end);
//    return v;
//  }
//};

template<class Converter>
class PythonToKaldiHolder {
 public:
  typedef bp::object T;
  typedef typename Converter::HR HR;
  typedef typename Converter::HW HW;

  PythonToKaldiHolder() : h_() {
  }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    try {
      auto_ptr<typename HW::T> obj(Converter::python_to_kaldi(t));
      return HW::Write(os, binary, (*obj));
    } catch (std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object";
      if (!kaldi::IsKaldiError(e.what())) {std::cerr << e.what();}
      return false;
    }
  }

  bool Read(std::istream &is) {
    if (!h_.Read(is))
      return false;
    t_ = Converter::kaldi_to_python(h_.Value());
    return true;
  }

  static bool IsReadInBinary() {return true;}

  const T &Value() const {return t_;}  // if t is a pointer, would return *t_;

  void Clear() {}

  ~PythonToKaldiHolder() {}

private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(PythonToKaldiHolder);
  HR h_;
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


PyObject* KALDI_BASE_FLOAT() {
  return (PyObject*)PyArray_DescrFromType(get_np_type<kaldi::BaseFloat>());
}

BOOST_PYTHON_MODULE(kaldi_io_internal)
{
  import_array();

  bp::def("KALDI_BASE_FLOAT", &KALDI_BASE_FLOAT);

  //Python objects
  RandomAccessWrapper<kaldi::RandomAccessTableReader<PyObjectHolder> >("RandomAccessPythonReader", bp::init<std::string>());
  RandomAccessWrapper<kaldi::RandomAccessTableReaderMapped<PyObjectHolder> >("RandomAccessPythonReaderMapped", bp::init<std::string, std::string>());
  SequentialReaderWrapper<kaldi::SequentialTableReader<PyObjectHolder> >("SequentialPythonReader",bp::init<std::string>());
  WriterWrapper<kaldi::TableWriter<PyObjectHolder> >("PythonWriter", bp::init<std::string>());

  //Matrices as NdArrays
  RandomAccessWrapper<kaldi::RandomAccessTableReader<PythonToKaldiHolder<MatrixToNdArrayConverter<double> > > >("RandomAccessDoubleMatrixReader", bp::init<std::string>());
  RandomAccessWrapper<kaldi::RandomAccessTableReaderMapped<PythonToKaldiHolder<MatrixToNdArrayConverter<double> > > >("RandomAccessDoubleMatrixMapped",bp::init<std::string, std::string>());
  SequentialReaderWrapper<kaldi::SequentialTableReader<PythonToKaldiHolder<MatrixToNdArrayConverter<double> > > >("SequentialDoubleMatrixReader",bp::init<std::string>());
  WriterWrapper<kaldi::TableWriter<PythonToKaldiHolder<MatrixToNdArrayConverter<double> > > >("DoubleMatrixWriter", bp::init<std::string>());

  RandomAccessWrapper<kaldi::RandomAccessTableReader<PythonToKaldiHolder<MatrixToNdArrayConverter<float> > > >("RandomAccessFloatMatrixReader", bp::init<std::string>());
  RandomAccessWrapper<kaldi::RandomAccessTableReaderMapped<PythonToKaldiHolder<MatrixToNdArrayConverter<float> > > >("RandomAccessFloatMatrixMapped",bp::init<std::string, std::string>());
  SequentialReaderWrapper<kaldi::SequentialTableReader<PythonToKaldiHolder<MatrixToNdArrayConverter<float> > > >("SequentialFloatMatrixReader",bp::init<std::string>());
  WriterWrapper<kaldi::TableWriter<PythonToKaldiHolder<MatrixToNdArrayConverter<float> > > >("FloatMatrixWriter", bp::init<std::string>());

  //Vectors as NdArrays
  RandomAccessWrapper<kaldi::RandomAccessTableReader<PythonToKaldiHolder<VectorToNdArrayConverter<double> > > >("RandomAccessDoubleVectorReader", bp::init<std::string>());
  RandomAccessWrapper<kaldi::RandomAccessTableReaderMapped<PythonToKaldiHolder<VectorToNdArrayConverter<double> > > >("RandomAccessDoubleVectorReaderMapped",bp::init<std::string, std::string>());
  SequentialReaderWrapper<kaldi::SequentialTableReader<PythonToKaldiHolder<VectorToNdArrayConverter<double> > > >("SequentialDoubleVectorReader",bp::init<std::string>());
  WriterWrapper<kaldi::TableWriter<PythonToKaldiHolder<VectorToNdArrayConverter<double> > > >("DoubleVectorWriter", bp::init<std::string>());

  RandomAccessWrapper<kaldi::RandomAccessTableReader<PythonToKaldiHolder<VectorToNdArrayConverter<float> > > >("RandomAccessFloatVectorReader", bp::init<std::string>());
  RandomAccessWrapper<kaldi::RandomAccessTableReaderMapped<PythonToKaldiHolder<VectorToNdArrayConverter<float> > > >("RandomAccessFloatVectorReaderMapped",bp::init<std::string, std::string>());
  SequentialReaderWrapper<kaldi::SequentialTableReader<PythonToKaldiHolder<VectorToNdArrayConverter<float> > > >("SequentialFloatVectorReader",bp::init<std::string>());
  WriterWrapper<kaldi::TableWriter<PythonToKaldiHolder<VectorToNdArrayConverter<float> > > >("FloatVectorWriter", bp::init<std::string>());

  //Integers
  RandomAccessWrapper<kaldi::RandomAccessInt32Reader >("RandomAccessInt32Reader", bp::init<std::string>());
  SequentialReaderWrapper<kaldi::SequentialInt32Reader >("SequentialInt32Reader",bp::init<std::string>());
  WriterWrapper<kaldi::Int32Writer >("Int32Writer", bp::init<std::string>());

  // std::vector<int32>
  RandomAccessWrapper<kaldi::RandomAccessTableReader<PythonToKaldiHolder<VectorToListConverter<kaldi::int32> > > >("RandomAccessInt32VectorReader", bp::init<std::string>());
  SequentialReaderWrapper<kaldi::SequentialTableReader<PythonToKaldiHolder<VectorToListConverter<kaldi::int32> > > >("SequentialInt32VectorReader",bp::init<std::string>());
  WriterWrapper<kaldi::TableWriter<PythonToKaldiHolder<VectorToListConverter<kaldi::int32> > > >("Int32VectorWriter", bp::init<std::string>());
}
