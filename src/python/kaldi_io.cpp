/*
 * kaldi-io.cpp
 *
 *  Created on: Jul 29, 2014
 *      Author: chorows
 */

#include "Python.h"

#include <boost/shared_ptr.hpp>

#include <boost/python.hpp>
#include <boost/python/operators.hpp>

#include <util/kaldi-io.h>
#include <util/kaldi-table.h>

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
        bp::object py_string = PW()->dumps(t);
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

template<class T>
const T& get_self_ref(const T& t) {
  return t;
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
          .def("__exit__", &Reader::Close)
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
          .def("__exit__", &Reader::Close)
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
          .def("__exit__",&Writer::Close)
          ;
      }
};

BOOST_PYTHON_MODULE(kaldi_io)
{
  RandomAccessWrapper<kaldi::RandomAccessTableReader<PyObjectHolder> >("RandomAccessTableReader", bp::init<std::string>())
      ;

  RandomAccessWrapper<kaldi::RandomAccessTableReaderMapped<PyObjectHolder> >("RandomAccessTableReaderMapped",
                                                                             bp::init<std::string, std::string>())
      ;

  SequentialReaderWrapper<kaldi::SequentialTableReader<PyObjectHolder> >("SequentialTableReader",bp::init<std::string>())
      ;

  WriterWrapper<kaldi::TableWriter<PyObjectHolder> >("TableWriter", bp::init<std::string>())
      ;
}
