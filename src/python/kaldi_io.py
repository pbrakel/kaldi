'''
Created on Jul 31, 2014

@author: chorows
'''


import numpy as np
from kaldi_io_internal import *

if KALDI_BASE_FLOAT()==np.float64:
    RandomAccessBaseFloatMatrixReader = RandomAccessDoubleMatrixReader
    RandomAccessBaseFloatMatrixMapped = RandomAccessDoubleMatrixMapped
    SequentialBaseFloatMatrixReader = SequentialDoubleMatrixReader
    BaseFloatMatrixWriter = DoubleMatrixWriter
    
    RandomAccessBaseFloatVectorReader = RandomAccessDoubleVectorReader
    RandomAccessBaseFloatVectorReaderMapped = RandomAccessDoubleVectorReaderMapped
    SequentialBaseFloatVectorReader = SequentialDoubleVectorReader
    BaseFloatVectorWriter = DoubleVectorWriter
    
if KALDI_BASE_FLOAT()==np.float32:
    RandomAccessBaseFloatMatrixReader = RandomAccessFloatMatrixReader
    RandomAccessBaseFloatMatrixMapped = RandomAccessFloatMatrixMapped
    SequentialBaseFloatMatrixReader = SequentialFloatMatrixReader
    BaseFloatMatrixWriter = FloatMatrixWriter
    
    RandomAccessBaseFloatVectorReader = RandomAccessFloatVectorReader
    RandomAccessBaseFloatVectorReaderMapped = RandomAccessFloatVectorReaderMapped
    SequentialBaseFloatVectorReader = SequentialFloatVectorReader
    BaseFloatVectorWriter = FloatVectorWriter

def get_io_for_dtype(access, dtype, element=''):
    '''
    Get a writer or reader for the given dtype. eg:
    get_io_for_dtype('Sequential',np.float32,'MatrixReader')
    get_io_for_dtype('float32,'MatrixWriter')
    '''
    if element=='': #assume we want a writer
        access, dtype,element = '',access,dtype
    dtypemap = {np.int32:'Int32',
                np.float32:'Float',
                np.float64:'Double',
                'float32':'Float',
                'float64':'Double'}
    dtype = dtypemap[dtype]
    return globals()[access + dtype + element] 

class _Transformed(object):
    def __init__(self, reader, transform_function, **kwargs):
        super(_Transformed, self).__init__(**kwargs)
        self.reader=reader
        self.transform_function = transform_function
    
    def __getattr__(self, attr):
        return getattr(self.reader,attr)
    
class TransRA(_Transformed):
    def __init__(self, *args, **kwargs):
        super(TransRA, self).__init__(*args, **kwargs)
    
    def value(self, key):
        return self.transform_function(self.reader.value(key))
    
    def __getitem__(self, key):
        return self.value(key)
    
class TransSeq(_Transformed):
    def __init__(self, *args, **kwargs):
        super(TransSeq, self).__init__(*args, **kwargs)
        
    def next(self):
        return self.transform_function(self.reader.next())

    def _kaldi_value(self):
        return self.transform_function(self.reader._kaldi_value())
    