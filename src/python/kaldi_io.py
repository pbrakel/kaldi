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
