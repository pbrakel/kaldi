#!/usr/bin/env python

import os
import os.path

if 'KALDI_ROOT' in os.environ:
    print os.environ['KALDI_ROOT']
else:
    print os.path.abspath(os.path.join(os.path.realpath(os.path.dirname(__file__)),'..','..','..','..'))
