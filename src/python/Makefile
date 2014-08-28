
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = 


OBJFILES = 


ADDLIBS = ../lm/kaldi-lm.a ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
	      ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a \
          ../base/kaldi-base.a  ../thread/kaldi-thread.a

TESTFILES =

PYLIB = /opt/lisa/os/epd-7.1.2/bin
PYINC = /opt/lisa/os/epd-7.1.2/include/python2.7

PYLIBS = kaldi_io_internal.so kaldi_sequence_loss.so

#include ../makefiles/default_rules.mk

%.so: %.cpp
		g++ -shared -o $@ -Wall -fPIC -I$(PYINC) $(CXXFLAGS) $< $(ADDLIBS) $(LDFLAGS) -L$(PYLIB) $(LOADLIBES) $(LDLIBS) -lpython2.7 -lboost_python -lboost_system 

clean:
	-rm -f *.o *.a *.so $(TESTFILES) $(BINFILES) $(TESTOUTPUTS) tmp* *.tmp

depend:
	-$(CXX) -M $(CXXFLAGS) *.cpp > .depend.mk  

all: $(PYLIBS)
