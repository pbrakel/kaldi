// bin/show-transitions.cc
// Copyright 2009-2011 Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Generate transition labels suitable to use as isymbol for fstprint\n"
            "Usage:  write-transition-labels [opts] phones-symbol-table transition/model-file [disambiguation_symbols]\n"
            "e.g.: \n"
            " write-transition-labels phones.txt 1.mdl\n";

    ParseOptions po(usage);

    bool writeHMMStates = true, writePdfIds = true, writeDest = true;

    po.Register("write-hmm", &writeHMMStates, "Write HMM states");
    po.Register("write-pdf-id", &writePdfIds, "Write Pdf-Ids states");
    po.Register("write-dest", &writeDest,
                "Write destination Transaction-states");

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string phones_symtab_filename = po.GetArg(1),
        transition_model_filename = po.GetArg(2),
        disambiguation_filename = po.GetArg(3);

    fst::SymbolTable *syms = fst::SymbolTable::ReadText(phones_symtab_filename);
    if (!syms)
      KALDI_ERR<< "Could not read symbol table from file "
      << phones_symtab_filename;
    std::vector<std::string> names(syms->NumSymbols());
    for (size_t i = 0; i < syms->NumSymbols(); i++)
      names[i] = syms->Find(i);

    TransitionModel trans_model;
    ReadKaldiObject(transition_model_filename, &trans_model);

    cout << "<eps> 0" << endl;
    for (int32 tid = 1; tid <= trans_model.NumTransitionIds(); tid++) {
      cout << tid;
      cout << '<' << trans_model.TransitionIdToTransitionState(tid) << ":";
      cout << names[trans_model.TransitionIdToPhone(tid)];
      if (writeHMMStates) {
        cout << "_H" << trans_model.TransitionIdToHmmState(tid);
        if (writeDest) {
          cout << "->";
          //<< trans_model.TRa
          int32 trans_index = trans_model.TransitionIdToTransitionIndex(tid);
          int32 next_state = trans_model.GetTopo()
              .TopologyForPhone(trans_model.TransitionIdToPhone(tid)
              )[trans_model.TransitionIdToHmmState(tid)]
                .transitions[trans_index].first;
          cout << next_state;
        }
      }

      if (writePdfIds) {
        cout << "_P" << trans_model.TransitionIdToPdf(tid);
      }
      cout << "> " << tid << endl;
    }

    if (disambiguation_filename != "") {
      bool binary_in;
      Input ki(disambiguation_filename, &binary_in);
      std::istream &in = ki.Stream();
      int32 symbol_id;
      while (in >> symbol_id) {
        //ReadBasicType(in, binary_in, &symbol_id);
        cout << symbol_id << ' ' << symbol_id << endl;
      }

    }



    delete syms;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

