/*
 * sequence_loss.cpp
 *
 *  Created on: Aug 28, 2014
 *      Author: chorows
 */

#include <python/python_wrappers.h>
#include <python/bp_converters.h>
#include <python/kaldi_sequence_loss.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

namespace kaldi {
namespace pylearn2 {

//The actual computation happens here

//copied from nnet-train-mmi-sequential.cc
void LatticeAcousticRescore(const MatrixBase<BaseFloat> &log_like,
                            const TransitionModel &trans_model,
                            const std::vector<int32> &state_times,
                            Lattice *lat) {
  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  KALDI_ASSERT(!state_times.empty());
  std::vector<std::vector<int32> > time_to_state(log_like.NumRows());
  for (size_t i = 0; i < state_times.size(); i++) {
    KALDI_ASSERT(state_times[i] >= 0);
    if (state_times[i] < log_like.NumRows())  // end state may be past this..
      time_to_state[state_times[i]].push_back(i);
    else
      KALDI_ASSERT(state_times[i] == log_like.NumRows()
                   && "There appears to be lattice/feature mismatch.");
  }

  for (int32 t = 0; t < log_like.NumRows(); t++) {
    for (size_t i = 0; i < time_to_state[t].size(); i++) {
      int32 state = time_to_state[t][i];
      for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
          aiter.Next()) {
        LatticeArc arc = aiter.Value();
        int32 trans_id = arc.ilabel;
        if (trans_id != 0) {  // Non-epsilon input label on arc
          int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
          arc.weight.SetValue2(-log_like(t, pdf_id) + arc.weight.Value2());
          aiter.SetValue(arc);
        }
      }
    }
  }
}

//
// This class basically copies nnet-train-mmi-sequential main loop
//
//
class KaldiMMISequenceLoss : public AbstractSequenceLoss {
 public:
  KaldiMMISequenceLoss(const Id2UttMap  & utt_map,
                       BaseFloat _acoustic_scale,
                       BaseFloat _old_acoustic_scale,
                       BaseFloat _lm_scale,
                       const std::string & transition_model_filename,
                       const std::string & den_lat_rspecifier,
                       const std::string & num_ali_rspecifier,
                       bool _drop_frames
  ) : AbstractSequenceLoss(utt_map),
  acoustic_scale(_acoustic_scale), old_acoustic_scale(_old_acoustic_scale), lm_scale(_lm_scale),
  den_lat_reader(den_lat_rspecifier),
  num_ali_reader(num_ali_rspecifier),
  drop_frames(_drop_frames) {
    //ReadKaldiObject(transition_model_filename, &trans_model);

    if (drop_frames) {
      KALDI_LOG << "--drop-frames=true :"
          " we will zero gradient for frames with total den/num mismatch."
          " The mismatch is likely to be caused by missing correct path "
          " from den-lattice due wrong annotation or search error."
          " Leaving such frames out stabilizes the training.";
    }
  }

  double compute_loss_and_diff(const std::string& utt, const MatrixBase<BaseFloat>& frame_log_likelihoods, MatrixBase<BaseFloat>& diff) {
    bool compute_derivative = diff.NumCols() * diff.NumRows()>0;
    if (compute_derivative) {
      KALDI_ASSERT(frame_log_likelihoods.NumRows() == diff.NumRows() && frame_log_likelihoods.NumCols() == diff.NumCols());
      diff.SetZero();
    }

    // get actual dims for this utt and nnet
    int32 num_frames = frame_log_likelihoods.NumRows();

    // 1) get the numerator alignment
    const std::vector<int32> &num_ali = num_ali_reader.Value(utt);
    // check for temporal length of numerator alignments
    if ((int32)num_ali.size() != num_frames) {
      KALDI_WARN << "Numerator alignment has wrong length "
          << num_ali.size() << " vs. "<< num_frames;
      PROCESS_ERROR();
    }

    // 2) get the denominator lattice, preprocess
    Lattice den_lat = den_lat_reader.Value(utt);
    if (den_lat.Start() == -1) {
      KALDI_WARN << "Empty lattice for utt " << utt;
      PROCESS_ERROR();
    }
    if (old_acoustic_scale != 1.0) {
      fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale), &den_lat);
    }
    // optional sort it topologically
    kaldi::uint64 props = den_lat.Properties(fst::kFstProperties, false);
    if (!(props & fst::kTopSorted)) {
      if (fst::TopSort(&den_lat) == false)
        KALDI_ERR << "Cycles detected in lattice.";
    }
    // get the lattice length and times of states
    vector<int32> state_times;
    int32 max_time = kaldi::LatticeStateTimes(den_lat, &state_times);
    // check for temporal length of denominator lattices
    if (max_time != num_frames) {
      KALDI_WARN << "Denominator lattice has wrong length "
          << max_time << " vs. " << num_frames;
      KALDI_WARN << "Empty lattice for utt " << utt;
      PROCESS_ERROR();
    }

    // 3) rescore the latice
    LatticeAcousticRescore(frame_log_likelihoods, trans_model, state_times, &den_lat);
    if (acoustic_scale != 1.0 || lm_scale != 1.0)
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &den_lat);

    // 5) get the posteriors
    kaldi::Posterior post;
    double lat_like; // total likelihood of the lattice
    double lat_ac_like; // acoustic likelihood weighted by posterior.
    lat_like = kaldi::LatticeForwardBackward(den_lat, &post, &lat_ac_like);

    // 7) Calculate the MMI-objective function
    // Calculate the likelihood of correct path from acoustic score,
    // the denominator likelihood is the total likelihood of the lattice.
    double mmi_obj = 0.0;
    double path_ac_like = 0.0;
    for(int32 t=0; t<num_frames; t++) {
      int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
      path_ac_like += frame_log_likelihoods(t,pdf);
    }
    path_ac_like *= acoustic_scale;
    mmi_obj = path_ac_like - lat_like;
    //
    // Note: numerator likelihood does not include graph score,
    // while denominator likelihood contains graph scores.
    // The result is offset at the MMI-objective.
    // However the offset is constant for given alignment,
    // so it is not harmful.

    if (!compute_derivative) {
      return mmi_obj;
    }

    // 6) convert the Posterior to a matrix
    //diff.SetZero(); we do it at the beginning of the function
    for (int32 t = 0; t < post.size(); t++) {
      for (int32 arc = 0; arc < post[t].size(); arc++) {
        int32 pdf = trans_model.TransitionIdToPdf(post[t][arc].first);
        diff(t, pdf) += post[t][arc].second;
      }
    }

    // Sum the den-posteriors under the correct path:
    double post_on_ali = 0.0;
    for(int32 t=0; t<num_frames; t++) {
      int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
      double posterior = diff(t, pdf);
      post_on_ali += posterior;
    }

    // Report
    KALDI_VLOG(1) << "Processed lattice for utterance "
        << " (" << utt << "): found " << den_lat.NumStates()
        << " states and " << fst::NumArcs(den_lat) << " arcs.";

    KALDI_VLOG(1) << "Utterance " << utt << ": Average MMI obj. value = "
        << (mmi_obj/num_frames) << " over " << num_frames
        << " frames."
        << " (Avg. den-posterior on ali " << post_on_ali/num_frames << ")";


    // 7a) Search for the frames with num/den mismatch
    int32 frm_drop = 0;
    std::vector<int32> frm_drop_vec;
    for(int32 t=0; t<num_frames; t++) {
      int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
      double posterior = diff(t, pdf);
      if(posterior < 1e-20) {
        frm_drop++;
        frm_drop_vec.push_back(t);
      }
    }

    // 8) subtract the pdf-Viterbi-path
    for(int32 t=0; t<diff.NumRows(); t++) {
      int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
      diff(t, pdf) -= 1.0;
    }

    // 9) Drop mismatched frames from the training by zeroing the derivative
    int32 num_frm_drop = 0;
    if(drop_frames) {
      for(int32 i=0; i<frm_drop_vec.size(); i++) {
        diff.Row(frm_drop_vec[i]).Set(0.0);
      }
      num_frm_drop += frm_drop;
    }
    // Report the frame dropping
    if (frm_drop > 0) {
      std::stringstream ss;
      ss << (drop_frames?"Dropped":"[dropping disabled] Would drop")
                           << " frames in " << utt << " " << frm_drop << "/" << num_frames << ",";
      //get frame intervals from vec frm_drop_vec
      ss << " intervals :";
      //search for streaks of consecutive numbers:
      int32 beg_streak=frm_drop_vec[0];
      int32 len_streak=0;
      int32 i;
      for(i=0; i<frm_drop_vec.size(); i++,len_streak++) {
        if(beg_streak + len_streak != frm_drop_vec[i]) {
          ss << " " << beg_streak << ".." << frm_drop_vec[i-1] << "frm";
          beg_streak = frm_drop_vec[i];
          len_streak = 0;
        }
      }
      ss << " " << beg_streak << ".." << frm_drop_vec[i-1] << "frm";
      //print
      KALDI_WARN << ss.str();
    }

    // Return the objective value
    return mmi_obj;
  }

 protected:
  BaseFloat acoustic_scale, old_acoustic_scale, lm_scale;
  RandomAccessLatticeReader den_lat_reader;
  RandomAccessInt32VectorReader num_ali_reader;
  TransitionModel trans_model;
  bool drop_frames;
}; //KaldiMMISequenceLoss

} //end pylearn2
} //end kaldi


//here comes the glue

namespace bp = boost::python;

template <class SeqLoss>
double compute_loss_and_diff(SeqLoss& self,
                             kaldi::int32 utt_id, bp::object frame_log_likelihoods, bp::object diff) {
  cout << "In top level compute_loss_and_diff" << endl;
  if (!PyArray_CheckExact(frame_log_likelihoods.ptr()) || !PyArray_CheckExact(diff.ptr())) {
    KALDI_ERR << "compute_loss_and_diff has to be called with NDarrays" << endl;
  }
  kaldi::NpWrapperMatrix<kaldi::BaseFloat> frame_log_likelihoods_((PyArrayObject*)frame_log_likelihoods.ptr());
  kaldi::NpWrapperMatrix<kaldi::BaseFloat> diff_((PyArrayObject*)diff.ptr());

  return self.compute_loss_and_diff_from_id(utt_id, frame_log_likelihoods_, diff_);
}

template <class SeqLoss>
class SequenceLossWrapper: public bp::class_<SeqLoss, boost::noncopyable> {
 public:
  template <class DerivedT>
  inline SequenceLossWrapper(char const* name, bp::init_base<DerivedT> const& i)
  : bp::class_<SeqLoss, boost::noncopyable>(name, i) {
    (*this)
        .def("compute_loss_and_diff", compute_loss_and_diff<SeqLoss>)
        ;
  }
};

//That is just for testing
//class DummyLoss: public kaldi::pylearn2::AbstractSequenceLoss {
// public:
//  DummyLoss(const Id2UttMap& utt_map)
// : AbstractSequenceLoss(utt_map) {
//    cout << "Building dummy loss" << endl;
//    cout << "Utt map:" << endl;
//    for (Id2UttMap::const_iterator iter=utt_map.begin(); iter!=utt_map.end(); ++iter) {
//      cout << iter->first << ' ' << iter->second << endl;
//    }
//  }
//
//  double compute_loss_and_diff(const std::string& utt_name,
//                               const kaldi::MatrixBase<kaldi::BaseFloat>& frame_log_likelihoods, kaldi::MatrixBase<kaldi::BaseFloat>& diff) {
//    cout << "In dummyloss string compute_loss_and_diff" << endl;
//    diff.CopyFromMat(frame_log_likelihoods);
//    return frame_log_likelihoods.Sum();
//  }
//
// private:
//  DummyLoss(const DummyLoss& other);
//  DummyLoss& operator=(const DummyLoss&);
//};

BOOST_PYTHON_MODULE(kaldi_sequence_loss)
{
  import_array();

  kaldi::MapFromDictBPConverter<kaldi::pylearn2::AbstractSequenceLoss::Id2UttMap>();

//  SequenceLossWrapper<DummyLoss>("DummyLoss", bp::init<kaldi::pylearn2::AbstractSequenceLoss::Id2UttMap >());
  SequenceLossWrapper<kaldi::pylearn2::KaldiMMISequenceLoss>("KaldiMMISequenceLoss",
                                            bp::init<kaldi::pylearn2::AbstractSequenceLoss::Id2UttMap,
                                                     kaldi::BaseFloat,
                                                     kaldi::BaseFloat,
                                                     kaldi::BaseFloat,
                                                     std::string,
                                                     std::string,
                                                     std::string,
                                                     bool
                                                     >());
}
