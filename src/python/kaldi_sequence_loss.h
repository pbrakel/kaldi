/*
 * sequence_loss.h
 *
 *  Created on: Aug 27, 2014
 *      Author: chorows
 */

#ifndef SEQUENCE_LOSS_H_
#define SEQUENCE_LOSS_H_

#include <map>
#include <vector>
#include <exception>


#include "base/kaldi-common.h"

#define PROCESS_ERROR(msg) (throw new std::exception())

namespace kaldi {
namespace pylearn2 {

class AbstractSequenceLoss {
 public:
  typedef std::map<int32, std::string> Id2UttMap;

  AbstractSequenceLoss(const Id2UttMap  & utt_map)
 : sequence_names(utt_map) {
  }

  virtual double compute_loss_and_diff(const std::string& utt_name, const MatrixBase<BaseFloat>& frame_log_likelihoods, MatrixBase<BaseFloat>& diff) =0;

  virtual double compute_loss_and_diff_from_id(int32 utt_id, const MatrixBase<BaseFloat>& frame_log_likelihoods, MatrixBase<BaseFloat>& diff) {
    std::string seq_name = sequence_names.at(utt_id);
    return compute_loss_and_diff(seq_name, frame_log_likelihoods, diff);
  }

  virtual ~AbstractSequenceLoss() {};

 protected:
  Id2UttMap sequence_names;
}; //class AbstractSequenceLoss


} //namespace pylearn2
} //namespace kaldi


#endif /* SEQUENCE_LOSS_H_ */
