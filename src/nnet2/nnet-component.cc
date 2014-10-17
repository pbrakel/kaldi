// nnet2/nnet-component.cc

// Copyright 2011-2012  Karel Vesely
//           2013-2014  Johns Hopkins University (author: Daniel Povey)
//                  2013  Xiaohui Zhang    

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

#include <sstream>
#include "nnet2/nnet-component.h"
#include "nnet2/nnet-precondition.h"
#include "nnet2/nnet-precondition-online.h"
#include "util/text-utils.h"
#include "util/kaldi-io.h"

namespace kaldi {
namespace nnet2 {

// static
Component* Component::ReadNew(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token); // e.g. "<SigmoidComponent>".
  token.erase(0, 1); // erase "<".
  token.erase(token.length()-1); // erase ">".
  Component *ans = NewComponentOfType(token);
  if (!ans)
    KALDI_ERR << "Unknown component type " << token;
  ans->Read(is, binary);
  return ans;
}


// static
Component* Component::NewComponentOfType(const std::string &component_type) {
  Component *ans = NULL;
  if (component_type == "SigmoidComponent") {
    ans = new SigmoidComponent();
  } else if (component_type == "TanhComponent") {
    ans = new TanhComponent();
  } else if (component_type == "PowerComponent") {
    ans = new PowerComponent();
  } else if (component_type == "SoftmaxComponent") {
    ans = new SoftmaxComponent();
  } else if (component_type == "RectifiedLinearComponent") {
    ans = new RectifiedLinearComponent();
  } else if (component_type == "NormalizeComponent") {
    ans = new NormalizeComponent();
  } else if (component_type == "SoftHingeComponent") {
    ans = new SoftHingeComponent();
  } else if (component_type == "PnormComponent") {
    ans = new PnormComponent();
  } else if (component_type == "MaxoutComponent") {
    ans = new MaxoutComponent();
  } else if (component_type == "ScaleComponent") {
    ans = new ScaleComponent();
  } else if (component_type == "AffineComponent") {
    ans = new AffineComponent();
  } else if (component_type == "AffineComponentPreconditioned") {
    ans = new AffineComponentPreconditioned();
  } else if (component_type == "AffineComponentPreconditionedOnline") {
    ans = new AffineComponentPreconditionedOnline();
  } else if (component_type == "SumGroupComponent") {
    ans = new SumGroupComponent();
  } else if (component_type == "BlockAffineComponent") {
    ans = new BlockAffineComponent();
  } else if (component_type == "BlockAffineComponentPreconditioned") {
    ans = new BlockAffineComponentPreconditioned();
  } else if (component_type == "PermuteComponent") {
    ans = new PermuteComponent();
  } else if (component_type == "DctComponent") {
    ans = new DctComponent();
  } else if (component_type == "FixedLinearComponent") {
    ans = new FixedLinearComponent();
  } else if (component_type == "FixedAffineComponent") {
    ans = new FixedAffineComponent();
  } else if (component_type == "FixedScaleComponent") {
    ans = new FixedScaleComponent();
  } else if (component_type == "FixedBiasComponent") {
    ans = new FixedBiasComponent();
  } else if (component_type == "SpliceComponent") {
    ans = new SpliceComponent();
  } else if (component_type == "SpliceMaxComponent") {
    ans = new SpliceMaxComponent();
  } else if (component_type == "DropoutComponent") {
    ans = new DropoutComponent();
  } else if (component_type == "AdditiveNoiseComponent") {
    ans = new AdditiveNoiseComponent();
  }
  return ans;
}

// static
Component* Component::NewFromString(const std::string &initializer_line) {
  std::istringstream istr(initializer_line);
  std::string component_type; // e.g. "SigmoidComponent".
  istr >> component_type >> std::ws; 
  std::string rest_of_line;
  getline(istr, rest_of_line);
  Component *ans = NewComponentOfType(component_type);
  if (ans == NULL)
    KALDI_ERR << "Bad initializer line (no such type of Component): "
              << initializer_line;
  ans->InitFromString(rest_of_line);
  return ans;
}


// This is like ExpectToken but for two tokens, and it
// will either accept token1 and then token2, or just token2.
// This is useful in Read functions where the first token
// may already have been consumed.
static void ExpectOneOrTwoTokens(std::istream &is, bool binary,
                                 const std::string &token1,
                                 const std::string &token2) {
  KALDI_ASSERT(token1 != token2);
  std::string temp;
  ReadToken(is, binary, &temp);
  if (temp == token1) {
    ExpectToken(is, binary, token2);
  } else {
    if (temp != token2) {
      KALDI_ERR << "Expecting token " << token1 << " or " << token2
                << " but got " << temp;
    }
  }
}


// static
bool ParseFromString(const std::string &name, std::string *string,
                     int32 *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      if (!ConvertStringToInteger(split_string[i].substr(len), param))
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     bool *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      std::string b = split_string[i].substr(len);
      if (b.empty())
        KALDI_ERR << "Bad option " << split_string[i];
      if (b[0] == 'f' || b[0] == 'F') *param = false;
      else if (b[0] == 't' || b[0] == 'T') *param = true;
      else
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     BaseFloat *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      if (!ConvertStringToReal(split_string[i].substr(len), param))
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;      
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     std::string *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      *param = split_string[i].substr(len);

      // Set "string" to all the pieces but the one we used.
      *string = "";
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;      
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     std::vector<int32> *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      if (!SplitStringToIntegers(split_string[i].substr(len), ":",
                                 false, param))
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}


Component *PermuteComponent::Copy() const {
  PermuteComponent *ans = new PermuteComponent();
  ans->reorder_ = reorder_;
  return ans;
}
void PermuteComponent::Init(const std::vector<int32> &reorder) {
  reorder_ = reorder;
  KALDI_ASSERT(!reorder.empty());
  std::vector<int32> indexes(reorder);
  std::sort(indexes.begin(), indexes.end());
  for (int32 i = 0; i < static_cast<int32>(indexes.size()); i++)
    KALDI_ASSERT(i == indexes[i] && "Not a permutation");
}


std::string Component::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim();
  return stream.str();
}

std::string UpdatableComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim() << ", learning-rate="
         << LearningRate();
  return stream.str();
}


void NonlinearComponent::SetDim(int32 dim) {
  KALDI_ASSERT(dim>0);
  dim_ = dim;
  value_sum_.Resize(dim);
  deriv_sum_.Resize(dim);
  count_ = 0.0;
}

void NonlinearComponent::UpdateStats(const CuMatrixBase<BaseFloat> &out_value,
                                     const CuMatrixBase<BaseFloat> *deriv) {
  KALDI_ASSERT(out_value.NumCols() == InputDim());
  // Check we have the correct dimensions.
  if (value_sum_.Dim() != InputDim() ||
      (deriv != NULL && deriv_sum_.Dim() != InputDim())) {
    mutex_.Lock();
    if (value_sum_.Dim() != InputDim()) {
      value_sum_.Resize(InputDim());
      count_ = 0.0;
    }
    if (deriv != NULL && deriv_sum_.Dim() != InputDim()) {
      deriv_sum_.Resize(InputDim());
      count_ = 0.0;
      value_sum_.SetZero();
    }
    mutex_.Unlock();
  }
  count_ += out_value.NumRows();
  CuVector<BaseFloat> temp(InputDim());
  temp.AddRowSumMat(1.0, out_value, 0.0);
  value_sum_.AddVec(1.0, temp);
  if (deriv != NULL) {
    temp.AddRowSumMat(1.0, *deriv, 0.0);
    deriv_sum_.AddVec(1.0, temp);
  }
}

void NonlinearComponent::Scale(BaseFloat scale) {
  value_sum_.Scale(scale);
  deriv_sum_.Scale(scale);
  count_ *= scale;
}

void NonlinearComponent::Add(BaseFloat alpha, const NonlinearComponent &other) {
  if (value_sum_.Dim() == 0 && other.value_sum_.Dim() != 0)
    value_sum_.Resize(other.value_sum_.Dim());
  if (deriv_sum_.Dim() == 0 && other.deriv_sum_.Dim() != 0)
    deriv_sum_.Resize(other.deriv_sum_.Dim());
  if (other.value_sum_.Dim() != 0)
    value_sum_.AddVec(alpha, other.value_sum_);
  if (other.deriv_sum_.Dim() != 0)
    deriv_sum_.AddVec(alpha, other.deriv_sum_);
  count_ += alpha * other.count_;
}

void NonlinearComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<Dim>");
  ReadBasicType(is, binary, &dim_); // Read dimension.
  std::string tok; // TODO: remove back-compatibility code.
  ReadToken(is, binary, &tok);
  if (tok == "<ValueSum>") {
    value_sum_.Read(is, binary);
    ExpectToken(is, binary, "<DerivSum>");
    deriv_sum_.Read(is, binary);
    ExpectToken(is, binary, "<Count>");
    ReadBasicType(is, binary, &count_);
    ExpectToken(is, binary, ostr_end.str());  
  } else if (tok == "<Counts>") { // Back-compat code for SoftmaxComponent.
    value_sum_.Read(is, binary); // Set both value_sum_ and deriv_sum_ to the same value,
    // and count_ to its sum.
    count_ = value_sum_.Sum();
    ExpectToken(is, binary, ostr_end.str());  
  } else {
    KALDI_ASSERT(tok == ostr_end.str());
  }
}

void NonlinearComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<ValueSum>");
  value_sum_.Write(os, binary);
  WriteToken(os, binary, "<DerivSum>");
  deriv_sum_.Write(os, binary);
  WriteToken(os, binary, "<Count>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, ostr_end.str());  
}

NonlinearComponent::NonlinearComponent(const NonlinearComponent &other):
    dim_(other.dim_), value_sum_(other.value_sum_), deriv_sum_(other.deriv_sum_),
    count_(other.count_) { }

void NonlinearComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  bool ok = ParseFromString("dim", &args, &dim);
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim);
}

void MaxoutComponent::Init(int32 input_dim, int32 output_dim)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  if (input_dim_ == 0)
    input_dim_ = 10 * output_dim_; // default group size : 10
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ >= 0);
  KALDI_ASSERT(input_dim_ % output_dim_ == 0) 
}

void MaxoutComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 input_dim = 0;
  int32 output_dim = 0;
  bool ok = ParseFromString("output-dim", &args, &output_dim) &&
      ParseFromString("input-dim", &args, &input_dim);
  KALDI_LOG << output_dim << " " << input_dim << " " << ok;
  if (!ok || !args.empty() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(input_dim, output_dim);
}


void MaxoutComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                int32 num_chunks,
                                CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), output_dim_, kUndefined);
  int32 group_size = input_dim_ / output_dim_;
  for (MatrixIndexT j = 0; j < output_dim_; j++) {
    CuSubMatrix<BaseFloat> pool(out->ColRange(j, 1));
    pool.Set(-1e20);
    for (MatrixIndexT i = 0; i < group_size; i++)
      pool.Max(in.ColRange(j * group_size + i, 1));
  }
}

void MaxoutComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value,
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               int32, // num_chunks
                               Component *to_update, // to_update
                               CuMatrix<BaseFloat> *in_deriv) const {
  int32 group_size = input_dim_ / output_dim_;
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols(), kSetZero);
  for (MatrixIndexT j = 0; j < output_dim_; j++) {
    CuSubMatrix<BaseFloat> out_j(out_value.ColRange(j, 1));
    for (MatrixIndexT i = 0; i < group_size; i++) {
        CuSubMatrix<BaseFloat> in_i(in_value.ColRange(j * group_size + i, 1));
        CuSubMatrix<BaseFloat> in_deriv_i(in_deriv->ColRange(j * group_size + i, 1));
        CuMatrix<BaseFloat> out_deriv_j(out_deriv.ColRange(j, 1));

        // Only the pool-inputs with 'max-values' are used to back-propagate into,
        // the rest of derivatives is zeroed-out by a mask.
        CuMatrix<BaseFloat> mask;
        in_i.EqualElementMask(out_j, &mask);
        out_deriv_j.MulElements(mask);
        in_deriv_i.AddMat(1.0, out_deriv_j); 
    }
  }
}

void MaxoutComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<MaxoutComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "</MaxoutComponent>");
}

void MaxoutComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MaxoutComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "</MaxoutComponent>");
}

std::string MaxoutComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim = " << input_dim_
         << ", output-dim = " << output_dim_;
  return stream.str();
}

void PnormComponent::Init(int32 input_dim, int32 output_dim, BaseFloat p)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  if (input_dim_ == 0)
    input_dim_ = 10 * output_dim_; // default group size : 10
  p_ = p;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ >= 0 && p_ >= 0);
  KALDI_ASSERT(input_dim_ % output_dim_ == 0) 
}

void PnormComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat p = 2;
  bool ok = ParseFromString("output-dim", &args, &output_dim) &&
      ParseFromString("input-dim", &args, &input_dim);
  ParseFromString("p", &args, &p);
  if (!ok || !args.empty() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(input_dim, output_dim, p);
}


void PnormComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                               int32 num_chunks,
                               CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), output_dim_, kUndefined);
  out->GroupPnorm(in, p_);
}

void PnormComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              int32, // num_chunks
                              Component *to_update, // to_update
                              CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols(), kSetZero);
  in_deriv->GroupPnormDeriv(in_value, out_value, p_);
  in_deriv->MulRowsGroupMat(out_deriv); 
}

void PnormComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PnormComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<P>");
  ReadBasicType(is, binary, &p_);
  ExpectToken(is, binary, "</PnormComponent>");
}

void PnormComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PnormComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<P>");
  WriteBasicType(os, binary, p_);
  WriteToken(os, binary, "</PnormComponent>");
}

std::string PnormComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim = " << input_dim_
         << ", output-dim = " << output_dim_
     << ", p = " << p_;
  return stream.str();
}


const BaseFloat NormalizeComponent::kNormFloor = pow(2.0, -66);
// This component modifies the vector of activations by scaling it so that the
// root-mean-square equals 1.0.

void NormalizeComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                              int32, // num_chunks
                              CuMatrix<BaseFloat> *out) const {
  *out = in;
  CuVector<BaseFloat> in_norm(in.NumRows());
  in_norm.AddDiagMat2(1.0 / in.NumCols(),
                      in, kNoTrans, 0.0);
  in_norm.ApplyFloor(kNormFloor);
  in_norm.ApplyPow(-0.5);
  out->MulRowsVec(in_norm);
}

/*
  A note on the derivative of NormalizeComponent...
  let both row_in and row_out be vectors of dimension D.
  Let p = row_in^T row_in / D, and let
      f = 1 / sqrt(max(kNormFloor, p)), and we compute row_out as:
row_out = f row_in.
  Suppose we have a quantity deriv_out which is the derivative
  of the objective function w.r.t. row_out.  We want to compute
  deriv_in which is the derivative of the objective function w.r.t.
  row_in.  Let the objective function be F.  One term is obvious: we have
     deriv_in = f deriv_out + ....
  next we have to take into account the derivative that gets back-propagated
  through f.  Obviously, dF/df = deriv_out^T row_in.
  And df/dp = (p <= kNormFloor ? 0.0 : -0.5 p^{-1.5}) = (f == 1 / sqrt(kNormFloor) ? 0.0 : -0.5 f^3),
  and dp/d(row_in) = 2/D row_in. [it's vector_valued].
  So this term in dF/d(row_in) equals:
    dF/df df/dp dp/d(row_in)   =    2/D (f == 1 / sqrt(kNormFloor)  ? 0.0 : -0.5 f^3) (deriv_out^T row_in) row_in
  So
     deriv_in = f deriv_out + (f == 1.0 ? 0.0 : -f^3 / D) (deriv_out^T row_in) row_in

*/

void NormalizeComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  int32, // num_chunks
                                  Component *to_update,
                                  CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  
  CuVector<BaseFloat> in_norm(in_value.NumRows());
  in_norm.AddDiagMat2(1.0 / in_value.NumCols(),
                      in_value, kNoTrans, 0.0);
  in_norm.ApplyFloor(kNormFloor);
  in_norm.ApplyPow(-0.5);
  in_deriv->AddDiagVecMat(1.0, in_norm, out_deriv, kNoTrans, 0.0);
  in_norm.ReplaceValue(1.0 / sqrt(kNormFloor), 0.0);
  in_norm.ApplyPow(3.0);
  CuVector<BaseFloat> dot_products(in_deriv->NumRows());
  dot_products.AddDiagMatMat(1.0, out_deriv, kNoTrans, in_value, kTrans, 0.0);
  dot_products.MulElements(in_norm);
  
  in_deriv->AddDiagVecMat(-1.0 / in_value.NumCols(), dot_products, in_value, kNoTrans, 1.0);
}

void SigmoidComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                 int32, // num_chunks
                                 CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), in.NumCols());
  out->Sigmoid(in);
}

void SigmoidComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                int32, // num_chunks
                                Component *to_update,
                                CuMatrix<BaseFloat> *in_deriv) const {
  // we ignore in_value and to_update.

  // The element by element equation would be:
  // in_deriv = out_deriv * out_value * (1.0 - out_value);
  // We can accomplish this via calls to the matrix library.

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  in_deriv->Set(1.0);
  in_deriv->AddMat(-1.0, out_value);
  // now in_deriv = 1.0 - out_value [element by element]
  in_deriv->MulElements(out_value);
  // now in_deriv = out_value * (1.0 - out_value) [element by element], i.e.
  // it contains the element-by-element derivative of the nonlinearity.
  if (to_update != NULL)
    dynamic_cast<NonlinearComponent*>(to_update)->UpdateStats(out_value,
                                                              in_deriv);
  in_deriv->MulElements(out_deriv);
  // now in_deriv = out_deriv * out_value * (1.0 - out_value) [element by element]
}


void TanhComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                              int32, // num_chunks
                              CuMatrix<BaseFloat> *out) const {
  // Apply tanh function to each element of the output...
  // the tanh function may be written as -1 + ( 2 / (1 + e^{-2 x})),
  // which is a scaled and shifted sigmoid.
  out->Resize(in.NumRows(), in.NumCols(), kUndefined);
  out->Tanh(in);
}

void TanhComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                             const CuMatrixBase<BaseFloat> &out_value,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             int32, // num_chunks
                             Component *to_update,
                             CuMatrix<BaseFloat> *in_deriv) const {
  /*
    Note on the derivative of the tanh function:
    tanh'(x) = sech^2(x) = -(tanh(x)+1) (tanh(x)-1) = 1 - tanh^2(x)

    The element by element equation of what we're doing would be:
    in_deriv = out_deriv * (1.0 - out_value^2).
    We can accomplish this via calls to the matrix library. */

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  in_deriv->CopyFromMat(out_value);
  in_deriv->ApplyPow(2.0);
  in_deriv->Scale(-1.0);
  in_deriv->Add(1.0);
  // now in_deriv = (1.0 - out_value^2), the element-by-element derivative of
  // the nonlinearity.
  if (to_update != NULL)
    dynamic_cast<NonlinearComponent*>(to_update)->UpdateStats(out_value,
                                                              in_deriv);
  in_deriv->MulElements(out_deriv);
}  

void PowerComponent::Init(int32 dim, BaseFloat power) {
  dim_ = dim;
  power_ = power;
  KALDI_ASSERT(dim > 0 && power >= 0);
}

void PowerComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  BaseFloat power = 2.0;
  ParseFromString("power", &args, &power); // Optional.
  // Accept either "dim" or "input-dim" to specify the input dim.
  // "input-dim" is the canonical one; "dim" simplifies the testing code.
  bool ok = (ParseFromString("dim", &args, &dim) ||
             ParseFromString("input-dim", &args, &dim));
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim, power);
}

void PowerComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                              int32, // num_chunks
                              CuMatrix<BaseFloat> *out) const {
  // Apply power operation to each element of the input...
  out->Resize(in.NumRows(), in.NumCols(), kUndefined);
  out->CopyFromMat(in);
  out->ApplyPowAbs(power_);
}

void PowerComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                             const CuMatrixBase<BaseFloat> &out_value,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             int32, // num_chunks
                             Component *to_update,
                             CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols());
  // in scalar terms: in_deriv += p * in_value^(p-1) * out_deriv
  in_deriv->CopyFromMat(in_value); 
  in_deriv->ApplyPowAbs(power_ - 1.0, true);
  in_deriv->Scale(power_);
  in_deriv->MulElements(out_deriv);
}

void PowerComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PowerComponent>", "<InputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<Power>");
  ReadBasicType(is, binary, &power_);
  ExpectToken(is, binary, "</PowerComponent>");
}

void PowerComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PowerComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Power>");
  WriteBasicType(os, binary, power_);
  WriteToken(os, binary, "</PowerComponent>");
}

std::string PowerComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", dim = " << dim_
     << ", power = " << power_;
  return stream.str();
}

void RectifiedLinearComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                              int32, // num_chunks
                              CuMatrix<BaseFloat> *out) const {
  // Apply rectified linear function (x >= 0 ? 1.0 : 0.0) 
  *out = in;
  out->ApplyFloor(0.0);
}

void RectifiedLinearComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                        const CuMatrixBase<BaseFloat> &out_value,
                                        const CuMatrixBase<BaseFloat> &out_deriv,
                                        int32, // num_chunks
                                        Component *to_update,
                                        CuMatrix<BaseFloat> *in_deriv) const {

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols(),
                   kUndefined);
  in_deriv->CopyFromMat(out_value);
  in_deriv->ApplyHeaviside();
  // Now in_deriv(i, j) equals (out_value(i, j) > 0.0 ? 1.0 : 0.0),
  // which is the derivative of the nonlinearity (well, except at zero
  // where it's undefined).
  if (to_update != NULL)
    dynamic_cast<NonlinearComponent*>(to_update)->UpdateStats(out_value,
                                                              in_deriv);
  in_deriv->MulElements(out_deriv);
}  

void SoftHingeComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                   int32, // num_chunks
                                   CuMatrix<BaseFloat> *out) const {
  // Apply function x = log(1 + exp(x))
  out->Resize(in.NumRows(), in.NumCols(), kUndefined);
  out->SoftHinge(in);
}

void SoftHingeComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  int32, // num_chunks
                                  Component *to_update,
                                  CuMatrix<BaseFloat> *in_deriv) const {

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols(),
                   kUndefined);
  // note: d/dx: log(1 + exp(x)) = (exp(x) / (1 + exp(x)) = 1 / (1 + exp(-x)),
  // which is the sigmoid function.
  
  // if the output is y, then dy/dx =  (exp(x) / (1 + exp(x)),
  // and using y = log(1 + exp(x)) -> exp(x) = exp(y) - 1, we have
  // dy/dx = (exp(y) - 1) / exp(y)
  

  in_deriv->Sigmoid(in_value);

  if (to_update != NULL)
    dynamic_cast<NonlinearComponent*>(to_update)->UpdateStats(out_value,
                                                              in_deriv);
  in_deriv->MulElements(out_deriv);
}  


void ScaleComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                   int32, // num_chunks
                                   CuMatrix<BaseFloat> *out) const {
  *out = in;
  out->Scale(scale_);
}

void ScaleComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                              const CuMatrixBase<BaseFloat> &, // out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              int32, // num_chunks
                              Component *, // to_update
                              CuMatrix<BaseFloat> *in_deriv) const {

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols(),
                   kUndefined);
  in_deriv->CopyFromMat(out_deriv);
  in_deriv->Scale(scale_);
}  

void ScaleComponent::Init(int32 dim, BaseFloat scale) {
  dim_ = dim;
  scale_ = scale;
  KALDI_ASSERT(dim_ > 0);
  KALDI_ASSERT(scale_ != 0.0);
}

void ScaleComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  BaseFloat scale;
  if (!ParseFromString("dim", &args, &dim))
    KALDI_ERR << "Dimension not specified for ScaleComponent in config file";
  if (!ParseFromString("scale", &args, &scale))
    KALDI_ERR << "Scale not specified for ScaleComponent in config file";
  Init(dim, scale);
}

void ScaleComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ScaleComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "</ScaleComponent>");
}

void ScaleComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<ScaleComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "</ScaleComponent>");
}

std::string ScaleComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", dim=" << dim_ << ", scale=" << scale_;
  return stream.str();
}

void SoftmaxComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                 int32, // num_chunks
                                 CuMatrix<BaseFloat> *out) const {
  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).

  out->Resize(in.NumRows(), in.NumCols(), kUndefined);
  out->ApplySoftMaxPerRow(in);
  
  // This floor on the output helps us deal with
  // almost-zeros in a way that doesn't lead to overflow.
  out->ApplyFloor(1.0e-20);
}

void SoftmaxComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                int32 num_chunks,
                                Component *to_update, // only thing updated is counts_.
                                CuMatrix<BaseFloat> *in_deriv) const {
  /*
    Note on the derivative of the softmax function: let it be
    p_i = exp(x_i) / sum_i exp_i
    The [matrix-valued] Jacobian of this function is
    diag(p) - p p^T
    Let the derivative vector at the output be e, and at the input be
    d.  We have
    d = diag(p) e - p (p^T e).
    d_i = p_i e_i - p_i (p^T e).    
  */
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());  
  KALDI_ASSERT(SameDim(out_value, out_deriv) && SameDim(out_value, *in_deriv));
  const CuMatrixBase<BaseFloat> &P(out_value), &E(out_deriv);
  CuMatrixBase<BaseFloat> &D (*in_deriv);


#if 1
  D.CopyFromMat(P);
  D.MulElements(E);
  // At this point, D = P .* E (in matlab notation)
  CuVector<BaseFloat> pe_vec(D.NumRows()); // For each row i, the dot product (p_t . e_t).
  pe_vec.AddDiagMatMat(1.0, P, kNoTrans, E, kTrans, 0.0);

  D.AddDiagVecMat(-1.0, pe_vec, P, kNoTrans, 1.0); // does D -= diag(pe_vec) * P.
#else  
  // The old code, where we did stuff row-by-row, is as follows;
  //   we had to rework it to use whole-matrix operations in order
  //   to use CUDA more effectively. 
  for (int32 r = 0; r < P.NumRows(); r++) {
    CuSubVector<BaseFloat> p(P, r), e(E, r), d(D, r);
    d.AddVecVec(1.0, p, e, 0.0); // d_i = p_i e_i.
    BaseFloat pT_e = VecVec(p, e); // p^T e.
    d.AddVec(-pT_e, p); // d_i -= (p^T e) p_i
  }
#endif
  
  // The SoftmaxComponent does not have any real trainable parameters, but
  // during the backprop we store some statistics on the average counts;
  // these may be used in mixing-up.
  if (to_update != NULL) {
    NonlinearComponent *to_update_nonlinear =
        dynamic_cast<NonlinearComponent*>(to_update);
    to_update_nonlinear->UpdateStats(out_value);
  }
}

void AffineComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
}

void AffineComponent::Add(BaseFloat alpha, const UpdatableComponent &other_in) {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

AffineComponent::AffineComponent(const AffineComponent &component):
    UpdatableComponent(component),
    linear_params_(component.linear_params_),
    bias_params_(component.bias_params_),
    is_gradient_(component.is_gradient_) { }

AffineComponent::AffineComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 const CuVectorBase<BaseFloat> &bias_params,
                                 BaseFloat learning_rate):
    UpdatableComponent(learning_rate),
    linear_params_(linear_params),
    bias_params_(bias_params) {
  KALDI_ASSERT(linear_params.NumRows() == bias_params.Dim()&&
               bias_params.Dim() != 0);
  is_gradient_ = false;
}



void AffineComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
  if (treat_as_gradient)
    is_gradient_ = true;
}

void AffineComponent::SetParams(const VectorBase<BaseFloat> &bias,
                                const MatrixBase<BaseFloat> &linear) {
  bias_params_ = bias;
  linear_params_ = linear;
  KALDI_ASSERT(bias_params_.Dim() == linear_params_.NumRows());
}

void AffineComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  
  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

std::string AffineComponent::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols());
  BaseFloat linear_stddev =
      std::sqrt(TraceMatMat(linear_params_, linear_params_, kTrans) /
                linear_params_size),
      bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                              bias_params_.Dim());
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", linear-params-stddev=" << linear_stddev
         << ", bias-params-stddev=" << bias_stddev
         << ", learning-rate=" << LearningRate();
  return stream.str();
}

Component* AffineComponent::Copy() const {
  AffineComponent *ans = new AffineComponent();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}

BaseFloat AffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

void AffineComponent::Init(BaseFloat learning_rate, 
                           int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev, BaseFloat bias_stddev) {
  UpdatableComponent::Init(learning_rate);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
}

void AffineComponent::Init(BaseFloat learning_rate,
                           std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);  
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
}

void AffineComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  if (ParseFromString("matrix", &args, &matrix_filename)) {    
    Init(learning_rate, matrix_filename);
    if (ParseFromString("input-dim", &args, &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (ParseFromString("output-dim", &args, &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && ParseFromString("input-dim", &args, &input_dim);
    ok = ok && ParseFromString("output-dim", &args, &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0;
    ParseFromString("param-stddev", &args, &param_stddev);
    ParseFromString("bias-stddev", &args, &bias_stddev);
    Init(learning_rate, input_dim, output_dim,
         param_stddev, bias_stddev);    
  }
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
}


void AffineComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                int32, // num_chunks
                                CuMatrix<BaseFloat> *out) const {
  // No need for asserts as they'll happen within the matrix operations.
  out->Resize(in.NumRows(), linear_params_.NumRows());
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
}

void AffineComponent::UpdateSimple(const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv) {
  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
}

void AffineComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &,  // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               int32, //  num_chunks
                               Component *to_update_in,
                               CuMatrix<BaseFloat> *in_deriv) const {
  AffineComponent *to_update = dynamic_cast<AffineComponent*>(to_update_in);
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  // Propagate the derivative back to the input.
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                      0.0);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(in_value, out_deriv);  // by child classes.
  }
}

void AffineComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  // might not see the "<AffineComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  std::string tok;
  // back-compatibility code.  TODO: re-do this later.
  ReadToken(is, binary, &tok);
  if (tok == "<AvgInput>") { // discard the following.
    CuVector<BaseFloat> avg_input;
    avg_input.Read(is, binary);
    BaseFloat avg_input_count;
    ExpectToken(is, binary, "<AvgInputCount>");
    ReadBasicType(is, binary, &avg_input_count);
    ReadToken(is, binary, &tok);
  }
  if (tok == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ExpectToken(is, binary, ostr_end.str());
  } else {
    is_gradient_ = false;
    KALDI_ASSERT(tok == ostr_end.str());
  }
}

void AffineComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, ostr_end.str());
}

int32 AffineComponent::GetParameterDim() const {
  return (InputDim() + 1) * OutputDim();
}
void AffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
  params->Range(InputDim() * OutputDim(),
                OutputDim()).CopyFromVec(bias_params_);
}
void AffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  bias_params_.CopyFromVec(params.Range(InputDim() * OutputDim(),
                                        OutputDim()));
}

void AffineComponent::LimitRank(int32 d,
                                AffineComponent **a, AffineComponent **b) const {
  KALDI_ASSERT(d <= InputDim());

  // We'll limit the rank of just the linear part, keeping the bias vector full.
  Matrix<BaseFloat> M (linear_params_);
  int32 rows = M.NumRows(), cols = M.NumCols(), rc_min = std::min(rows, cols);
  Vector<BaseFloat> s(rc_min);
  Matrix<BaseFloat> U(rows, rc_min), Vt(rc_min, cols);
  // Do the destructive svd M = U diag(s) V^T.  It actually outputs the transpose of V.
  M.DestructiveSvd(&s, &U, &Vt);
  SortSvd(&s, &U, &Vt); // Sort the singular values from largest to smallest.
  BaseFloat old_svd_sum = s.Sum();
  U.Resize(rows, d, kCopyData);
  s.Resize(d, kCopyData);
  Vt.Resize(d, cols, kCopyData);
  BaseFloat new_svd_sum = s.Sum();
  KALDI_LOG << "Reduced rank from "
            << rc_min <<  " to " << d << ", SVD sum reduced from "
            << old_svd_sum << " to " << new_svd_sum;

  // U.MulColsVec(s); // U <-- U diag(s)
  Vt.MulRowsVec(s); // Vt <-- diag(s) Vt.

  *a = dynamic_cast<AffineComponent*>(this->Copy());
  *b = dynamic_cast<AffineComponent*>(this->Copy());
  
  (*a)->bias_params_.Resize(d, kSetZero);
  (*a)->linear_params_ = Vt;
  
  (*b)->bias_params_ = this->bias_params_;
  (*b)->linear_params_ = U;
}

Component *AffineComponent::CollapseWithNext(
    const AffineComponent &next_component) const {
  AffineComponent *ans = dynamic_cast<AffineComponent*>(this->Copy());
  KALDI_ASSERT(ans != NULL);
  // Note: it's possible that "ans" is really of a derived type such
  // as AffineComponentPreconditioned, but this will still work.
  // the "copy" call will copy things like learning rates, "alpha" value
  // for preconditioned component, etc.
  ans->linear_params_.Resize(next_component.OutputDim(), InputDim());
  ans->bias_params_ = next_component.bias_params_;

  ans->linear_params_.AddMatMat(1.0, next_component.linear_params_, kNoTrans,
                                this->linear_params_, kNoTrans, 0.0);
  ans->bias_params_.AddMatVec(1.0, next_component.linear_params_, kNoTrans,
                              this->bias_params_, 1.0);
  return ans;
}

Component *AffineComponent::CollapseWithNext(
    const FixedAffineComponent &next_component) const {
  // If at least one was non-updatable, make the whole non-updatable.
  FixedAffineComponent *ans =
      dynamic_cast<FixedAffineComponent*>(next_component.Copy());
  KALDI_ASSERT(ans != NULL);
  ans->linear_params_.Resize(next_component.OutputDim(), InputDim());
  ans->bias_params_ = next_component.bias_params_;

  ans->linear_params_.AddMatMat(1.0, next_component.linear_params_, kNoTrans,
                                this->linear_params_, kNoTrans, 0.0);
  ans->bias_params_.AddMatVec(1.0, next_component.linear_params_, kNoTrans,
                              this->bias_params_, 1.0);
  return ans;
}

Component *AffineComponent::CollapseWithPrevious(
    const FixedAffineComponent &prev_component) const {
  // If at least one was non-updatable, make the whole non-updatable.
  FixedAffineComponent *ans =
      dynamic_cast<FixedAffineComponent*>(prev_component.Copy());
  KALDI_ASSERT(ans != NULL);

  ans->linear_params_.Resize(this->OutputDim(), prev_component.InputDim());
  ans->bias_params_ = this->bias_params_;

  ans->linear_params_.AddMatMat(1.0, this->linear_params_, kNoTrans,
                                prev_component.linear_params_, kNoTrans, 0.0);
  ans->bias_params_.AddMatVec(1.0, this->linear_params_, kNoTrans,
                              prev_component.bias_params_, 1.0);
  return ans;
}

void AffineComponentPreconditioned::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponentPreconditioned>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponentPreconditioned>"
  // might not see the "<AffineComponentPreconditioned>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  // todo: remove back-compat code.  Will just be:
  // ExpectToken(is, binary, "<MaxChange>");
  // ReadBasicType(is, binary, &max_change_);
  // ExpectToken(is, binary, ostr_end);
  // [end of function]
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<MaxChange>") {
    ReadBasicType(is, binary, &max_change_);
    ExpectToken(is, binary, ostr_end.str());
  } else {
    max_change_ = 0.0;
    KALDI_ASSERT(tok == ostr_end.str());
  }
}

void AffineComponentPreconditioned::InitFromString(std::string args) {
  std::string orig_args(args);
  std::string matrix_filename;
  BaseFloat learning_rate = learning_rate_;
  BaseFloat alpha = 0.1, max_change = 0.0;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("alpha", &args, &alpha);
  ParseFromString("max-change", &args, &max_change);

  if (ParseFromString("matrix", &args, &matrix_filename)) {
    Init(learning_rate, alpha, max_change, matrix_filename);
    if (ParseFromString("input-dim", &args, &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (ParseFromString("output-dim", &args, &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    bool ok = true;
    ok = ok && ParseFromString("input-dim", &args, &input_dim);
    ok = ok && ParseFromString("output-dim", &args, &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0;
    ParseFromString("param-stddev", &args, &param_stddev);
    ParseFromString("bias-stddev", &args, &bias_stddev);
    if (!ok)
      KALDI_ERR << "Bad initializer " << orig_args;
    Init(learning_rate, input_dim, output_dim, param_stddev,
         bias_stddev, alpha, max_change);
  }
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
}

void AffineComponentPreconditioned::Init(BaseFloat learning_rate,
                                         BaseFloat alpha, BaseFloat max_change,
                                         std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);
  alpha_ = alpha;
  max_change_ = max_change;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
}

void AffineComponentPreconditioned::Init(
    BaseFloat learning_rate, 
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev,
    BaseFloat alpha, BaseFloat max_change) {
  UpdatableComponent::Init(learning_rate);
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  alpha_ = alpha;
  KALDI_ASSERT(alpha_ > 0.0);
  max_change_ = max_change; // Note: any value of max_change_is valid, but
  // only values > 0.0 will actually activate the code.
}


void AffineComponentPreconditioned::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
  WriteToken(os, binary, "<MaxChange>");
  WriteBasicType(os, binary, max_change_);
  WriteToken(os, binary, ostr_end.str());
}

std::string AffineComponentPreconditioned::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols());
  BaseFloat linear_stddev =
      std::sqrt(TraceMatMat(linear_params_, linear_params_, kTrans) /
                linear_params_size),
      bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                              bias_params_.Dim());
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", linear-params-stddev=" << linear_stddev
         << ", bias-params-stddev=" << bias_stddev
         << ", learning-rate=" << LearningRate()
         << ", alpha=" << alpha_
         << ", max-change=" << max_change_;
  return stream.str();
}

Component* AffineComponentPreconditioned::Copy() const {
  AffineComponentPreconditioned *ans = new AffineComponentPreconditioned();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->alpha_ = alpha_;
  ans->max_change_ = max_change_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}


BaseFloat AffineComponentPreconditioned::GetScalingFactor(
    const CuMatrix<BaseFloat> &in_value_precon,
    const CuMatrix<BaseFloat> &out_deriv_precon) {
  static int scaling_factor_printed = 0;

  KALDI_ASSERT(in_value_precon.NumRows() == out_deriv_precon.NumRows());
  CuVector<BaseFloat> in_norm(in_value_precon.NumRows()),
      out_deriv_norm(in_value_precon.NumRows());
  in_norm.AddDiagMat2(1.0, in_value_precon, kNoTrans, 0.0);
  out_deriv_norm.AddDiagMat2(1.0, out_deriv_precon, kNoTrans, 0.0);
  // Get the actual l2 norms, not the squared l2 norm.
  in_norm.ApplyPow(0.5);
  out_deriv_norm.ApplyPow(0.5);
  BaseFloat sum = learning_rate_ * VecVec(in_norm, out_deriv_norm);
  // sum is the product of norms that we are trying to limit
  // to max_value_.
  KALDI_ASSERT(sum == sum && sum - sum == 0.0 &&
               "NaN in backprop");
  KALDI_ASSERT(sum >= 0.0);
  if (sum <= max_change_) return 1.0;
  else {
    BaseFloat ans = max_change_ / sum;
    if (scaling_factor_printed < 10) {
      KALDI_LOG << "Limiting step size to " << max_change_
                << " using scaling factor " << ans << ", for component index "
                << Index();
      scaling_factor_printed++;
    }
    return ans;
  }
}

void AffineComponentPreconditioned::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  CuMatrix<BaseFloat> in_value_temp;
  
  in_value_temp.Resize(in_value.NumRows(),
                       in_value.NumCols() + 1, kUndefined);
  in_value_temp.Range(0, in_value.NumRows(),
                      0, in_value.NumCols()).CopyFromMat(in_value);

  // Add the 1.0 at the end of each row "in_value_temp"  
  in_value_temp.Range(0, in_value.NumRows(),
                      in_value.NumCols(), 1).Set(1.0);
  
  CuMatrix<BaseFloat> in_value_precon(in_value_temp.NumRows(),
                                      in_value_temp.NumCols(), kUndefined),
      out_deriv_precon(out_deriv.NumRows(),
                       out_deriv.NumCols(), kUndefined);
  // each row of in_value_precon will be that same row of
  // in_value, but multiplied by the inverse of a Fisher
  // matrix that has been estimated from all the other rows,
  // smoothed by some appropriate amount times the identity
  // matrix (this amount is proportional to \alpha).
  PreconditionDirectionsAlphaRescaled(in_value_temp, alpha_, &in_value_precon);
  PreconditionDirectionsAlphaRescaled(out_deriv, alpha_, &out_deriv_precon);

  BaseFloat minibatch_scale = 1.0;

  if (max_change_ > 0.0)
    minibatch_scale = GetScalingFactor(in_value_precon, out_deriv_precon);
  
  
  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_precon,
                                            0, in_value_precon.NumRows(),
                                            0, in_value_precon.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
  CuVector<BaseFloat> precon_ones(in_value_precon.NumRows());
  
  precon_ones.CopyColFromMat(in_value_precon, in_value_precon.NumCols() - 1);

  BaseFloat local_lrate = minibatch_scale * learning_rate_;
  bias_params_.AddMatVec(local_lrate, out_deriv_precon, kTrans,
                         precon_ones, 1.0);
  linear_params_.AddMatMat(local_lrate, out_deriv_precon, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
}

void AffineComponentPreconditionedOnline::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">";
  ostr_end << "</" << Type() << ">";
  // might not see the "<AffineComponentPreconditionedOnline>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<Rank>") {  // back-compatibility (temporary)
    ReadBasicType(is, binary, &rank_in_);
    rank_out_ = rank_in_;
  } else {
    KALDI_ASSERT(tok == "<RankIn>");
    ReadBasicType(is, binary, &rank_in_);
    ExpectToken(is, binary, "<RankOut>");
    ReadBasicType(is, binary, &rank_out_);    
  }
  ReadToken(is, binary, &tok);
  if (tok == "<UpdatePeriod>") {
    ReadBasicType(is, binary, &update_period_);
    ExpectToken(is, binary, "<NumSamplesHistory>");
  } else {
    update_period_ = 1;
    KALDI_ASSERT(tok == "<NumSamplesHistory>");
  }
  ReadBasicType(is, binary, &num_samples_history_);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  ExpectToken(is, binary, "<MaxChangePerSample>");
  ReadBasicType(is, binary, &max_change_per_sample_);
  ExpectToken(is, binary, ostr_end.str());
  SetPreconditionerConfigs();
}

void AffineComponentPreconditionedOnline::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  std::string matrix_filename;
  BaseFloat learning_rate = learning_rate_;
  BaseFloat num_samples_history = 2000.0, alpha = 4.0,
      max_change_per_sample = 0.1;
  int32 input_dim = -1, output_dim = -1, rank_in = 30, rank_out = 80,
      update_period = 1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("num-samples-history", &args, &num_samples_history);
  ParseFromString("alpha", &args, &alpha);
  ParseFromString("max-change-per-sample", &args, &max_change_per_sample);
  ParseFromString("rank-in", &args, &rank_in);
  ParseFromString("rank-out", &args, &rank_out);
  ParseFromString("update-period", &args, &update_period);

  if (ParseFromString("matrix", &args, &matrix_filename)) {
    Init(learning_rate, rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample,
         matrix_filename);
    if (ParseFromString("input-dim", &args, &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (ParseFromString("output-dim", &args, &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && ParseFromString("input-dim", &args, &input_dim);
    ok = ok && ParseFromString("output-dim", &args, &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0;
    ParseFromString("param-stddev", &args, &param_stddev);
    ParseFromString("bias-stddev", &args, &bias_stddev);
    Init(learning_rate, input_dim, output_dim, param_stddev,
         bias_stddev, rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample);
  }
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
}

void AffineComponentPreconditionedOnline::SetPreconditionerConfigs() {
  preconditioner_in_.SetRank(rank_in_);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_in_.SetAlpha(alpha_);
  preconditioner_in_.SetUpdatePeriod(update_period_);
  preconditioner_out_.SetRank(rank_out_);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_out_.SetAlpha(alpha_);
  preconditioner_out_.SetUpdatePeriod(update_period_);
}

void AffineComponentPreconditionedOnline::Init(
    BaseFloat learning_rate, int32 rank_in, int32 rank_out,
    int32 update_period, BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample,
    std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetPreconditionerConfigs();
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
}

AffineComponentPreconditionedOnline::AffineComponentPreconditionedOnline(
    const AffineComponent &orig,
    int32 rank_in, int32 rank_out, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha):
    max_change_per_sample_(0.1) {
  this->linear_params_ = orig.linear_params_;
  this->bias_params_ = orig.bias_params_;
  this->learning_rate_ = orig.learning_rate_;
  this->is_gradient_ = orig.is_gradient_;
  this->rank_in_ = rank_in;
  this->rank_out_ = rank_out;
  this->update_period_ = update_period;
  this->num_samples_history_ = num_samples_history;
  this->alpha_ = alpha;
  SetPreconditionerConfigs();
}

void AffineComponentPreconditionedOnline::Init(
    BaseFloat learning_rate, 
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev,
    int32 rank_in, int32 rank_out, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample) {
  UpdatableComponent::Init(learning_rate);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0 &&
               bias_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetPreconditionerConfigs();
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
}


void AffineComponentPreconditionedOnline::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<RankIn>");
  WriteBasicType(os, binary, rank_in_);
  WriteToken(os, binary, "<RankOut>");
  WriteBasicType(os, binary, rank_out_);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period_);
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history_);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
  WriteToken(os, binary, "<MaxChangePerSample>");
  WriteBasicType(os, binary, max_change_per_sample_);
  WriteToken(os, binary, ostr_end.str());
}

std::string AffineComponentPreconditionedOnline::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols());
  BaseFloat linear_stddev =
      std::sqrt(TraceMatMat(linear_params_, linear_params_, kTrans) /
                linear_params_size),
      bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                              bias_params_.Dim());
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", linear-params-stddev=" << linear_stddev
         << ", bias-params-stddev=" << bias_stddev
         << ", learning-rate=" << LearningRate()
         << ", rank-in=" << rank_in_
         << ", rank-out=" << rank_out_
         << ", num_samples_history=" << num_samples_history_
         << ", update_period=" << update_period_
         << ", alpha=" << alpha_
         << ", max-change-per-sample=" << max_change_per_sample_;
  return stream.str();
}

Component* AffineComponentPreconditionedOnline::Copy() const {
  AffineComponentPreconditionedOnline *ans = new AffineComponentPreconditionedOnline();
  ans->learning_rate_ = learning_rate_;
  ans->rank_in_ = rank_in_;
  ans->rank_out_ = rank_out_;
  ans->update_period_ = update_period_;
  ans->num_samples_history_ = num_samples_history_;
  ans->alpha_ = alpha_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->preconditioner_in_ = preconditioner_in_;
  ans->preconditioner_out_ = preconditioner_out_;
  ans->max_change_per_sample_ = max_change_per_sample_;
  ans->is_gradient_ = is_gradient_;
  ans->SetPreconditionerConfigs();
  return ans;
}



BaseFloat AffineComponentPreconditionedOnline::GetScalingFactor(
    const CuVectorBase<BaseFloat> &in_products,
    BaseFloat learning_rate_scale,
    CuVectorBase<BaseFloat> *out_products) {
  static int scaling_factor_printed = 0;
  int32 minibatch_size = in_products.Dim();

  out_products->MulElements(in_products);
  out_products->ApplyPow(0.5);
  BaseFloat prod_sum = out_products->Sum();
  BaseFloat tot_change_norm = learning_rate_scale * learning_rate_ * prod_sum,
      max_change_norm = max_change_per_sample_ * minibatch_size;
  // tot_change_norm is the product of norms that we are trying to limit
  // to max_value_.
  KALDI_ASSERT(tot_change_norm - tot_change_norm == 0.0 && "NaN in backprop");
  KALDI_ASSERT(tot_change_norm >= 0.0);
  if (tot_change_norm <= max_change_norm) return 1.0;
  else {
    BaseFloat factor = max_change_norm / tot_change_norm;
    if (scaling_factor_printed < 10) {
      KALDI_LOG << "Limiting step size using scaling factor "
                << factor << ", for component index " << Index();
      scaling_factor_printed++;
    }
    return factor;
  }
}

void AffineComponentPreconditionedOnline::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  CuMatrix<BaseFloat> in_value_temp;
  
  in_value_temp.Resize(in_value.NumRows(),
                       in_value.NumCols() + 1, kUndefined);
  in_value_temp.Range(0, in_value.NumRows(),
                      0, in_value.NumCols()).CopyFromMat(in_value);

  // Add the 1.0 at the end of each row "in_value_temp"
  in_value_temp.Range(0, in_value.NumRows(),
                      in_value.NumCols(), 1).Set(1.0);
  
  CuMatrix<BaseFloat> out_deriv_temp(out_deriv);

  CuMatrix<BaseFloat> row_products(2,
                                   in_value.NumRows());
  CuSubVector<BaseFloat> in_row_products(row_products, 0),
      out_row_products(row_products, 1);

  // These "scale" values get will get multiplied into the learning rate (faster
  // than having the matrices scaled inside the preconditioning code).
  BaseFloat in_scale, out_scale;
  
  preconditioner_in_.PreconditionDirections(&in_value_temp, &in_row_products,
                                            &in_scale);
  preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_row_products,
                                             &out_scale);

  // "scale" is a scaling factor coming from the PreconditionDirections calls
  // (it's faster to have them output a scaling factor than to have them scale
  // their outputs).
  BaseFloat scale = in_scale * out_scale;
  BaseFloat minibatch_scale = 1.0;
  
  if (max_change_per_sample_ > 0.0)
    minibatch_scale = GetScalingFactor(in_row_products, scale,
                                       &out_row_products);
  
  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                              0, in_value_temp.NumRows(),
                                              0, in_value_temp.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
  CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());
  
  precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);
  
  BaseFloat local_lrate = scale * minibatch_scale * learning_rate_;
  bias_params_.AddMatVec(local_lrate, out_deriv_temp, kTrans,
                         precon_ones, 1.0);
  linear_params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
}

void BlockAffineComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
}

void BlockAffineComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  
  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

BaseFloat BlockAffineComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const BlockAffineComponent *other =
      dynamic_cast<const BlockAffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

Component* BlockAffineComponent::Copy() const {
  BlockAffineComponent *ans = new BlockAffineComponent();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->num_blocks_ = num_blocks_;
  return ans;
}

void BlockAffineComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
}

void BlockAffineComponent::Add(BaseFloat alpha,
                               const UpdatableComponent &other_in) {
  const BlockAffineComponent *other =
      dynamic_cast<const BlockAffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void BlockAffineComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32, // num_chunks
                                     CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), bias_params_.Dim());

  // The matrix has a block structure where each matrix has input dim
  // (#rows) equal to input_block_dim.  The blocks are stored in linear_params_
  // as [ M
  //      N
  //      O ] but we actually treat it as:
  // [ M 0 0
  //   0 N 0
  //   0 0 O ]
  int32 input_block_dim = linear_params_.NumCols(),
       output_block_dim = linear_params_.NumRows() / num_blocks_,
             num_frames = in.NumRows();
  KALDI_ASSERT(in.NumCols() == input_block_dim * num_blocks_);
  KALDI_ASSERT(out->NumCols() == output_block_dim * num_blocks_);
  KALDI_ASSERT(in.NumRows() == out->NumRows());

  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  
  for (int32 b = 0; b < num_blocks_; b++) {
    CuSubMatrix<BaseFloat> in_block(in, 0, num_frames,
                                  b * input_block_dim, input_block_dim),
        out_block(*out, 0, num_frames,
                  b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);
    out_block.AddMatMat(1.0, in_block, kNoTrans, param_block, kTrans, 1.0);
  }
}

void BlockAffineComponent::UpdateSimple(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  int32 input_block_dim = linear_params_.NumCols(),
      output_block_dim = linear_params_.NumRows() / num_blocks_,
      num_frames = in_value.NumRows();

  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  for (int32 b = 0; b < num_blocks_; b++) {
    CuSubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        b * input_block_dim,
                                        input_block_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);
    // Update the parameters.
    param_block.AddMatMat(learning_rate_, out_deriv_block, kTrans,
                          in_value_block, kNoTrans, 1.0);
  }
}

void BlockAffineComponent::Backprop(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    int32, // num_chunks
    Component *to_update_in,
    CuMatrix<BaseFloat> *in_deriv) const {
  // This code mirrors the code in Propagate().
  int32 num_frames = in_value.NumRows();
  BlockAffineComponent *to_update = dynamic_cast<BlockAffineComponent*>(
      to_update_in);
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  int32 input_block_dim = linear_params_.NumCols(),
       output_block_dim = linear_params_.NumRows() / num_blocks_;
  KALDI_ASSERT(in_value.NumCols() == input_block_dim * num_blocks_);
  KALDI_ASSERT(out_deriv.NumCols() == output_block_dim * num_blocks_);

  for (int32 b = 0; b < num_blocks_; b++) {
    CuSubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        b * input_block_dim,
                                        input_block_dim),
        in_deriv_block(*in_deriv, 0, num_frames,
                       b * input_block_dim, input_block_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);

    // Propagate the derivative back to the input.
    in_deriv_block.AddMatMat(1.0, out_deriv_block, kNoTrans,
                             param_block, kNoTrans, 0.0);
  }
  if (to_update != NULL)
    to_update->Update(in_value, out_deriv);
}


void BlockAffineComponent::Init(BaseFloat learning_rate,
                                int32 input_dim, int32 output_dim,
                                BaseFloat param_stddev,
                                BaseFloat bias_stddev,
                                int32 num_blocks) {
  UpdatableComponent::Init(learning_rate);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  KALDI_ASSERT(input_dim % num_blocks == 0 && output_dim % num_blocks == 0);

  linear_params_.Resize(output_dim, input_dim / num_blocks);
  bias_params_.Resize(output_dim);

  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  num_blocks_ = num_blocks;
}

void BlockAffineComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  int32 input_dim = -1, output_dim = -1, num_blocks = 1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  ok = ok && ParseFromString("num-blocks", &args, &num_blocks);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
      bias_stddev = 1.0;
  ParseFromString("param-stddev", &args, &param_stddev);
  ParseFromString("bias-stddev", &args, &bias_stddev);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, input_dim, output_dim,
       param_stddev, bias_stddev, num_blocks);
}
  

void BlockAffineComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BlockAffineComponent>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<NumBlocks>");
  ReadBasicType(is, binary, &num_blocks_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "</BlockAffineComponent>");  
}

void BlockAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<BlockAffineComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<NumBlocks>");
  WriteBasicType(os, binary, num_blocks_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</BlockAffineComponent>");  
}


int32 BlockAffineComponent::GetParameterDim() const {
  // Note: num_blocks_ should divide both InputDim() and OutputDim().
  return InputDim() * OutputDim() / num_blocks_;
}

void BlockAffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  int32 l = linear_params_.NumRows() * linear_params_.NumCols(),
      b = bias_params_.Dim();
  params->Range(0, l).CopyRowsFromMat(linear_params_);
  params->Range(l, b).CopyFromVec(bias_params_);
}
void BlockAffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  int32 l = linear_params_.NumRows() * linear_params_.NumCols(),
      b = bias_params_.Dim();
  linear_params_.CopyRowsFromVec(params.Range(0, l));
  bias_params_.CopyFromVec(params.Range(l, b));
}


void BlockAffineComponentPreconditioned::Init(BaseFloat learning_rate,
                                              int32 input_dim, int32 output_dim,
                                              BaseFloat param_stddev,
                                              BaseFloat bias_stddev,
                                              int32 num_blocks,
                                              BaseFloat alpha) {
  BlockAffineComponent::Init(learning_rate, input_dim, output_dim,
                             param_stddev, bias_stddev, num_blocks);
  is_gradient_ = false;
  KALDI_ASSERT(alpha > 0.0);
  alpha_ = alpha;
}

void BlockAffineComponentPreconditioned::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  BaseFloat alpha = 4.0;
  int32 input_dim = -1, output_dim = -1, num_blocks = 1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("alpha", &args, &alpha);
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  ok = ok && ParseFromString("num-blocks", &args, &num_blocks);
  
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
      bias_stddev = 1.0;
  ParseFromString("param-stddev", &args, &param_stddev);
  ParseFromString("bias-stddev", &args, &bias_stddev);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, input_dim, output_dim,
       param_stddev, bias_stddev, num_blocks,
       alpha);
}

void BlockAffineComponentPreconditioned::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient)
    is_gradient_ = true;
  BlockAffineComponent::SetZero(treat_as_gradient);
}  

void BlockAffineComponentPreconditioned::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BlockAffineComponentPreconditioned>",
                       "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<NumBlocks>");
  ReadBasicType(is, binary, &num_blocks_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</BlockAffineComponentPreconditioned>");  
}

void BlockAffineComponentPreconditioned::Write(std::ostream &os,
                                               bool binary) const {
  WriteToken(os, binary, "<BlockAffineComponentPreconditioned>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<NumBlocks>");
  WriteBasicType(os, binary, num_blocks_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</BlockAffineComponentPreconditioned>");  
}

Component* BlockAffineComponentPreconditioned::Copy() const {
  BlockAffineComponentPreconditioned *ans = new
      BlockAffineComponentPreconditioned();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->num_blocks_ = num_blocks_;
  ans->alpha_ = alpha_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}

void BlockAffineComponentPreconditioned::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  if (is_gradient_) {
    UpdateSimple(in_value, out_deriv);
    // does the baseline update with no preconditioning.
    return;
  }
  int32 input_block_dim = linear_params_.NumCols(),
      output_block_dim = linear_params_.NumRows() / num_blocks_,
      num_frames = in_value.NumRows();

  CuMatrix<BaseFloat> in_value_temp(num_frames, input_block_dim + 1, kUndefined),
      in_value_precon(num_frames, input_block_dim + 1, kUndefined);
  in_value_temp.Set(1.0); // so last row will have value 1.0.
  CuSubMatrix<BaseFloat> in_value_temp_part(in_value_temp, 0, num_frames,
                                            0, input_block_dim); // all but last 1.0
  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_precon, 0, num_frames,
                                            0, input_block_dim);
  CuVector<BaseFloat> precon_ones(num_frames);
  CuMatrix<BaseFloat> out_deriv_precon(num_frames, output_block_dim, kUndefined);
  
  for (int32 b = 0; b < num_blocks_; b++) {
    CuSubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        b * input_block_dim,
                                        input_block_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);
    in_value_temp_part.CopyFromMat(in_value_block);

    PreconditionDirectionsAlphaRescaled(in_value_temp, alpha_,
                                        &in_value_precon);
    PreconditionDirectionsAlphaRescaled(out_deriv_block, alpha_,
                                        &out_deriv_precon);
    
    
    // Update the parameters.
    param_block.AddMatMat(learning_rate_, out_deriv_precon, kTrans,
                          in_value_precon_part, kNoTrans, 1.0);
    precon_ones.CopyColFromMat(in_value_precon, input_block_dim);
    bias_params_.Range(b * output_block_dim, output_block_dim).
        AddMatVec(learning_rate_, out_deriv_precon, kTrans,
                  precon_ones, 1.0);
  }
}


void PermuteComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PermuteComponent>", "<Reorder>");
  ReadIntegerVector(is, binary, &reorder_);
  ExpectToken(is, binary, "</PermuteComponent>");
}

void PermuteComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PermuteComponent>");
  WriteToken(os, binary, "<Reorder>");
  WriteIntegerVector(os, binary, reorder_);
  WriteToken(os, binary, "</PermuteComponent>");
}

void PermuteComponent::Init(int32 dim) {
  KALDI_ASSERT(dim > 0);
  reorder_.resize(dim);
  for (int32 i = 0; i < dim; i++) reorder_[i] = i;
  std::random_shuffle(reorder_.begin(), reorder_.end());
}

void PermuteComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  bool ok = ParseFromString("dim", &args, &dim);
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim);
}

void PermuteComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                 int32, // num_chunks
                                 CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), in.NumCols());
  std::vector<int32> reverse_reorder(reorder_.size());
  for (size_t i = 0; i < reorder_.size(); i++)
    reverse_reorder[reorder_[i]] = i;
  out->CopyCols(in, reverse_reorder);
}

void PermuteComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                int32, // num_chunks
                                Component *to_update,
                                CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  KALDI_ASSERT(out_deriv.NumCols() == OutputDim());
  in_deriv->CopyCols(out_deriv, reorder_);
}

void SumGroupComponent::Init(const std::vector<int32> &sizes) {
  KALDI_ASSERT(!sizes.empty());
  std::vector<Int32Pair> cpu_vec(sizes.size());
  std::vector<int32> reverse_cpu_vec;
  int32 cur_index = 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    KALDI_ASSERT(sizes[i] > 0);
    cpu_vec[i].first = cur_index;
    cpu_vec[i].second = cur_index + sizes[i];
    cur_index += sizes[i];
    for (int32 j = cpu_vec[i].first; j < cpu_vec[i].second; j++)
      reverse_cpu_vec.push_back(i);
  }
  this->indexes_ = cpu_vec;
  this->reverse_indexes_ = reverse_cpu_vec;
  this->input_dim_ = cur_index;
  this->output_dim_ = sizes.size();
}

void SumGroupComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  std::vector<int32> sizes;
  bool ok = ParseFromString("sizes", &args, &sizes);

  if (!ok || !args.empty() || sizes.empty())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  this->Init(sizes);
}
  
Component* SumGroupComponent::Copy() const {
  SumGroupComponent *ans = new SumGroupComponent();
  ans->indexes_ = indexes_;
  ans->reverse_indexes_ = reverse_indexes_;
  ans->input_dim_ = input_dim_;
  ans->output_dim_ = output_dim_;
  return ans;
}

void SumGroupComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SumGroupComponent>", "<Sizes>");
  std::vector<int32> sizes;
  ReadIntegerVector(is, binary, &sizes);
  ExpectToken(is, binary, "<SumGroupComponent>");
  this->Init(sizes);
}

void SumGroupComponent::GetSizes(std::vector<int32> *sizes) const {
  std::vector<Int32Pair> indexes;
  indexes_.CopyToVec(&indexes);
  sizes->resize(indexes.size());
  for (size_t i = 0; i < indexes.size(); i++) {
    (*sizes)[i] = indexes[i].second - indexes[i].first;
    if (i == 0) { KALDI_ASSERT(indexes[i].first == 0); }
    else { KALDI_ASSERT(indexes[i].first == indexes[i-1].second); }
    KALDI_ASSERT(indexes[i].second > indexes[i].first);
    (*sizes)[i] = indexes[i].second - indexes[i].first;
  }
}

void SumGroupComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SumGroupComponent>");
  WriteToken(os, binary, "<Sizes>");
  std::vector<int32> sizes;
  this->GetSizes(&sizes);
  WriteIntegerVector(os, binary, sizes);
  WriteToken(os, binary, "<SumGroupComponent>");
}

void SumGroupComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                  int32 num_chunks,
                                  CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), this->OutputDim(), kUndefined);
  out->SumColumnRanges(in, indexes_);
}

void SumGroupComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value,
                                 const CuMatrixBase<BaseFloat> &, // out_value,
                                 const CuMatrixBase<BaseFloat> &out_deriv,
                                 int32 num_chunks,
                                 Component *to_update,
                                 CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  in_deriv->CopyCols(out_deriv, reverse_indexes_);
}


std::string SpliceComponent::Info() const {
  std::stringstream stream;
  stream << Component::Info() << ", context=" << left_context_ << "/" << right_context_;
  if (const_component_dim_ != 0)
    stream << ", const_component_dim=" << const_component_dim_;

  return stream.str();
}

void SpliceComponent::Init(int32 input_dim, int32 left_context,
                           int32 right_context, int32 const_component_dim) {
  input_dim_ = input_dim;
  const_component_dim_ = const_component_dim;
  left_context_ = left_context;
  right_context_ = right_context;
  KALDI_ASSERT(input_dim_ > 0 && left_context >= 0 && right_context >= 0);
  KALDI_ASSERT(const_component_dim_ >= 0 && const_component_dim_ < input_dim_);
}


// e.g. args == "input-dim=10 left-context=2 right-context=2
void SpliceComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 input_dim, left_context, right_context, const_component_dim = 0;
  bool ok = ParseFromString("input-dim", &args, &input_dim);
  ok = ParseFromString("left-context", &args, &left_context) && ok;
  ok = ParseFromString("right-context", &args, &right_context) && ok;
  ParseFromString("const-component-dim", &args, &const_component_dim);

  if (!ok || !args.empty() || input_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(input_dim, left_context, right_context, const_component_dim);
}

int32 SpliceComponent::OutputDim() const {
  return (input_dim_  - const_component_dim_)
      * (1 + left_context_ + right_context_)
      + const_component_dim_;
}

void SpliceComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                int32 num_chunks,
                                CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() > 0 && num_chunks > 0);
  if (in.NumRows() % num_chunks != 0)
    KALDI_ERR << "Number of chunks " << num_chunks << "does not divide "
              << "number of frames " << in.NumRows();
  int32 input_chunk_size = in.NumRows() / num_chunks,
       output_chunk_size = input_chunk_size - left_context_ - right_context_,
               input_dim = in.NumCols(),
              output_dim = OutputDim();
  if (output_chunk_size <= 0)
    KALDI_ERR << "Splicing features: output will have zero dimension. "
              << "Probably a code error.";
  out->Resize(num_chunks * output_chunk_size, output_dim);

  // 'indexes' is, for each index from 0 to (left_context_+right_context_+1)-1,
  // then for each row of "out", the corresponding row of "in" that we copy from.
  int32 num_splice = left_context_ + right_context_ + 1,
      const_dim = const_component_dim_;
  std::vector<std::vector<int32> > indexes(num_splice);
  // const_component_dim_ != 0, "const_indexes" will be used to determine which
  // row of "in" we copy the last part of each row of "out" from (this part is
  // not subject to splicing, it's assumed constant for each frame of "input".
  std::vector<int32> const_indexes(const_dim == 0 ? 0 : out->NumRows());

  for (int32 c = 0; c < num_splice; c++) 
    indexes[c].resize(out->NumRows());

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    for (int32 c = 0; c < num_splice; c++) {
      for (int32 offset = 0; offset < output_chunk_size; offset++) {
        indexes[c][chunk * output_chunk_size + offset] =
            chunk * input_chunk_size + c + offset;
      }
    }
    if (const_dim != 0) {
      for (int32 offset = 0; offset < output_chunk_size; offset++)
        const_indexes[chunk * output_chunk_size + offset] =
            chunk * input_chunk_size + offset; // there is
      // an arbitrariness here; since we assume the const_component
      // is constant within a chunk, it doesn't matter from where we copy.
    }
  }
  for (int32 c = 0; c < num_splice; c++) {
    int32 dim = input_dim - const_dim; // dimension we
    // are splicing
    CuSubMatrix<BaseFloat> in_part(in, 0, in.NumRows(),
                                   0, dim),
        out_part(*out, 0, out->NumRows(),
                 c * dim, dim);
    out_part.CopyRows(in_part, indexes[c]);
  }
  if (const_dim != 0) {
    CuSubMatrix<BaseFloat> in_part(in, 0, in.NumRows(),
                                   in.NumCols() - const_dim, const_dim),
        out_part(*out, 0, out->NumRows(),
                 out->NumCols() - const_dim, const_dim);
    out_part.CopyRows(in_part, const_indexes);
  }
}

void SpliceComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                               const CuMatrixBase<BaseFloat> &, // out_value,
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               int32 num_chunks,
                               Component *to_update, // may == "this".
                               CuMatrix<BaseFloat> *in_deriv) const {
 
  KALDI_ASSERT(out_deriv.NumRows() > 0 && num_chunks > 0);

  if (out_deriv.NumRows() % num_chunks != 0)
    KALDI_ERR << "Number of chunks " << num_chunks << "does not divide "
              << "number of frames " << out_deriv.NumRows();
  
  int32 output_chunk_size = out_deriv.NumRows() / num_chunks,
      input_chunk_size = output_chunk_size + left_context_ + right_context_,
      output_dim = out_deriv.NumCols(),
      input_dim = InputDim();
 
  KALDI_ASSERT( OutputDim() == output_dim );

  in_deriv->Resize(num_chunks * input_chunk_size, input_dim, kUndefined);

  int32 num_splice = left_context_ + right_context_ + 1,
      const_dim = const_component_dim_;
  // 'indexes' is, for each index from 0 to num_splice - 1,
  // then for each row of "in_deriv", the corresponding row of "out_deriv" that
  // we add, or -1 if.
    
  std::vector<std::vector<int32> > indexes(num_splice);
  // const_dim != 0, "const_indexes" will be used to determine which
  // row of "in" we copy the last part of each row of "out" from (this part is
  // not subject to splicing, it's assumed constant for each frame of "input".
  std::vector<int32> const_indexes(const_dim == 0 ? 0 : in_deriv->NumRows(),
                                   -1);

  for (int32 c = 0; c < indexes.size(); c++) 
    indexes[c].resize(in_deriv->NumRows(), -1); // set to -1 by default,
  // this gets interpreted by the CopyRows() code as a signal to zero the output...

  int32 dim = input_dim - const_dim; // dimension we are splicing

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    for (int32 c = 0; c < num_splice; c++)
      for (int32 offset = 0; offset < output_chunk_size; offset++)
        indexes[c][chunk * input_chunk_size + c + offset] =
            chunk * output_chunk_size + offset;

    // Note: when changing over to the CUDA code, we also changed
    // how the derivatives are propagated through the splicing layer
    // for the const-component-dim.  The code was never being used,
    // so it doesn't matter.  The way we now do it probably makes more
    // sense (to get the derivative, you'd have to sum over time, not
    // pick an arbitrary time)
    if (const_dim != 0)
      for (int32 offset = 0; offset < output_chunk_size; offset++)
        const_indexes[chunk * input_chunk_size + offset] =
            chunk * output_chunk_size + offset;
  }
    
  CuMatrix<BaseFloat> temp_mat(in_deriv->NumRows(), dim, kUndefined);
    
  for (int32 c = 0; c < num_splice; c++) {
    int32 dim = input_dim - const_dim; // dimension we
    // are splicing
    CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, out_deriv.NumRows(),
                                          c * dim, dim),
        in_deriv_part(*in_deriv, 0, in_deriv->NumRows(),
                      0, dim);
    if (c == 0)
      in_deriv_part.CopyRows(out_deriv_part, indexes[c]);
    else {
      temp_mat.CopyRows(out_deriv_part, indexes[c]);
      in_deriv_part.AddMat(1.0, temp_mat);
    }
  }
  if (const_dim != 0) {
    CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, out_deriv.NumRows(),
                                          out_deriv.NumCols() - const_dim,
                                          const_dim),
        in_deriv_part(*in_deriv, 0, in_deriv->NumRows(),
                      in_deriv->NumCols() - const_dim, const_dim);
    in_deriv_part.CopyRows(out_deriv_part, const_indexes);
  }
}

Component *SpliceComponent::Copy() const {
  SpliceComponent *ans = new SpliceComponent();
  ans->input_dim_ = input_dim_;
  ans->left_context_ = left_context_;
  ans->right_context_ = right_context_;
  ans->const_component_dim_ = const_component_dim_;
  return ans;
}

void SpliceComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SpliceComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "<ConstComponentDim>");
  ReadBasicType(is, binary, &const_component_dim_);
  ExpectToken(is, binary, "</SpliceComponent>");
}

void SpliceComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SpliceComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "<ConstComponentDim>");
  WriteBasicType(os, binary, const_component_dim_);
  WriteToken(os, binary, "</SpliceComponent>");  
}


std::string SpliceMaxComponent::Info() const {
  std::stringstream stream;
  stream << Component::Info() << ", context=" << left_context_
         << "/" << right_context_;
  return stream.str();
}

void SpliceMaxComponent::Init(int32 dim, int32 left_context,
                              int32 right_context) {
  dim_ = dim;
  left_context_ = left_context;
  right_context_ = right_context;
  KALDI_ASSERT(dim_ > 0 && left_context >= 0 && right_context >= 0);
}


// e.g. args == "dim=10 left-context=2 right-context=2
void SpliceMaxComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim, left_context, right_context;
  bool ok = ParseFromString("dim", &args, &dim);
  ok = ParseFromString("left-context", &args, &left_context) && ok;
  ok = ParseFromString("right-context", &args, &right_context) && ok;
  
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim, left_context, right_context);
}

void SpliceMaxComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                   int32 num_chunks,
                                   CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() > 0 && in.NumCols() == InputDim());
  if (in.NumRows() % num_chunks != 0)
    KALDI_ERR << "Number of chunks " << num_chunks << "does not divide "
              << "number of frames " << in.NumRows();
  int32 input_chunk_size = in.NumRows() / num_chunks,
       output_chunk_size = input_chunk_size - left_context_ - right_context_,
                     dim = in.NumCols();
  if (output_chunk_size <= 0)
    KALDI_ERR << "Splicing features: output will have zero dimension. "
              << "Probably a code error.";
  out->Resize(num_chunks * output_chunk_size, dim);
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    CuSubMatrix<BaseFloat> input_chunk(in,
                                     chunk * input_chunk_size, input_chunk_size,
                                     0, dim),
                        output_chunk(*out,
                                     chunk * output_chunk_size, output_chunk_size,
                                     0, dim);
    for (int32 offset = 0;
         offset < 1 + left_context_ + right_context_;
         offset++) {
      CuSubMatrix<BaseFloat> input_chunk_part(input_chunk,
                                            offset, output_chunk_size, 0, dim);
      if (offset == 0) output_chunk.CopyFromMat(input_chunk_part);
      else {
        output_chunk.Max(input_chunk_part);
      }
    }
  }  
}

void SpliceMaxComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &, // out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  int32 num_chunks,
                                  Component *to_update, // may == "this".
                                  CuMatrix<BaseFloat> *in_deriv) const {
 KALDI_ASSERT(out_deriv.NumRows() > 0 && num_chunks > 0);

  if (out_deriv.NumRows() % num_chunks != 0)
    KALDI_ERR << "Number of chunks " << num_chunks << "does not divide "
              << "number of frames " << out_deriv.NumRows();
  
  int32 output_chunk_size = out_deriv.NumRows() / num_chunks,
         input_chunk_size = output_chunk_size + left_context_ + right_context_,
                      dim = out_deriv.NumCols();

  KALDI_ASSERT(dim == InputDim());

  in_deriv->Resize(num_chunks * input_chunk_size, dim); // Will zero it.
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    CuSubMatrix<BaseFloat> in_deriv_chunk(*in_deriv, 
                                        chunk * input_chunk_size,
                                        input_chunk_size, 
                                        0, dim),
                         in_value_chunk(in_value,
                                        chunk * input_chunk_size,
                                        input_chunk_size, 
                                        0, dim),
                        out_deriv_chunk(out_deriv,
                                        chunk * output_chunk_size,
                                        output_chunk_size,
                                        0, dim);
    for (int32 r = 0; r < out_deriv_chunk.NumRows(); r++) {
      for (int32 c = 0; c < dim; c++) {
        int32 in_r_begin = r, in_r_end = r + left_context_ + right_context_ + 1;
        int32 in_r_max = -1;
        BaseFloat max_input = -std::numeric_limits<BaseFloat>::infinity();
        for (int32 in_r = in_r_begin; in_r < in_r_end; in_r++) {
          BaseFloat input = in_value_chunk(in_r, c);
          if (input > max_input) {
            max_input = input;
            in_r_max = in_r;
          }
        }
        KALDI_ASSERT(in_r_max != -1);
        (*in_deriv)(in_r_max, c) += out_deriv_chunk(r, c);
      }
    }
  }
}

Component *SpliceMaxComponent::Copy() const {
  SpliceMaxComponent *ans = new SpliceMaxComponent();
  ans->Init(dim_, left_context_, right_context_);
  return ans;
}

void SpliceMaxComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SpliceMaxComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "</SpliceMaxComponent>");
}

void SpliceMaxComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SpliceMaxComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "</SpliceMaxComponent>");  
}

std::string DctComponent::Info() const {
  std::stringstream stream;
  stream << Component::Info() << ", dct_dim=" << dct_mat_.NumCols();
  if (dct_mat_.NumCols() != dct_mat_.NumRows())
    stream << ", dct_keep_dim=" << dct_mat_.NumRows();

  return stream.str();
}

void DctComponent::Init(int32 dim, int32 dct_dim, bool reorder, int32 dct_keep_dim) {
  int dct_keep_dim_ = (dct_keep_dim > 0) ? dct_keep_dim : dct_dim;

  KALDI_ASSERT(dim > 0 && dct_dim > 0);
  KALDI_ASSERT(dim % dct_dim == 0); // dct_dim must divide dim.
  KALDI_ASSERT(dct_dim >= dct_keep_dim_)
  dim_ = dim;
  dct_mat_.Resize(dct_keep_dim_, dct_dim);
  reorder_ = reorder;
  Matrix<BaseFloat> dct_mat(dct_keep_dim_, dct_dim);
  ComputeDctMatrix(&dct_mat);
  dct_mat_ = dct_mat;
}



void DctComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim, dct_dim, dct_keep_dim = 0;
  bool reorder = false;

  bool ok = ParseFromString("dim", &args, &dim);
  ok = ParseFromString("dct-dim", &args, &dct_dim) && ok;
  ok = ParseFromString("reorder", &args, &reorder) && ok;
  ParseFromString("dct-keep-dim", &args, &dct_keep_dim);

  if (!ok || !args.empty() || dim <= 0 || dct_dim <= 0 || dct_keep_dim < 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim, dct_dim, reorder, dct_keep_dim);
}

void DctComponent::Reorder(CuMatrixBase<BaseFloat> *mat, bool reverse) const {
  // reorders into contiguous blocks of dize "dct_dim_", assuming that
  // such blocks were interlaced before.  if reverse==true, does the
  // reverse.
  int32 dct_dim = dct_mat_.NumCols(),
      dct_keep_dim = dct_mat_.NumRows(),
      block_size_in = dim_ / dct_dim,
      block_size_out = dct_keep_dim;

  //This does not necesarily needs to be true anymore -- output must be reordered as well, but the dimension differs... 
  //KALDI_ASSERT(mat->NumCols() == dim_);
  if (reverse) std::swap(block_size_in, block_size_out);

  CuVector<BaseFloat> temp(mat->NumCols());
  for (int32 i = 0; i < mat->NumRows(); i++) {
    CuSubVector<BaseFloat> row(*mat, i);
    int32 num_blocks_in = block_size_out;
    for (int32 b = 0; b < num_blocks_in; b++) {
      for (int32 j = 0; j < block_size_in; j++) {
        temp(j * block_size_out + b) = row(b * block_size_in + j);
      }
    }
    row.CopyFromVec(temp);
  }
}

void DctComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                             int32, // num_chunks
                             CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == InputDim());
  
  int32 dct_dim = dct_mat_.NumCols(),
        dct_keep_dim = dct_mat_.NumRows(),
        num_chunks = dim_ / dct_dim,
        num_rows = in.NumRows();
  
  out->Resize(num_rows, num_chunks * dct_keep_dim);

  CuMatrix<BaseFloat> in_tmp;
  if (reorder_) {
    in_tmp = in;
    Reorder(&in_tmp, false);
  }
  
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    CuSubMatrix<BaseFloat> in_mat(reorder_ ? in_tmp : in,
                                0, num_rows, dct_dim * chunk, dct_dim),
                        out_mat(*out, 
                                0, num_rows, dct_keep_dim * chunk, dct_keep_dim);

    out_mat.AddMatMat(1.0, in_mat, kNoTrans, dct_mat_, kTrans, 0.0);
  }
  if (reorder_)
    Reorder(out, true);
}

void DctComponent::Backprop(const CuMatrixBase<BaseFloat>&, // in_value,
                            const CuMatrixBase<BaseFloat>&, // out_value,
                            const CuMatrixBase<BaseFloat> &out_deriv,
                            int32, // num_chunks
                            Component*,// to_update
                            CuMatrix<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(out_deriv.NumCols() == OutputDim());

  int32 dct_dim = dct_mat_.NumCols(),
        dct_keep_dim = dct_mat_.NumRows(),
        num_chunks = dim_ / dct_dim,
        num_rows = out_deriv.NumRows();

  in_deriv->Resize(num_rows, dim_);
  
  CuMatrix<BaseFloat> out_deriv_tmp;
  if (reorder_) {
    out_deriv_tmp = out_deriv;
    Reorder(&out_deriv_tmp, false);
  }
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    CuSubMatrix<BaseFloat> in_deriv_mat(*in_deriv,
                                      0, num_rows, dct_dim * chunk, dct_dim),
                        out_deriv_mat(reorder_ ? out_deriv_tmp : out_deriv,
                                      0, num_rows, dct_keep_dim * chunk, dct_keep_dim);

    // Note: in the reverse direction the DCT matrix is transposed.  This is
    // normal when computing derivatives; the necessity for the transpose is
    // obvious if you consider what happens when the input and output dims
    // differ.
    in_deriv_mat.AddMatMat(1.0, out_deriv_mat, kNoTrans,
                           dct_mat_, kNoTrans, 0.0);
  }
  if (reorder_)
    Reorder(in_deriv, true);
}

Component* DctComponent::Copy() const {
  DctComponent *ans = new DctComponent();
  ans->dct_mat_ = dct_mat_;
  ans->dim_ = dim_;
  ans->reorder_ = reorder_;
  return ans;
}

void DctComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DctComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<DctDim>");
  int32 dct_dim = dct_mat_.NumCols();
  WriteBasicType(os, binary, dct_dim);
  WriteToken(os, binary, "<Reorder>");
  WriteBasicType(os, binary, reorder_);
  WriteToken(os, binary, "<DctKeepDim>");
  int32 dct_keep_dim = dct_mat_.NumRows();
  WriteBasicType(os, binary, dct_keep_dim);
  WriteToken(os, binary, "</DctComponent>");  
}

void DctComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<DctComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  
  ExpectToken(is, binary, "<DctDim>");
  int32 dct_dim; 
  ReadBasicType(is, binary, &dct_dim);
  
  ExpectToken(is, binary, "<Reorder>");
  ReadBasicType(is, binary, &reorder_);

  int32 dct_keep_dim = dct_dim;
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<DctKeepDim>") {
    ReadBasicType(is, binary, &dct_keep_dim);
    ExpectToken(is, binary, "</DctComponent>");
  } else if (token != "</DctComponent>") {
    KALDI_ERR << "Expected token \"</DctComponent>\", got instead \""
              << token << "\".";
  }

  KALDI_ASSERT(dct_dim > 0 && dim_ > 0 && dim_ % dct_dim == 0);
  Init(dim_, dct_dim, reorder_, dct_keep_dim);
  //idct_mat_.Resize(dct_keep_dim, dct_dim);
  //ComputeDctMatrix(&dct_mat_);
}

void FixedLinearComponent::InitFromString(std::string args) {
  std::string orig_args = args;
  std::string filename;
  bool ok = ParseFromString("matrix", &args, &filename);

  if (!ok || !args.empty()) 
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";

  bool binary;
  Input ki(filename, &binary);
  CuMatrix<BaseFloat> mat;
  mat.Read(ki.Stream(), binary);
  KALDI_ASSERT(mat.NumRows() != 0);
  Init(mat);
}


std::string FixedLinearComponent::Info() const {
  std::stringstream stream;
  BaseFloat mat_size = static_cast<BaseFloat>(mat_.NumRows())
      * static_cast<BaseFloat>(mat_.NumCols()),
      mat_stddev = std::sqrt(TraceMatMat(mat_, mat_, kTrans) /
                         mat_size); 
  stream << Component::Info() << ", params-stddev=" << mat_stddev;
  return stream.str();
}

void FixedLinearComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), mat_.NumRows());
  out->AddMatMat(1.0, in, kNoTrans, mat_, kTrans, 0.0);
}

void FixedLinearComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), mat_.NumCols());
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, mat_, kNoTrans, 0.0);
}

Component* FixedLinearComponent::Copy() const {
  FixedLinearComponent *ans = new FixedLinearComponent();
  ans->Init(mat_);
  return ans;
}


void FixedLinearComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedLinearComponent>");
  WriteToken(os, binary, "<CuMatrix>");
  mat_.Write(os, binary);
  WriteToken(os, binary, "</FixedLinearComponent>");  
}

void FixedLinearComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedLinearComponent>", "<CuMatrix>");
  mat_.Read(is, binary);
  ExpectToken(is, binary, "</FixedLinearComponent>");
}

void FixedAffineComponent::Init(const CuMatrixBase<BaseFloat> &mat) {
  KALDI_ASSERT(mat.NumCols() > 1);
  linear_params_ = mat.Range(0, mat.NumRows(),
                             0, mat.NumCols() - 1);
  bias_params_.Resize(mat.NumRows());
  bias_params_.CopyColFromMat(mat, mat.NumCols() - 1);
}


void FixedAffineComponent::InitFromString(std::string args) {
  std::string orig_args = args;
  std::string filename;
  bool ok = ParseFromString("matrix", &args, &filename);

  if (!ok || !args.empty()) 
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";

  bool binary;
  Input ki(filename, &binary);
  CuMatrix<BaseFloat> mat;
  mat.Read(ki.Stream(), binary);
  KALDI_ASSERT(mat.NumRows() != 0);
  Init(mat);
}


std::string FixedAffineComponent::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols()),
      linear_params_stddev =
      std::sqrt(TraceMatMat(linear_params_,
                            linear_params_, kTrans) /
                linear_params_size),
      bias_params_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                                     bias_params_.Dim());
      
  stream << Component::Info() << ", linear-params-stddev=" << linear_params_stddev
         << ", bias-params-stddev=" << bias_params_stddev;
  return stream.str();
}

void FixedAffineComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), linear_params_.NumRows());
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 0.0);
  out->AddVecToRows(1.0, bias_params_);
}

void FixedAffineComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), linear_params_.NumCols());
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans, 0.0);
}

Component* FixedAffineComponent::Copy() const {
  FixedAffineComponent *ans = new FixedAffineComponent();
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  return ans;
}


void FixedAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedAffineComponent>");
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</FixedAffineComponent>");  
}

void FixedAffineComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedAffineComponent>", "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);  
  ExpectToken(is, binary, "</FixedAffineComponent>");
}


void FixedScaleComponent::Init(const CuVectorBase<BaseFloat> &scales) {
  KALDI_ASSERT(scales.Dim() != 0);
  scales_ = scales;
}

void FixedScaleComponent::InitFromString(std::string args) {
  std::string orig_args = args;
  std::string filename;
  bool ok = ParseFromString("scales", &args, &filename);

  if (!ok || !args.empty()) 
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";

  CuVector<BaseFloat> vec;
  ReadKaldiObject(filename, &vec);
  Init(vec);
}


std::string FixedScaleComponent::Info() const {
  std::stringstream stream;
  BaseFloat scales_size = static_cast<BaseFloat>(scales_.Dim()),
      scales_mean = scales_.Sum() / scales_size,
      scales_stddev = std::sqrt(VecVec(scales_, scales_) / scales_size)
       - (scales_mean * scales_mean);
  stream << Component::Info() << ", scales-mean=" << scales_mean
         << ", scales-stddev=" << scales_stddev;
  return stream.str();
}

void FixedScaleComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     CuMatrix<BaseFloat> *out) const {
  *out = in;
  out->MulColsVec(scales_);
}

void FixedScaleComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    CuMatrix<BaseFloat> *in_deriv) const {
  *in_deriv = out_deriv;
  in_deriv->MulColsVec(scales_);
}

Component* FixedScaleComponent::Copy() const {
  FixedScaleComponent *ans = new FixedScaleComponent();
  ans->scales_ = scales_;
  return ans;
}


void FixedScaleComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedScaleComponent>");
  WriteToken(os, binary, "<Scales>");
  scales_.Write(os, binary);
  WriteToken(os, binary, "</FixedScaleComponent>");  
}

void FixedScaleComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedScaleComponent>", "<Scales>");
  scales_.Read(is, binary);
  ExpectToken(is, binary, "</FixedScaleComponent>");
}

void FixedBiasComponent::Init(const CuVectorBase<BaseFloat> &bias) {
  KALDI_ASSERT(bias.Dim() != 0);
  bias_ = bias;
}

void FixedBiasComponent::InitFromString(std::string args) {
  std::string orig_args = args;
  std::string filename;
  bool ok = ParseFromString("bias", &args, &filename);

  if (!ok || !args.empty()) 
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";

  CuVector<BaseFloat> vec;
  ReadKaldiObject(filename, &vec);
  Init(vec);
}


std::string FixedBiasComponent::Info() const {
  std::stringstream stream;
  BaseFloat bias_size = static_cast<BaseFloat>(bias_.Dim()),
      bias_mean = bias_.Sum() / bias_size,
      bias_stddev = std::sqrt(VecVec(bias_, bias_) / bias_size)
       - (bias_mean * bias_mean);
  stream << Component::Info() << ", bias-mean=" << bias_mean
         << ", bias-stddev=" << bias_stddev;
  return stream.str();
}

void FixedBiasComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     CuMatrix<BaseFloat> *out) const {
  *out = in;
  out->AddVecToRows(1.0, bias_, 1.0);
}

void FixedBiasComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    CuMatrix<BaseFloat> *in_deriv) const {
  *in_deriv = out_deriv;
}

Component* FixedBiasComponent::Copy() const {
  FixedBiasComponent *ans = new FixedBiasComponent();
  ans->bias_ = bias_;
  return ans;
}


void FixedBiasComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedBiasComponent>");
  WriteToken(os, binary, "<Bias>");
  bias_.Write(os, binary);
  WriteToken(os, binary, "</FixedBiasComponent>");  
}

void FixedBiasComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedBiasComponent>", "<Bias>");
  bias_.Read(is, binary);
  ExpectToken(is, binary, "</FixedBiasComponent>");
}




std::string DropoutComponent::Info() const {
  std::stringstream stream;
  stream << Component::Info() << ", dropout_proportion = "
         << dropout_proportion_ << ", dropout_scale = "
         << dropout_scale_;
  return stream.str();
}

void DropoutComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  BaseFloat dropout_proportion = 0.5, dropout_scale = 0.0;
  bool ok = ParseFromString("dim", &args, &dim);
  ParseFromString("dropout-proportion", &args, &dropout_proportion);
  ParseFromString("dropout-scale", &args, &dropout_scale);
  
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type DropoutComponent: \""
              << orig_args << "\"";
  Init(dim, dropout_proportion, dropout_scale);
}

void DropoutComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<DropoutComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<DropoutScale>");
  ReadBasicType(is, binary, &dropout_scale_);
  ExpectToken(is, binary, "<DropoutProportion>");
  ReadBasicType(is, binary, &dropout_proportion_);
  ExpectToken(is, binary, "</DropoutComponent>");
}

void DropoutComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DropoutComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<DropoutScale>");
  WriteBasicType(os, binary, dropout_scale_);
  WriteToken(os, binary, "<DropoutProportion>");
  WriteBasicType(os, binary, dropout_proportion_);
  WriteToken(os, binary, "</DropoutComponent>");  
}


void DropoutComponent::Init(int32 dim,
                            BaseFloat dropout_proportion,
                            BaseFloat dropout_scale){
  dim_ = dim;
  dropout_proportion_ = dropout_proportion;
  dropout_scale_ = dropout_scale;
}
  
void DropoutComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                     int32 num_chunks,
                     CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == this->InputDim());
  out->Resize(in.NumRows(), in.NumCols());

  BaseFloat dp = dropout_proportion_;
  KALDI_ASSERT(dp < 1.0 && dp >= 0.0);
  KALDI_ASSERT(dropout_scale_ <= 1.0 && dropout_scale_ >= 0.0);

  BaseFloat low_scale = dropout_scale_,
      high_scale = (1.0 - (dp * low_scale)) / (1.0 - dp),
      average = (low_scale * dp) +
                (high_scale * (1.0 - dp));
  KALDI_ASSERT(fabs(average - 1.0) < 0.01);

  out->Resize(in.NumRows(), in.NumCols(), kUndefined);

  // This const_cast is only safe assuming you don't attempt
  // to use multi-threaded code with the GPU.
  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(out);

  
  out->Add(-dp); // now, a proportion "dp" will be <0.0
  out->ApplyHeaviside(); // apply the function (x>0?1:0).  Now, a proportion "dp" will
                         // be zero and (1-dp) will be 1.0.
  if ((high_scale - low_scale) != 1.0)
    out->Scale(high_scale - low_scale); // now, "dp" are 0 and (1-dp) are "high_scale-low_scale".
  if (low_scale != 0.0)
    out->Add(low_scale); // now "dp" equal "low_scale" and (1.0-dp) equal "high_scale".

  out->MulElements(in);
}

void DropoutComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                int32, // num_chunks
                                Component *, // to_update
                                CuMatrix<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(SameDim(in_value, out_value) && SameDim(in_value, out_deriv));
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  in_deriv->AddMatMatDivMat(out_deriv, out_value, in_value);
}

Component* DropoutComponent::Copy() const {
  return new DropoutComponent(dim_,
                              dropout_proportion_,
                              dropout_scale_);
}

void AdditiveNoiseComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  BaseFloat stddev = 1.0;
  bool ok = ParseFromString("dim", &args, &dim);
  ParseFromString("stddev", &args, &stddev);  
  
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type AdditiveNoiseComponent: \""
              << orig_args << "\"";
  Init(dim, stddev);
}

void AdditiveNoiseComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<AdditiveNoiseComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<Stddev>");
  ReadBasicType(is, binary, &stddev_);
  ExpectToken(is, binary, "</AdditiveNoiseComponent>");
}

void AdditiveNoiseComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AdditiveNoiseComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Stddev>");
  WriteBasicType(os, binary, stddev_);
  WriteToken(os, binary, "</AdditiveNoiseComponent>");  
}

void AdditiveNoiseComponent::Init(int32 dim, BaseFloat stddev) {
  dim_ = dim;
  stddev_ = stddev;
}
  
void AdditiveNoiseComponent::Propagate(
    const CuMatrixBase<BaseFloat> &in,
    int32 num_chunks,
    CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == this->InputDim());
  *out = in;
  CuMatrix<BaseFloat> rand(in.NumRows(), in.NumCols());
  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&rand);
  out->AddMat(stddev_, rand);
}

} // namespace nnet2
} // namespace kaldi

