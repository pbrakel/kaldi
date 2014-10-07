#!/bin/bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this 
# if you're not on the CLSP grid.
data=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

. cmd.sh
. path.sh

# you might not want to do this for interactive shells.
set -e

# download the data.  Note: we're using the 100 hour setup for
# now; later in the script we'll download more and use it to train neural
# nets.
for part in dev-clean test-clean dev-other test-other train-clean-100; do
  local/download_and_untar.sh $data $data_url $part
done

# download the LM resources
local/download_lm.sh $lm_url data/local/lm || exit 1

# format the data as Kaldi data directories
for part in dev-clean test-clean dev-other test-other train-clean-100; do
  # use underscore-separated names in data directories.
  local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g) || exit 1
done

## Optional text corpus normalization and LM training
## These scripts are here primarily as a documentation of the process that has been
## used to build the LM. Most users of this recipe will NOT need/want to run
## this step
#local/lm/train_lm.sh $LM_CORPUS_ROOT \
#  data/local/lm/norm/tmp data/local/lm/norm/norm_texts data/local/lm || exit 1

## Optional G2P training scripts.
## As the LM training scripts above, this script is intended primarily to
## document our G2P model creation process
#local/g2p/train_g2p.sh data/local/dict/cmudict data/local/lm

local/prepare_dict.sh --nj 30 --cmd "$train_cmd" \
   data/local/lm data/local/lm data/local/dict || exit 1

utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

local/format_lms.sh data/local/lm || exit 1

# Create ConstArpaLm format language model for full trigram language model.
utils/build_const_arpa_lm.sh \
  data/local/lm/3-gram.arpa.gz data/lang data/lang_test_tglarge || exit 1;

mfccdir=mfcc
# spread the mfccs over various machines, as this data-set is quite large.
if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then 
  mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
  utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/librispeech/s5/$dir/$mfcc/storage \
    $mfccdir/storage
fi


for part in dev_clean test_clean dev_other test_other train_clean_100; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
  steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
done

# Make some small data subsets for early system-build stages.  Note, there are 29k
# utterances in the train_clean_100 directory which has 100 hours of data.
# For the monophone stages we select the shortest utterances, which should make it
# easier to align the data from a flat start.

utils/subset_data_dir.sh --shortest data/train_clean_100 2000 data/train_2kshort
utils/subset_data_dir.sh data/train_clean_100 5000 data/train_5k
utils/subset_data_dir.sh data/train_clean_100 10000 data/train_10k

# train a monophone system
steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
  data/train_2kshort data/lang exp/mono || exit 1;

# decode using the monophone model
(
  utils/mkgraph.sh --mono data/lang_test_tgsmall exp/mono exp/mono/graph_tgsmall || exit 1
  for test in dev_clean dev_other; do
    steps/decode.sh --nj 20 --cmd "$decode_cmd" \
      exp/mono/graph_tgsmall data/$test exp/mono/decode_tgsmall_$test
  done
)&

steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train_5k data/lang exp/mono exp/mono_ali_5k

# train a first delta + delta-delta triphone system on a subset of 5000 utterances
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_5k data/lang exp/mono_ali_5k exp/tri1 || exit 1;

# decode using the tri1 model
(
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri1 exp/tri1/graph_tgsmall || exit 1;
  for test in dev_clean dev_other; do
    steps/decode.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri1/graph_tgsmall data/$test exp/tri1/decode_tgsmall_$test || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri1/decode_{tgsmall,tgmed}_$test  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri1/decode_{tgsmall,tglarge}_$test || exit 1;
  done
)&

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train_10k data/lang exp/tri1 exp/tri1_ali_10k || exit 1;


# train an LDA+MLLT system.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train_10k data/lang exp/tri1_ali_10k exp/tri2b || exit 1;

# decode using the LDA+MLLT model
(
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri2b exp/tri2b/graph_tgsmall || exit 1;
  for test in dev_clean dev_other; do
    steps/decode.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri2b/graph_tgsmall data/$test exp/tri2b/decode_tgsmall_$test || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri2b/decode_{tgsmall,tgmed}_$test  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri2b/decode_{tgsmall,tglarge}_$test || exit 1;
  done
)&

# Align a 10k utts subset using the tri2b model
steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  --use-graphs true data/train_10k data/lang exp/tri2b exp/tri2b_ali_10k || exit 1;

# Train tri3b, which is LDA+MLLT+SAT on 10k utts
steps/train_sat.sh --cmd "$train_cmd" \
  2500 15000 data/train_10k data/lang exp/tri2b_ali_10k exp/tri3b || exit 1;

# decode using the tri3b model
(
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri3b exp/tri3b/graph_tgsmall || exit 1;
  for test in dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri3b/graph_tgsmall data/$test exp/tri3b/decode_tgsmall_$test || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri3b/decode_{tgsmall,tgmed}_$test  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri3b/decode_{tgsmall,tglarge}_$test || exit 1;
  done
)&

# align the entire train_clean_100 subset using the tri3b model
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train_clean_100 data/lang exp/tri3b exp/tri3b_ali_clean_100 || exit 1;

# train another LDA+MLLT+SAT system on the entire 100 hour subset
steps/train_sat.sh  --cmd "$train_cmd" \
  4200 40000 data/train_clean_100 data/lang exp/tri3b_ali_clean_100 exp/tri4b || exit 1;

# decode using the tri4b model
(
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri4b exp/tri4b/graph_tgsmall || exit 1;
  for test in dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri4b/graph_tgsmall data/$test exp/tri4b/decode_tgsmall_$test || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri4b/decode_{tgsmall,tgmed}_$test  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri4b/decode_{tgsmall,tglarge}_$test || exit 1;
  done
)&

# align train_clean_100 using the tri4b model
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_clean_100 data/lang exp/tri4b exp/tri4b_ali_clean_100 || exit 1;

# if you want at this point you can train and test NN model(s) on the 100 hour
# subset
local/nnet2/run_5a_clean_100.sh || exit 1

local/download_and_untar.sh $data $data_url train-clean-360 || exit 1;

# now add the "clean-360" subset to the mix ...
local/data_prep.sh $data/LibriSpeech/train-clean-360 data/train_clean_360 || exit 1
steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train_clean_360 \
  exp/make_mfcc/train_clean_360 $mfccdir || exit 1
steps/compute_cmvn_stats.sh data/train_clean_360 exp/make_mfcc/train_clean_360 $mfccdir || exit 1

# ... and then combine the two sets into a 460 hour one
utils/combine_data.sh data/train_clean_460 data/train_clean_100 data/train_clean_360 || exit 1

# align the new, combined set, using the tri4b model
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_clean_460 data/lang exp/tri4b exp/tri4b_ali_clean_460 || exit 1;

# create a larger SAT model, trained on the 460 hours of data.
steps/train_sat.sh  --cmd "$train_cmd" \
  5000 100000 data/train_clean_460 data/lang exp/tri4b_ali_clean_460 exp/tri5b || exit 1;

# decode using the tri5b model
(
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri5b exp/tri5b/graph_tgsmall || exit 1;
  for test in dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri5b/graph_tgsmall data/$test exp/tri5b/decode_tgsmall_$test || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri5b/decode_{tgsmall,tgmed}_$test  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri5b/decode_{tgsmall,tglarge}_$test || exit 1;
  done
)&

# train a NN model on the 460 hour set
local/nnet2/run_6a_clean_460.sh || exit 1

local/download_and_untar.sh $data $data_url train-other-500 || exit 1;

# prepare the 500 hour subset.
local/data_prep.sh $data/LibriSpeech/train-other-500 data/train_other_500 || exit 1
steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train_other_500 \
  exp/make_mfcc/train_other_500 $mfccdir || exit 1
steps/compute_cmvn_stats.sh data/train_other_500 exp/make_mfcc/train_other_500 $mfccdir || exit 1

# combine all the data
utils/combine_data.sh data/train_960 data/train_clean_460 data/train_other_500 || exit 1

steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_960 data/lang exp/tri5b exp/tri5b_ali_960 || exit 1;

# train a SAT model on the 960 hour mixed data.  Use the train_quick.sh script
# as it is faster.
steps/train_quick.sh --cmd "$train_cmd" \
  7000 150000 data/train_960 data/lang exp/tri5b_ali_960 exp/tri6b || exit 1;

# decode using the tri6b model
(
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri6b exp/tri6b/graph_tgsmall || exit 1;
  for test in dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri6b/graph_tgsmall data/$test exp/tri6b/decode_tgsmall_$test || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri6b/decode_{tgsmall,tgmed}_$test  || exit 1;
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri6b/decode_{tgsmall,tglarge}_$test || exit 1;
  done
)&

# train NN models on the entire dataset
local/nnet2/run_7a_960.sh || exit 1

## train models on cleaned-up data
## we've found that this isn't helpful-- see the comments in local/run_data_cleaning.sh
#local/run_data_cleaning.sh
