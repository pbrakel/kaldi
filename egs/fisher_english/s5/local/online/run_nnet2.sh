#!/bin/bash


. cmd.sh


stage=1
train_stage=-10
use_gpu=true
set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
  # the _a is in case I want to change the parameters.
  dir=exp/nnet2_online/nnet_a_gpu 
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet2_online/nnet_a
fi


if [ $stage -le 1 ]; then
  mkdir -p exp/nnet2_online
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 400000 \
    data/train_30k 512 exp/tri5a exp/nnet2_online/diag_ubm
fi

if [ $stage -le 2 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 100k subset (about one sixteenth of the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_100k exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 3 ]; then
  ivectordir=exp/nnet2_online/ivectors_train
  if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-online/egs/fisher_english/s5/$ivectordir $ivectordir/storage
  fi

  # We extract iVectors on all the train data, which will be what we
  # train the system on.  This version of the iVector-extraction script
  # pairs the utterances into twos (by default, see --utts-per-spk-max option) 
  # and treats each of these pairs as one speaker.
  # Note that these are extracted 'online'.
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    --utts-per-spk-max 2 \
    data/train exp/nnet2_online/extractor $ivectordir || exit 1;
fi


if [ $stage -le 4 ]; then
  if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-online/egs/fisher_english/s5/$dir/egs $dir/egs/storage
  fi

  # Because we have a lot of data here and we don't want the training to take
  # too long, we reduce the number of epochs from the defaults (15 + 5) to (1 +
  # 1).  The option "--io-opts '-tc 12'" is to have more than the default number
  # (5) of jobs dumping the egs to disk; this is OK since we're splitting our
  # data across four filesystems for speed.

  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
    --num-epochs 3 --num-epochs-extra 1 \
    --splice-width 7 --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "-tc 12" \
    --num-jobs-nnet 6 \
    --num-hidden-layers 4 \
    --mix-up 12000 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 3500 \
    --pnorm-output-dim 350 \
    data/train data/lang exp/tri5a $dir  || exit 1;
fi

if [ $stage -le 5 ]; then
  # dump iVectors for the testing data.
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
    data/dev exp/nnet2_online/extractor exp/nnet2_online/ivectors_dev || exit 1;
fi


if [ $stage -le 6 ]; then
  # this does offline decoding that should give about the same results as the
  # real online decoding (the one with --per-utt true)
  steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
       --online-ivector-dir exp/nnet2_online/ivectors_dev \
       exp/tri5a/graph data/dev $dir/decode_dev || exit 1;
fi


if [ $stage -le 7 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang exp/nnet2_online/extractor \
    "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors, carrying info forward from 
  # previous utterances of the same speaker.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      exp/tri5a/graph data/dev ${dir}_online/decode_dev || exit 1;
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
     --per-utt true \
      exp/tri5a/graph data/dev ${dir}_online/decode_dev_utt || exit 1;
fi

if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
     --per-utt true --online false \
      exp/tri5a/graph data/dev ${dir}_online/decode_dev_utt_offline || exit 1;
fi

exit 0;


#Baseline: GMM+SAT system.
#%WER 31.07 [ 12163 / 39141, 1869 ins, 2705 del, 7589 sub ] exp/tri5a/decode_dev/wer_13

# Baseline: p-norm system on top of fMLLR features.
#%WER 23.66 [ 9259 / 39141, 1495 ins, 2432 del, 5332 sub ] exp/nnet6c4_gpu/decode_dev/wer_11

# Our experiment, carrying forward the adaptation state between
# utterances of each speaker.
#%WER 23.79 [ 9311 / 39141, 1499 ins, 2277 del, 5535 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_dev/wer_11


# Our experiment, with per-utterance decoding:
%WER 24.84 [ 9721 / 39141, 1445 ins, 2410 del, 5866 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_dev_utt/wer_11


 # below, with --max-chunks-at-once 3.  The WER is slightly worse but I expect in general it will
 # be slightly better, to to more iVector right context; this is likely just noise.  The average
 # latency was reduced vs the baseline,
 #%WER 24.92 [ 9753 / 39141, 1423 ins, 2429 del, 5901 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_dev_utt_mc3/wer_11


# The following results (obtained after ./run_nnet2_discriminative.sh was run), show
# the effect of discriminative training.  After 2 epochs, we reduce the WER from 23.58 to 22.07.
%WER 23.58 [ 9229 / 39141, 1382 ins, 2400 del, 5447 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_dev/wer_12
%WER 22.16 [ 8675 / 39141, 1522 ins, 1886 del, 5267 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_dev_smbr_epoch1/wer_13
%WER 22.07 [ 8637 / 39141, 1540 ins, 1873 del, 5224 sub ] exp/nnet2_online/nnet_a_gpu_online/decode_dev_smbr_epoch2/wer_13
