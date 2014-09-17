#!/bin/bash

decode_nj=4
decode_cmd=run.pl

for subset in "train" "dev" "test"
do
  steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd"\
    exp/tri3/graph data/$subset exp/tri3/decode_${subset}
done

for subset in "train" "dev" "test"
do
  steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"\
    --transform-dir exp/tri3/decode_${subset} exp/sgmm2_4/graph data/${subset} \
    exp/sgmm2_4/decode_${subset}
done

data_fmllr=data-fmllr-tri3
dir=exp/dnn4_pretrain-dbn_dnn_smbr
acwt=0.2
for subset in "train" "dev" "test"
do
  for ITER in 1 6; do
    steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph $data_fmllr/${subset} $dir/decode_${subset}_it${ITER} || exit 1
  done
done
