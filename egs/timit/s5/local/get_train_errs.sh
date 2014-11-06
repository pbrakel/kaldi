#!/bin/bash

stage=0 #restart with --stage n
decode_nj=4
decode_cmd=run.pl

utils/parse_options.sh || exit 1;


if [ $stage -le 0 ]; then
    for subset in "train" "dev" "test"
    do
	steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
            exp/tri3/graph data/$subset exp/tri3/decode_${subset}
    done
fi

if [ $stage -le 1 ]; then

    for subset in "train" "dev" "test"
    do
	steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd" \
            --transform-dir exp/tri3/decode_${subset} exp/sgmm2_4/graph data/${subset} \
	    exp/sgmm2_4/decode_${subset}
    done
fi

data_fmllr=data-fmllr-tri3
dir=exp/dnn4_pretrain-dbn_dnn_smbr
gmmdir=exp/tri3
acwt=0.2
if [ $stage -le 2 ]; then
    for subset in "train" "dev" "test"
    do
	for ITER in 1 6; do
	    steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
		--nnet $dir/${ITER}.nnet --acwt $acwt \
		$gmmdir/graph $data_fmllr/${subset} $dir/decode_${subset}_it${ITER} || exit 1
	done
    done
fi
