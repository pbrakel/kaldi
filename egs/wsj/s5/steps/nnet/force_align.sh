#!/bin/bash
# Copyright 2014 Jan Chorowski
# Apache 2.0.

# Compute forced alignments.

# Begin configuration section.
nj=4
cmd=run.pl
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/$0 [options] <data-dir> <model-dir> <lang-dir> <log-dir>"
   echo "  e.g.: steps/$0 data/train data/lang exp/tri1 exp/tri1_denlats"
   echo "Works for plain features (or CMN, delta), forwarded through feature-transform."
   echo ""
   echo "Main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --sub-split <n-split>                            # e.g. 40; use this for "
   echo "                           # large databases so your jobs will be smaller and"
   echo "                           # will (individually) finish reasonably soon."
   exit 1;
fi

data=$1
modeldir=$2
lang=$3
logdir=$4

sdata=$data/split$nj
mkdir -p $logdir
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

#echo $nj > $dir/num_jobs

oov=`cat $lang/oov.int` || exit 1;



if [ true ]; then
  echo "$0: Compiling graphs of transcripts"
  $cmd JOB=1:$nj $logdir/compile_graphs.JOB.log \
compile-train-graphs $modeldir/tree $modeldir/final.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
      "ark:|gzip -c >$data/fsts.JOB.gz" || exit 1;
fi

if [ true ]; then
  echo "$0: Aligning data"
  mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $modeldir/final.mdl - |"
  $cmd JOB=1:$nj $logdir/align.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl" \
    "ark:gunzip -c $data/fsts.JOB.gz|" "scp:$sdata/JOB/feats.scp" \
    "ark:|gzip -c >$data/ali.JOB.gz" || exit 1;
fi

for n in $(seq 1 $nj); do
  gunzip -c $data/ali.$n.gz 
done | gzip -c > $data/ali.gz

echo "$0 finished aligning... $data"
