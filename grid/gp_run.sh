#!/bin/bash
# Run gridpolish.py with W parallel workers and wait.  Usage: bash gp_run.sh W BASE lam step padx pady half tag thr
cd "$(dirname "$0")"
W=$1; shift; BASE=$1; LAM=$2; STEP=$3; PX=$4; PY=$5; HALF=$6; TAG=$7; THR=$8
for ((w=0; w<W; w++)); do python gridpolish.py "$BASE" $LAM $STEP $PX $PY $HALF $w $W $TAG $THR > gp_${TAG}_$w.log 2>&1 & done
wait; echo "ALLDONE $TAG" >> gp_$TAG.log
