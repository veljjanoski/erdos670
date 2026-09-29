#!/bin/bash
# Remove one point of a set in turn and re-insert it by grid start (12 workers).  Usage: bash reinsert.sh SRC TAG [step]
cd "$(dirname "$0")"; SRC=$1; TAG=$2; STEP=${3:-1}
python drop_one.py "$SRC" ri_$TAG.jsonl
N=$(wc -l < ri_$TAG.jsonl)
for ((i=0; i<N; i++)); do bash gp_run.sh 12 ri_$TAG.jsonl:$i 1 $STEP 0.4 0.4 0 ${TAG}_$i 95; done
echo DONE > ri_$TAG.done
