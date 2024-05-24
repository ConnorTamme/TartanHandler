#!/bin/bash
#$1 = batch number, $2 = start index, $3 = end index
mkdir ./tmp/data/B$1

for ((i=$2;i<$3;i++));
do
    ./utils/downloadFileX.sh $1 $i &
    pids="$pids $!"
done
wait $pids
