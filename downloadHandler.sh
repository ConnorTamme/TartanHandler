#!/bin/bash
#$1 = batch number, $2 = start index, $3 = end index
mkdir ./tmp/data/b$1

for ((i=$2;i<$3;i++));
do
    ./downloadFileX $1 $i
    pids="$pids $!"
done
wait $pids
