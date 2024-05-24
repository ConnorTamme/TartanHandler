#!/bin/bash
#$1 = batch number, $2 = index of file to download

mkdir ./tmp/newData/F$1$2/
mkdir ./tmp/data/B$1/F$1$2/

python ./utils/download_training.py $2 F$1$2 &
pid=$!
wait $pid

unzip -q ./tmp/newData/F$1$2/*.zip -d ./tmp/data/B$1/F$1$2/ &
pid=$!
wait $pid

rm -rf ./tmp/newData/F$1$2/
