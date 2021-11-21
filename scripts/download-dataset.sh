#!/bin/bash

project_dir="$( dirname "$( realpath "$0" )" )/.."

DS_URL="https://www.statmt.org/europarl/v9/training/europarl-v9.de-en.tsv.gz"
DS_HASH="d7be6898ae6ef9173222b355b43229f1"
DS_DIR=${DS_DIR:-${project_dir}/dataset}
DS_LOC=${DS_DIR}/de-en.tsv.gz

mkdir -p "${DS_DIR}/intermediate"

while [[ $( md5sum "$DS_LOC" | awk '{ print $1 }' ) != "$DS_HASH" ]]; do
    echo "Checksum mismatch or file not present. Downloading..."
    wget -O "$DS_LOC" "$DS_URL"
    echo "Dataset downloaded. Checking hash..."
done

echo "Dataset downloaded successfully. Extracting..."

pushd "$DS_DIR" || exit
pv -F "Extracting %r %p [%e]" de-en.tsv.gz | gunzip > intermediate/de-en.tsv

pv -F "Removing lines with multiple tabs %r %p [%e]" intermediate/de-en.tsv | grep -vE "	.*	" > intermediate/de-en-1tab.tsv
pv -F "Removing lines with NULL values %r %p [%e]" intermediate/de-en-1tab.tsv | grep -vE "(^	|	$)" > intermediate/de-en-1tab-nonull.tsv
echo "Adding header..."
cat - intermediate/de-en-1tab-nonull.tsv <<< "de	en" > intermediate/de-en-1tab-header.tsv
popd || exit

echo "Preproceesing..."
python scripts/lower_normalize.py "${DS_DIR}/intermediate/de-en-1tab-header.tsv"

pushd "$DS_DIR" || exit
rm dataset.sqlite
echo "Importing as sqlite db..."
sqlite3 dataset.sqlite <<EOF
.mode tabs
.import intermediate/de-en-final.tsv dataset
EOF
popd || exit

# anything else that needs to run
