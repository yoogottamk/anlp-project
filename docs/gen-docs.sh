#!/bin/bash

function mkfile {
    mkdir -p $( dirname $1 )
    touch $1
}

# go to repo root
cd "$( git rev-parse --show-toplevel )"
# get all python files
pyfiles=$( find anlp_project -name '*.py' ! -name '*__*' | sed 's;.py$;;' )

# come back to docs dir
cd docs

for file in $pyfiles; do
    # if file doesn't already exist
    if [[ ! -s ${file}.md ]]; then
        # create dir and file
        mkfile ${file}.md
        echo "::: ${file//\//\.}" > ${file}.md
    fi
done

python populate-nav.py
