#!/usr/bin/env bash

## Pass directory as first argument and optionally pass python version (for scripts/make_venv.sh venv_example 3.12)

if [[ $# -lt 1 ]]; then
    >&2 echo "Wrong number of arguments " $@
    exit 1;
fi

if [ -f ${1} ]; then
    >&2 echo "Path specifies file, instead of directory"
    exit 1;
fi

if [ ! -d ${1} ]; then
    echo "$DIRECTORY does exist. Creating..."
    mkdir ${1}
fi
echo "Creating venv"
(cd ${1} && /usr/bin/env "python${2:-3}" -m venv env)
