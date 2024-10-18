#!/usr/bin/env bash

## Arguments:
## 1. destination directory
## 2. [python version]
## 3. [Requirements.txt] if not specified default_requirements.txt is used

SCRIPT_DIR="$( cd -- "$(dirname ${0})" >/dev/null 2>&1 && pwd )"

${SCRIPT_DIR}/make_venv.sh ${1} ${2}

cp ${SCRIPT_DIR}/default_requirements.txt ${1}/requirements.txt
(cd ${1} && pwd)
(cd ${1} && source env/bin/activate && pip install -r requirements.txt && pip freeze > requirements.txt)