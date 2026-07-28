#! /usr/bin/env sh

set -euo pipefail

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <username> <password>"
  exit 1
fi

kubectl -n core-ns create secret docker-registry registry-secret --docker-server=https://dxb.ocir.io --docker-username=$1 --docker-password=$2