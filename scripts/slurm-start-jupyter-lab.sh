#!/bin/bash -e

[ -d "$HOME"/.ir_datasets ] || mkdir "$HOME"/.ir_datasets
srun \
  --cpus-per-task 1 \
  --mem=100G \
  --container-writable \
  --container-image=alpine:3.15 \
  --container-name=sigir22-ir-axioms-"$USER" \
  --pty \
  --container-mounts="$PWD":/workspace,"$HOME"/.ir_datasets:/root/.ir_datasets \
  --chdir "$PWD" \
  bash -c "cd /workspace &&
    apk update &&
    apk upgrade &&
    apk add bash unzip curl openjdk7-jre python3
    python3 -m ensurepip &&
    pip3 install --upgrade pip setuptools &&
    pip3 install -e . &&
    pip3 install -e .[pyterrier] &&
    pip3 install -e .[test] &&
    jupyter-lab --ip 0.0.0.0 --allow-root"
