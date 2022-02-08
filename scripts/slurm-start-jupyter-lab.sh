#!/bin/bash -e

[ -d "$HOME"/.ir_datasets ] || mkdir "$HOME"/.ir_datasets
srun \
  --cpus-per-task 1 \
  --mem=100G \
  --container-writable \
  --container-image=python:3.8.10 \
  --container-name=sigir22-ir-axioms-"$USER" \
  --pty \
  --container-mounts="$PWD":/workspace,"$HOME"/.ir_datasets:/root/.ir_datasets \
  --chdir "$PWD" \
  bash -c "cd /workspace &&
    apt-get -y update &&
    apt-get -y install git openjdk-11-jdk &&
    python -m pip install --upgrade pip &&
    python -m pip install -e . &&
    python -m pip install -e .[pyterrier] &&
    python -m pip install -e .[test] &&
    jupyter-lab --ip 0.0.0.0 --allow-root"
