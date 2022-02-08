#!/bin/bash -e

[ -d "$HOME"/.ir_datasets ] || mkdir "$HOME"/.ir_datasets
srun \
  --cpus-per-task 1 \
  --mem=100G \
  --container-writable \
  --container-image=python:3.8.10 \
  --container-name=sigir22-ir-axioms-02 \
  --pty \
  --container-mounts="$PWD":/workspace,"$HOME"/.ir_datasets:/root/.ir_datasets \
  --chdir "$PWD" \
  bash -c "cd /workspace && python -m pip install --upgrade pip && python -m pip install -e . -e .[pyterrier] -e .[test] && jupyter-lab --ip 0.0.0.0 --allow-root"
