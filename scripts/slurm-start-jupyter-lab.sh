#!/bin/bash -e

[ -d "$HOME"/.ir_datasets ] || mkdir "$HOME"/.ir_datasets
srun \
  --cpus-per-task 1 \
  --mem=100G \
  --container-writable \
  --container-image=python:3.9-buster \
  --container-name=sigir22-ir-axioms-"$USER"-python3.9-jdk11 \
  --container-mounts="$PWD":/workspace,"$HOME"/.ir_datasets:/root/.ir_datasets \
  --chdir "$PWD" \
  --pty \
  su -c "cd /workspace &&
    apt update -y &&
    apt install -y openjdk-11-jdk &&
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/ &&
    pip install cython &&
    pip install --editable .[pyterrier,examples,test] &&
    jupyter-lab --ip 0.0.0.0 --allow-root"
