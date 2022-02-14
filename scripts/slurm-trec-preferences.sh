#!/bin/bash -e

[ -d "$HOME"/.ssh ] || mkdir "$HOME"/.ssh
[ -d "$HOME"/.ir_datasets ] || mkdir "$HOME"/.ir_datasets
[ -d "$HOME"/.magnitude ] || mkdir "$HOME"/.magnitude
srun \
  --cpus-per-task 5 \
  --mem=100G \
  --container-writable \
  --container-image=python:3.9-buster \
  --container-name=sigir22-ir-axioms-"$USER"-python3.9-jdk11 \
  --container-mounts="$PWD":/workspace,"$HOME"/.ssh:/root/.ssh,"$HOME"/.ir_datasets:/root/.ir_datasets,"$HOME"/.magnitude:/root/.magnitude \
  --chdir "$PWD" \
  --pty \
  su -c "cd /workspace &&
    apt update -y &&
    apt install -y openjdk-11-jdk &&
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/ &&
    pip install cython &&
    pip install --editable .[pyterrier,examples,test] &&
    python examples/trec_preferences.py"
