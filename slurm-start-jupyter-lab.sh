#!/bin/bash -e

srun \
  --cpus-per-task 1 \
  --mem=100G \
	--container-writable \
	--container-image=python:3.8.10 \
	--container-name=sigir22-ir-axioms-02 \
	--pty \
	--container-mounts="$PWD":/workspace,~/.ir_datasets:/root/.ir_datasets \
	--chdir "$PWD" \
	bash -c "cd /workspace && pip install .[pyterrier,test] && jupyter-lab --ip 0.0.0.0 --allow-root"
