#!/bin/bash -e

# This image has no jupyter-lab installed by default.
# For the first start, please replace the command `bash -c '...'` with bash
# so that your job just starts a bash in the container.
# Then you can install jupyter-lab with pip, and end the bash command by executing the exit command (changes like this installation are persisted)
# For subsequent executions, you can use the command as initially defined `bash -c '...'`

srun -c 1  --mem=100G \
	--container-writable \
	--container-image=pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel \
	--container-name=sigir22-ir-axioms-02 --pty \
	--container-mounts=${PWD}:/workspace,/mnt/ceph/storage/data-tmp/2021/kibi9872/.ir_datasets:/root/.ir_datasets \
	--chdir ${PWD} \
	bash -c 'cd /workspace && jupyter-lab --ip 0.0.0.0 --allow-root'

